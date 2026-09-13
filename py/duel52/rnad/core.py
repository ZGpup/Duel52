"""R-NaD's arithmetic, ported from the reference — ``PLAN.md`` item 8.

The reference is OpenSpiel's ``open_spiel/python/algorithms/rnad/rnad.py`` at commit
``d1dcdf5d`` (removed from master in ``e165bbdd``). Every function here names the reference
function it ports, and ``py/tests/test_rnad.py`` checks each against outputs that the
reference itself produced (``py/tests/fixtures/rnad_reference.npz``).

One structural change, and it is the reason this file looks different from the reference
while computing the same numbers. The reference's V-trace scans ``[T, B, A]`` tensors; here
the backwards scan carries **scalars only** and returns, per step, the one coefficient the
per-action ``learning_output`` needs::

    learning_output[t, a] = v[t] + eta_log_policy[t, a] + onehot(a_t)[a] · coef[t]

Every per-action term is then formed in flat ``[N, A]`` form over the steps that exist,
rather than over ``T × B × 2194`` padded cells — the difference between megabytes and
gigabytes at Duel 52's action width. ``learning_output_from`` rebuilds the reference's full
tensor for the parity test.

Shapes: ``T`` time steps (decisions, both players interleaved in game order), ``B`` games,
``A`` actions. A step is *own* for player ``p`` when it is valid and ``p`` is to act there.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

__all__ = [
    "EntropySchedule",
    "legal_policy",
    "legal_log_policy",
    "post_process_policy",
    "v_trace",
    "learning_output_from",
    "nerd_advantage",
    "nerd_loss",
]


# ----------------------------------------------------------------- the schedule --


class EntropySchedule:
    """The reference ``EntropySchedule``: when the regularisation policy moves, and ``alpha``.

    ``sizes`` and ``repeats`` are parallel: ``EntropySchedule([3, 5, 10], [2, 4, 1])`` updates
    at learner steps ``0, 3, 6, 11, 16, 21, 26, 36``, and the last size repeats forever.
    ``alpha`` ramps from 0 to 1 over the first half of each iteration and blends the two most
    recent regularisation policies, so the transformed reward moves smoothly across an update.
    """

    def __init__(self, *, sizes: Sequence[int], repeats: Sequence[int]):
        sizes, repeats = list(sizes), list(repeats)
        if len(repeats) != len(sizes):
            raise ValueError("entropy schedule: repeats must be parallel to sizes")
        if not sizes:
            raise ValueError("entropy schedule: sizes must not be empty")
        if any(r <= 0 for r in repeats):
            raise ValueError("entropy schedule: every repeat must be positive")
        if repeats[-1] != 1:
            raise ValueError("entropy schedule: the last repeat must be 1 — it repeats forever")
        if any(s <= 0 for s in sizes):
            raise ValueError("entropy schedule: every size must be positive")
        schedule = [0]
        for size, repeat in zip(sizes, repeats):
            # A list, not a generator: `schedule[-1]` must be read once per group, before the
            # extend starts appending to it.
            last = schedule[-1]
            schedule.extend([last + (i + 1) * size for i in range(repeat)])
        self.schedule = schedule

    def __call__(self, learner_step: int) -> tuple[float, bool]:
        """``(alpha, update_target_net)`` for ``learner_step`` (counted before the step)."""
        s = self.schedule
        if learner_step >= s[-1]:
            size = s[-1] - s[-2]
            start = s[-1] + (learner_step - s[-1]) // size * size
        else:
            start = max(x for x in s if x <= learner_step)
            finish = min(x for x in s if x > learner_step)
            size = finish - start
        update = learner_step > 0 and learner_step == start + size - 1
        alpha = min(2.0 * (learner_step - start) / size, 1.0)
        return alpha, update

    def iterations_by(self, learner_step: int) -> int:
        """How many regularisation updates have happened by ``learner_step`` steps."""
        return sum(1 for n in range(learner_step) if self(n)[1])


# ------------------------------------------------------------------- policies --


def legal_policy(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """The reference ``_legal_policy``: a softmax over the legal actions, 0 elsewhere."""
    legal_f = legal.to(logits.dtype)
    l_min = logits.min(dim=-1, keepdim=True).values
    logits = torch.where(legal, logits, l_min)
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logits = logits * legal_f
    exp = torch.where(legal, torch.exp(logits), torch.zeros_like(logits))
    return exp / exp.sum(dim=-1, keepdim=True)


def legal_log_policy(logits: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """The reference ``legal_log_policy``: log of the legal softmax, 0 on illegal actions."""
    legal_f = legal.to(logits.dtype)
    masked = logits + torch.log(legal_f)
    max_legal = masked.max(dim=-1, keepdim=True).values
    baseline = torch.log(torch.exp(masked - max_legal).sum(dim=-1, keepdim=True))
    return legal_f * (logits - max_legal - baseline)


def post_process_policy(
    policy: np.ndarray, legal: np.ndarray, threshold: float = 0.03, discretization: int = 32
) -> np.ndarray:
    """The reference ``FineTuning.post_process_policy``, row by row.

    Drop actions under ``threshold`` (unless every action is under it) and renormalise, then
    round to multiples of ``1 / discretization``. The Rust ``netsample`` agent carries the
    same algorithm; this copy exists so both can be checked against the reference.
    """
    policy = np.asarray(policy, dtype=np.float64)
    legal = np.asarray(legal, dtype=np.float64)
    out = np.empty_like(policy)
    for row in range(policy.shape[0]):
        p = policy[row]
        if threshold > 0:
            keep = legal[row] * ((p >= threshold) | (p.max() < threshold))
            p = keep * p / (keep * p).sum()
        if discretization > 0:
            roundup = np.ceil(p * discretization).astype(np.int64)
            result = np.zeros_like(roundup)
            left = discretization
            for k in np.argsort(-p, kind="stable"):
                x = min(roundup[k], left)
                result[k] += x
                left -= x
            if left > 0:
                result[np.argsort(-p, kind="stable")[0]] += left
            p = result / discretization
        out[row] = p
    return out


# --------------------------------------------------------------------- V-trace --


def v_trace(
    *,
    v: np.ndarray,
    valid: np.ndarray,
    player_id: np.ndarray,
    policy_ratio: np.ndarray,
    inv_mu: np.ndarray,
    kl: np.ndarray,
    reward: np.ndarray,
    player: int,
    eta: float,
    lambda_: float = 1.0,
    c: float = 1.0,
    rho: float = np.inf,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The reference ``v_trace``, as a scan over scalars. All inputs ``[T, B]``.

    ``v`` is the *target* network's value; ``policy_ratio`` is ``π(a_t)/μ(a_t)`` for the
    action taken; ``inv_mu`` is ``1/μ(a_t)``; ``kl`` is ``Σ_a π(a)·log(π(a)/π_reg(a))``;
    ``reward`` is ``player``'s reward on the transition out of step ``t``.

    Returns ``(v_target, has_played, coef)``, each ``[T, B]``: the value target and the
    ``learning_output`` coefficient at ``player``'s own steps, zero elsewhere.
    """
    gamma = 1.0
    valid = valid.astype(bool)
    own_all = valid & (player_id == player)
    opp_all = valid & (player_id != player)
    others = np.where(own_all, 1.0, np.where(opp_all, -1.0, 0.0))
    eta_reg_entropy = -eta * kl * others

    T, B = v.shape
    carry_reward = np.zeros(B)
    carry_ru = np.zeros(B)
    next_value = np.zeros(B)
    next_vt = np.zeros(B)
    importance = np.ones(B)
    v_target = np.zeros((T, B))
    coef = np.zeros((T, B))

    for t in range(T - 1, -1, -1):
        own, opp = own_all[t], opp_all[t]
        cs = policy_ratio[t]
        reward_uncorrected = reward[t] + gamma * carry_ru + eta_reg_entropy[t]
        discounted_reward = reward[t] + gamma * carry_reward
        weight = cs * importance
        our_vt = (
            v[t]
            + np.minimum(rho, weight) * (reward_uncorrected + gamma * next_value - v[t])
            + lambda_ * np.minimum(c, weight) * gamma * (next_vt - next_value)
        )
        our_coef = inv_mu[t] * (discounted_reward + gamma * importance * next_vt - v[t])
        v_target[t] = np.where(own, our_vt, 0.0)
        coef[t] = np.where(own, our_coef, 0.0)

        # The three carries: our step resets the running sums, an opponent's step
        # accumulates them, an invalid step returns everything to the initial state.
        carry_reward = np.where(own, 0.0, np.where(opp, eta_reg_entropy[t] + cs * discounted_reward, 0.0))
        carry_ru = np.where(own, 0.0, np.where(opp, reward_uncorrected, 0.0))
        next_value, next_vt, importance = (
            np.where(own, v[t], np.where(opp, gamma * next_value, 0.0)),
            np.where(own, our_vt, np.where(opp, gamma * next_vt, 0.0)),
            np.where(own, 1.0, np.where(opp, cs * importance, 1.0)),
        )

    # The reference's `_has_played` carries a zero that nothing ever sets, so it reduces to
    # "valid and this player's move".
    return v_target, own_all.astype(np.float64), coef


def learning_output_from(
    *,
    v: np.ndarray,
    coef: np.ndarray,
    has_played: np.ndarray,
    merged_log_policy: np.ndarray,
    actions_oh: np.ndarray,
    eta: float,
) -> np.ndarray:
    """Rebuild the reference's ``[T, B, A]`` ``learning_output`` from :func:`v_trace`'s coef."""
    own = has_played[..., None]
    return own * (v[..., None] - eta * merged_log_policy + actions_oh * coef[..., None])


# ----------------------------------------------------------------------- NeuRD --


def nerd_advantage(
    *,
    pi: torch.Tensor,
    log_ratio: torch.Tensor,
    action: torch.Tensor,
    coef: torch.Tensor,
    eta: float,
    clip: float,
) -> torch.Tensor:
    """``q − Σ π·q`` for each own step, from the decomposed ``learning_output``. ``[N, A]``.

    With ``q = v − η·log_ratio + onehot(a)·coef``, the value ``v`` cancels:
    ``adv = −η·(log_ratio − Σ π·log_ratio) + coef·(onehot(a) − π(a))``. Clipped and detached,
    as the reference's ``get_loss_nerd`` does.
    """
    reg = log_ratio - (pi * log_ratio).sum(dim=-1, keepdim=True)
    onehot = torch.zeros_like(pi).scatter_(1, action[:, None], 1.0)
    pi_a = pi.gather(1, action[:, None])
    adv = -eta * reg + coef[:, None] * (onehot - pi_a)
    return adv.clamp(-clip, clip).detach()


def nerd_loss(
    *,
    logits: torch.Tensor,
    advantage: torch.Tensor,
    legal: torch.Tensor,
    weight: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """The reference ``get_loss_nerd``, summed over steps with per-step ``weight``.

    ``weight`` is ``1 / (own steps of the acting player)`` so the sum is the reference's
    per-player ``renormalize`` summed over players. The logits are centred on their legal
    mean, and the force pushes each only while it is within ``±beta`` of zero.
    """
    legal_f = legal.to(logits.dtype)
    mean = (logits * legal_f).sum(dim=-1, keepdim=True) / legal_f.sum(dim=-1, keepdim=True)
    centred = logits - mean
    can_decrease = (centred > -beta).to(logits.dtype)
    can_increase = (centred < beta).to(logits.dtype)
    force = can_decrease * advantage.clamp(max=0.0) + can_increase * advantage.clamp(min=0.0)
    per_step = (legal_f * centred * force.detach()).sum(dim=-1)
    return -(per_step * weight).sum()
