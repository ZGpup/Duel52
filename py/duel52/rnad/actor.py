"""The actor: a batch of Duel 52 games played by sampling the network — ``PLAN.md`` item 8.

The engine's ``GameBatch`` holds the games and does everything that is rules: it writes each
waiting game's observation and legal mask, and it decodes and legality-checks the action
indices it is handed, playing forced moves itself. This module only evaluates the network on
those rows, samples, and keeps what the learner needs.

**Strictly on-policy, like the reference.** The actor plays with the live online network and
the learner steps on exactly those games, so the behaviour policy ``μ`` is the policy being
learned. ``μ(a)`` is still recorded, because V-trace's per-action estimate divides by it.

**One uniform per decision**, drawn from a numpy generator on the CPU and sampled on the
device by inverse CDF. The stream therefore does not depend on the device, and a game's moves
depend only on the network, the seed and the step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .._engine import GameBatch
from ..train.config import GameSettings
from .core import legal_policy

__all__ = ["Trajectories", "play_batch"]


@dataclass
class Trajectories:
    """Every decision of a batch of games, flat, in game-then-time order per tick.

    ``game[i]`` and ``t[i]`` place decision ``i`` in its game; ``player[i]`` is who acted and
    whose point of view ``obs[i]`` is encoded from. ``returns[g]`` is ``[p0, p1]`` on
    ``-1..1``, received on the transition out of game ``g``'s last decision.
    """

    obs: torch.Tensor  # [N, obs_dim] float32
    legal: torch.Tensor  # [N, action_dim] bool
    action: torch.Tensor  # [N] int64
    mu: torch.Tensor  # [N] float32
    player: torch.Tensor  # [N] int64
    game: torch.Tensor  # [N] int64
    t: torch.Tensor  # [N] int64
    lengths: np.ndarray  # [games] int64
    returns: np.ndarray  # [games, 2] float64
    outcomes: list[str] = field(default_factory=list)
    plies: list[int] = field(default_factory=list)

    @property
    def decisions(self) -> int:
        return int(self.action.shape[0])

    @property
    def games(self) -> int:
        return int(self.lengths.shape[0])


def sample_actions(
    probs: torch.Tensor, uniforms: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse-CDF sampling, one uniform per row. Returns ``(action, prob of action)``.

    ``searchsorted(right=True)`` returns the first index whose cumulative probability exceeds
    the draw, which can never land on a zero-probability action.
    """
    cdf = probs.cumsum(dim=-1)
    target = (uniforms * cdf[:, -1]).unsqueeze(1)
    action = torch.searchsorted(cdf, target, right=True).squeeze(1)
    action = action.clamp_(max=probs.shape[1] - 1)
    mu = probs.gather(1, action[:, None]).squeeze(1)
    # Floating-point residue at the very top of the CDF: fall back to the likeliest action.
    bad = mu <= 0
    if bool(bad.any()):
        action = torch.where(bad, probs.argmax(dim=1), action)
        mu = probs.gather(1, action[:, None]).squeeze(1)
    return action, mu


@torch.no_grad()
def play_batch(
    model: torch.nn.Module,
    *,
    games: int,
    seed: int,
    rng: np.random.Generator,
    game: GameSettings,
    device: torch.device,
    threads: int = 0,
    chunk: int = 0,
) -> Trajectories:
    """Play ``games`` games to the end with ``model`` on both seats."""
    batch = GameBatch(
        games=games,
        seed=seed,
        variant=None if game.rules_file else game.variant,
        rules_file=game.rules_file,
        encoding_slots=game.encoding_slots,
        two_power=game.two_power,
        stalemate=game.stalemate,
        stalemate_value=game.stalemate_value,
        threads=threads,
    )
    od, ad = batch.obs_dim, batch.action_dim
    obs_buf, mask_buf = bytearray(games * od * 4), bytearray(games * ad)
    ids_buf, players_buf = bytearray(games * 4), bytearray(games)
    obs_np = np.frombuffer(obs_buf, dtype=np.float32).reshape(games, od)
    mask_np = np.frombuffer(mask_buf, dtype=np.uint8).reshape(games, ad)
    ids_np = np.frombuffer(ids_buf, dtype=np.uint32)
    players_np = np.frombuffer(players_buf, dtype=np.uint8)

    was_training = model.training
    model.eval()
    parts: dict[str, list[torch.Tensor]] = {k: [] for k in ("obs", "legal", "action", "mu", "player", "game", "t")}
    counter = np.zeros(games, dtype=np.int64)

    while not batch.done():
        k = batch.observe(obs_buf, mask_buf, ids_buf, players_buf)
        ids = ids_np[:k].astype(np.int64)
        # `.to(device)` copies off the reused buffers on an accelerator; on the CPU it would
        # not, so clone explicitly there.
        obs = torch.from_numpy(obs_np[:k]).to(device)
        legal = torch.from_numpy(mask_np[:k]).to(device).bool()
        if obs.device.type == "cpu":
            obs = obs.clone()

        step = k if chunk <= 0 else chunk
        actions, mus = [], []
        uniforms = torch.from_numpy(rng.random(k)).to(device=device, dtype=torch.float32)
        for lo in range(0, k, step):
            hi = min(k, lo + step)
            logits, _ = model(obs[lo:hi])
            probs = legal_policy(logits.float(), legal[lo:hi])
            a, m = sample_actions(probs, uniforms[lo:hi])
            actions.append(a)
            mus.append(m)
        action = torch.cat(actions)
        mu = torch.cat(mus)

        action_np = action.cpu().numpy().astype(np.uint32)
        batch.apply(ids_np[:k].tobytes(), action_np.tobytes())

        parts["obs"].append(obs)
        parts["legal"].append(legal)
        parts["action"].append(action)
        parts["mu"].append(mu)
        parts["player"].append(torch.from_numpy(players_np[:k].astype(np.int64)).to(device))
        parts["game"].append(torch.from_numpy(ids).to(device))
        parts["t"].append(torch.from_numpy(counter[ids]).to(device))
        counter[ids] += 1

    model.train(was_training)
    lengths = np.asarray(batch.decisions(), dtype=np.int64)
    assert np.array_equal(lengths, counter), "the engine and the actor disagree about game length"
    return Trajectories(
        **{k: torch.cat(v) for k, v in parts.items()},
        lengths=lengths,
        returns=np.frombuffer(batch.returns(), dtype=np.float32).reshape(games, 2).astype(np.float64),
        outcomes=batch.outcomes(),
        plies=batch.plies(),
    )
