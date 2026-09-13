"""One R-NaD learner step — ``PLAN.md`` item 8.

The reference's ``RNaDSolver.loss`` and ``update_parameters``, in PyTorch, over the flat
:class:`~duel52.rnad.actor.Trajectories` the actor produces.

Four networks, initialised identically as the reference does:

* ``online``   — trained, and the one the actor plays.
* ``target``   — ``target ← target + τ·(online − target)`` after every step. The shipped net,
  and the value baseline inside V-trace.
* ``reg``, ``reg_prev`` — the two most recent regularisation policies (the reference's
  ``params_prev`` and ``params_prev_``). At the end of each schedule iteration
  ``reg_prev ← reg`` and ``reg ← target``.

A step is two passes over the batch:

1. **No gradient.** Online policy, target value and both regularisation policies, giving
   ``log_ratio = log π − (α·log π_reg + (1−α)·log π_reg_prev)``. V-trace then runs on the CPU
   (``core.v_trace`` carries scalars only) for each player.
2. **Gradient.** The online net again, with the value loss against the V-trace targets and the
   NeuRD loss against the advantages from pass 1, accumulated chunk by chunk so a batch that
   does not fit on the device still takes exactly the same step.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .actor import Trajectories
from .config import RNaDSettings
from .core import EntropySchedule, legal_log_policy, legal_policy, nerd_advantage, nerd_loss, v_trace

__all__ = ["RNaDLearner", "StepStats"]


@dataclass
class StepStats:
    step: int = 0
    alpha: float = 0.0
    updated_reg: bool = False
    reg_updates: int = 0
    loss_value: float = 0.0
    loss_nerd: float = 0.0
    v_target_min: float = 0.0
    v_target_max: float = 0.0
    policy_entropy: float = 0.0
    kl_to_reg: float = 0.0
    ratio_error: float = 0.0
    positions: int = 0
    seconds: float = 0.0


class RNaDLearner:
    def __init__(self, model: nn.Module, settings: RNaDSettings, device: torch.device):
        self.settings = settings
        self.device = device
        self.online = model.to(device)
        self.target = copy.deepcopy(self.online)
        self.reg = copy.deepcopy(self.online)
        self.reg_prev = copy.deepcopy(self.online)
        for net in (self.target, self.reg, self.reg_prev):
            net.requires_grad_(False)
            net.eval()
        self.optimizer = torch.optim.Adam(
            self.online.parameters(),
            lr=settings.learning_rate,
            betas=(settings.adam_b1, settings.adam_b2),
            eps=settings.adam_eps,
        )
        self.schedule = EntropySchedule(
            sizes=settings.entropy_schedule_size, repeats=settings.entropy_schedule_repeats
        )
        self.steps = 0
        self.reg_updates = 0

    # ------------------------------------------------------------------- state --

    def state_dict(self) -> dict:
        return {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "reg": self.reg.state_dict(),
            "reg_prev": self.reg_prev.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps": self.steps,
            "reg_updates": self.reg_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.reg.load_state_dict(state["reg"])
        self.reg_prev.load_state_dict(state["reg_prev"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.steps = int(state["steps"])
        self.reg_updates = int(state["reg_updates"])

    # -------------------------------------------------------------------- step --

    def _chunks(self, n: int):
        size = n if self.settings.learner_chunk <= 0 else self.settings.learner_chunk
        return [(lo, min(n, lo + size)) for lo in range(0, n, size)]

    def step(self, traj: Trajectories) -> StepStats:
        started = time.perf_counter()
        cfg = self.settings
        alpha, update = self.schedule(self.steps)
        n = traj.decisions
        chunks = self._chunks(n)

        # ---- pass 1: everything V-trace and NeuRD need, without gradient ----
        v_bar = torch.empty(n, device=self.device)
        pi_a = torch.empty(n, device=self.device)
        kl = torch.empty(n, device=self.device)
        entropy = torch.empty(n, device=self.device)
        pis, log_ratios = [], []
        self.online.eval()
        with torch.no_grad():
            for lo, hi in chunks:
                obs, legal = traj.obs[lo:hi], traj.legal[lo:hi]
                logits, _ = self.online(obs)
                _, value = self.target(obs)
                reg_logits, _ = self.reg(obs)
                prev_logits, _ = self.reg_prev(obs)
                pi = legal_policy(logits, legal)
                log_pi = legal_log_policy(logits, legal)
                log_ratio = log_pi - (
                    alpha * legal_log_policy(reg_logits, legal)
                    + (1.0 - alpha) * legal_log_policy(prev_logits, legal)
                )
                v_bar[lo:hi] = value
                pi_a[lo:hi] = pi.gather(1, traj.action[lo:hi, None]).squeeze(1)
                kl[lo:hi] = (pi * log_ratio).sum(-1)
                entropy[lo:hi] = -(pi * log_pi).sum(-1)
                pis.append(pi)
                log_ratios.append(log_ratio)
        self.online.train()

        # ---- V-trace, per player, on [T, B] scalars ----
        games, t_max = traj.games, int(traj.lengths.max()) if traj.games else 0
        g = traj.game.cpu().numpy()
        t = traj.t.cpu().numpy()
        player = traj.player.cpu().numpy()
        # To the CPU first, then float64: MPS has no float64.
        mu = traj.mu.cpu().double().numpy()

        def grid(values: np.ndarray, fill: float) -> np.ndarray:
            out = np.full((t_max, games), fill)
            out[t, g] = values
            return out

        valid = grid(np.ones(n), 0.0)
        player_grid = grid(player.astype(np.float64), 0.0)
        ratio = grid(pi_a.cpu().double().numpy() / mu, 1.0)
        inv_mu = grid(1.0 / mu, 1.0)
        kl_grid = grid(kl.cpu().double().numpy(), 0.0)
        v_grid = grid(v_bar.cpu().double().numpy(), 0.0)
        last = traj.lengths - 1

        v_target = np.zeros(n)
        coef = np.zeros(n)
        for p in (0, 1):
            reward = np.zeros((t_max, games))
            has = traj.lengths > 0
            reward[last[has], np.nonzero(has)[0]] = traj.returns[has, p]
            vt, _, cf = v_trace(
                v=v_grid, valid=valid, player_id=player_grid, policy_ratio=ratio, inv_mu=inv_mu,
                kl=kl_grid, reward=reward, player=p, eta=cfg.eta_reward_transform,
                c=cfg.c_vtrace, rho=cfg.rho_vtrace,
            )
            mine = player == p
            v_target[mine] = vt[t[mine], g[mine]]
            coef[mine] = cf[t[mine], g[mine]]

        counts = np.bincount(player, minlength=2).astype(np.float64)
        weight_np = 1.0 / np.maximum(counts[player], 1.0)
        v_target_t = torch.as_tensor(v_target, dtype=torch.float32, device=self.device)
        coef_t = torch.as_tensor(coef, dtype=torch.float32, device=self.device)
        weight_t = torch.as_tensor(weight_np, dtype=torch.float32, device=self.device)

        # ---- pass 2: the gradient ----
        self.optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        loss_nerd = 0.0
        for (lo, hi), pi, log_ratio in zip(chunks, pis, log_ratios):
            advantage = nerd_advantage(
                pi=pi, log_ratio=log_ratio, action=traj.action[lo:hi], coef=coef_t[lo:hi],
                eta=cfg.eta_reward_transform, clip=cfg.nerd_clip,
            )
            logits, value = self.online(traj.obs[lo:hi])
            lv = (weight_t[lo:hi] * (value - v_target_t[lo:hi]) ** 2).sum()
            ln = nerd_loss(
                logits=logits, advantage=advantage, legal=traj.legal[lo:hi],
                weight=weight_t[lo:hi], beta=cfg.nerd_beta,
            )
            (lv + ln).backward()
            loss_value += float(lv.detach())
            loss_nerd += float(ln.detach())
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.online.parameters(), cfg.grad_clip)
        self.optimizer.step()

        # ---- the target average, then the regularisation roll ----
        with torch.no_grad():
            for target, online in zip(self.target.parameters(), self.online.parameters()):
                target.add_(online.detach() - target, alpha=cfg.target_network_avg)
        if update:
            self.reg_prev.load_state_dict(self.reg.state_dict())
            self.reg.load_state_dict(self.target.state_dict())
            self.reg_updates += 1
        self.steps += 1

        ratio_error = float(np.max(np.abs(ratio[valid > 0] - 1.0))) if n else 0.0
        return StepStats(
            step=self.steps,
            alpha=alpha,
            updated_reg=update,
            reg_updates=self.reg_updates,
            loss_value=loss_value,
            loss_nerd=loss_nerd,
            v_target_min=float(v_target.min()) if n else 0.0,
            v_target_max=float(v_target.max()) if n else 0.0,
            policy_entropy=float(entropy.mean()) if n else 0.0,
            kl_to_reg=float(kl.mean()) if n else 0.0,
            ratio_error=ratio_error,
            positions=n,
            seconds=time.perf_counter() - started,
        )
