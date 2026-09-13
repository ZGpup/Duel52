"""The fitting half of the loop: batches in, a new checkpoint out.

The loss is AlphaZero's, with the one adaptation the action encoding forces:

``policy`` — cross-entropy against the search's **visit distribution**, which is a soft
target supported on the legal actions (``DESIGN.md`` §6). The log-softmax is taken over the
whole 2195-logit head rather than over the legal subset, so illegal actions are pushed down
as a side effect of every step. That is the standard choice and it is also the safe one
here: masking during training would teach the network nothing about the mask, and the mask
is applied at *play* time by the engine anyway (``engine/src/encode.rs::legal_mask``), which
is the authority.

``value`` — mean squared error against the game's eventual result, in ``-1..=1`` to match
the ``tanh`` head.

Device
------

``CLAUDE.md``: *"Device-agnostic. Code must run on MPS locally and CUDA on a rented box with
no edits beyond a config value."* :func:`resolve_device` is that config value's whole
implementation, and nothing below mentions a device by name.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from ..nn.checkpoint import read_checkpoint, write_checkpoint
from ..nn.model import NetConfig, build_net, lane_spec_for
from .buffer import Generation, ReplayBuffer
from .config import TrainConfig
from .durable import replace_atomically

__all__ = ["Trainer", "StepStats", "EvalStats", "resolve_device"]


def _canonical(value: Any) -> Any:
    """``value`` as it reads back from JSON — tuples as lists — so an identity written into a
    checkpoint compares equal to the same identity rebuilt on resume."""
    return json.loads(json.dumps(value))


def resolve_device(name: str) -> torch.device:
    """``"auto"`` → MPS, else CUDA, else CPU. Anything else is taken literally."""
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class StepStats:
    """Averages over one generation's optimisation steps, for the readout."""

    steps: int = 0
    policy_first: float = 0.0
    policy_last: float = 0.0
    value_first: float = 0.0
    value_last: float = 0.0
    policy_mean: float = 0.0
    value_mean: float = 0.0
    seconds: float = 0.0
    lr: float = 0.0

    @property
    def samples_per_sec(self) -> float:
        return 0.0 if self.seconds <= 0 else self.steps / self.seconds


@dataclass
class EvalStats:
    """One pass over a held-out set. ``PLAN.md`` §4.2 change 7.

    The training-batch value loss is computed on a replay window that slides underneath it,
    so it cannot tell "the value head learned all it can" from "the positions got harder".
    This one is measured on samples that were never trained on and never change.
    """

    samples: int = 0
    policy_loss: float = 0.0
    value_mse: float = 0.0


class Trainer:
    """Owns the model, the optimiser and the device. One instance for a whole run."""

    def __init__(self, config: TrainConfig, spec: dict, checkpoint: Path | None = None):
        self.config = config
        self.spec = spec
        self.device = resolve_device(config.train.device)

        # The lane partition, when the architecture needs it. Built from the engine every
        # time rather than cached on the config — `CLAUDE.md`'s encoder rule, and it costs
        # microseconds.
        # `rules_file` matters here since the encoder reserve (`MODULAR_RULES.md` §7): an
        # extended ruleset has a lane-owned `CHOOSE_LANE` block, so its partition differs
        # from the canonical one and building the net against the wrong table is silent.
        lanes = lambda: lane_spec_for(  # noqa: E731
            config.game.variant, config.game.encoding_slots, config.game.rules_file
        )

        if checkpoint is not None:
            ckpt = read_checkpoint(checkpoint)
            ckpt.check_against(spec)
            # The checkpoint's own `arch`, not the config's: after generation 1 the shape
            # comes from the file, which is the only thing that can be right about it.
            net_config = NetConfig(
                obs_dim=ckpt.obs_dim,
                action_dim=ckpt.action_dim,
                width=ckpt.width,
                blocks=ckpt.blocks,
                value_hidden=ckpt.value_hidden,
                arch=ckpt.arch,
            )
            self.model = build_net(
                net_config, lanes() if net_config.arch == "lane" else None
            ).to(self.device)
            self.model.load_tensors(ckpt.tensors)
        else:
            net_config = NetConfig(
                obs_dim=spec["obs_dim"],
                action_dim=spec["action_dim"],
                width=config.net.width,
                blocks=config.net.blocks,
                value_hidden=config.net.value_hidden,
                arch=config.net.arch,
            )
            self.model = build_net(
                net_config, lanes() if net_config.arch == "lane" else None
            ).to(self.device)

        self.net_config = net_config
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.lr,
            weight_decay=config.train.weight_decay,
        )

    # ------------------------------------------------------------------- fitting --

    def _to_device(
        self, batch: dict[str, np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Scatter one sparse batch into the dense tensors the network takes.

        Returns ``(x, target, value, policy_mask)``. The mask is 1.0 for rows whose policy
        target is real and 0.0 for rows playout cap randomisation capped — see
        :meth:`_policy_loss`, which is where forgetting it would go wrong quietly.
        """
        n = len(batch["value"])
        x = torch.zeros((n, self.net_config.obs_dim), device=self.device)
        x[
            torch.from_numpy(batch["obs_rows"]).to(self.device),
            torch.from_numpy(batch["obs_cols"]).to(self.device),
        ] = torch.from_numpy(batch["obs_vals"]).to(self.device)

        target = torch.zeros((n, self.net_config.action_dim), device=self.device)
        target[
            torch.from_numpy(batch["policy_rows"]).to(self.device),
            torch.from_numpy(batch["policy_cols"]).to(self.device),
        ] = torch.from_numpy(batch["policy_vals"]).to(self.device)

        value = torch.from_numpy(batch["value"]).to(self.device)
        # `.get` with a full-ones fallback so a shard replayed by an older path, or a test
        # that builds a batch by hand, still trains every row — the mask is an addition, not
        # a new requirement.
        flags = batch.get("policy_target")
        mask = (
            torch.ones(n, device=self.device)
            if flags is None
            else torch.from_numpy(flags.astype(np.float32)).to(self.device)
        )
        return x, target, value, mask

    @staticmethod
    def _policy_loss(log_probs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        """Cross-entropy over the rows that have a policy target, and the count of them.

        ⚠️ **The denominator is the masked count, not the batch size.** A capped row's target
        is all zeros, so its cross-entropy is exactly 0 and a plain ``.mean()`` would average
        those zeros in — scaling the policy gradient by the full-search fraction with nothing
        anywhere to say so. At 25% full search that is a silent 4× cut in the policy learning
        rate, which would look like "the policy head stopped learning" and send the search
        for a bug into the wrong file entirely.

        Returns ``(sum, count)`` rather than a mean so the caller can decide, and so the
        holdout can accumulate across batches of different sizes.
        """
        per_row = -(target * log_probs).sum(dim=-1) * mask
        return per_row.sum(), mask.sum()

    def lr_for(self, generation: int) -> float:
        """The learning rate in force at `generation`.

        ``train.lr`` scaled by the last schedule entry the generation index has reached, so
        an empty schedule is a constant rate and every Phase 3 run reproduces unchanged.
        Keyed to the generation index rather than to an optimiser step count because
        ``--resume`` rebuilds this object and the step count does not survive it — see
        :class:`TrainSettings`.
        """
        multiplier = 1.0
        for entry in self.config.train.lr_schedule:
            if generation >= int(entry[0]):
                multiplier = float(entry[1])
        return self.config.train.lr * multiplier

    def fit(
        self,
        buffer: ReplayBuffer,
        rng: np.random.Generator,
        steps: int,
        generation: int = 0,
        *,
        checkpoint: Path | None = None,
        save_every: float = 30.0,
        should_stop: Callable[[], bool] | None = None,
    ) -> StepStats:
        """Take `steps` optimisation steps over batches drawn from `buffer`.

        With ``checkpoint`` set the fit is **resumable**: every ``save_every`` seconds, and on
        the last step, the weights, AdamW's moments, `rng`'s state and the running loss sums
        are written there. A later call for the same fit picks up at the saved step, and the
        weights it ends on are the ones an uninterrupted fit would have produced
        (``test_a_fit_resumed_from_its_checkpoint_ends_on_the_same_weights``). A call that
        finds a *finished* checkpoint takes no steps and restores its end state, which is how
        the loop resumes a generation that was paused after its fit.

        A checkpoint is only resumed by the same fit — same generation, step count, learning
        rate, buffer size and ``[train]`` settings. Anything else starts over from the state
        this trainer is already in, which on a ``--resume`` is the last committed generation's.

        ``should_stop`` is polled between steps. When it says so the fit saves and raises
        :class:`KeyboardInterrupt`, which is how a ``SIGTERM`` costs nothing here rather than up
        to ``save_every``. The poll is deliberately between steps: stopping inside one could
        leave ``optimizer.step`` half-applied.
        """
        cfg = self.config.train
        self.model.train()
        stats = StepStats()
        stats.lr = self.lr_for(generation)
        identity = _canonical(
            {
                "generation": generation,
                "steps": steps,
                "lr": stats.lr,
                "buffer_samples": buffer.samples,
                # Not `device`: a fit paused on CUDA may be resumed on CPU, and the numbers are
                # the same fit either way.
                "train": {k: v for k, v in asdict(cfg).items() if k != "device"},
            }
        )
        policy_sum = value_sum = 0.0
        start = 0
        carried = 0.0

        saved = self._load_fit(checkpoint, identity)
        if saved is not None:
            self.model.load_state_dict(saved["model"])
            self.optimizer.load_state_dict(saved["optimizer"])
            rng.bit_generator.state = json.loads(saved["rng"])
            start = int(saved["step"])
            policy_sum, value_sum = float(saved["policy_sum"]), float(saved["value_sum"])
            stats.policy_first, stats.value_first = saved["policy_first"], saved["value_first"]
            stats.policy_last, stats.value_last = saved["policy_last"], saved["value_last"]
            stats.steps = start
            carried = float(saved["seconds"])
            if start < steps:
                print(f"  train       resuming the fit at step {start:,} of {steps:,}", flush=True)
        for group in self.optimizer.param_groups:
            group["lr"] = stats.lr
        started = last_save = time.perf_counter()

        def save(step: int) -> None:
            state = {
                "identity": identity,
                "step": step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "rng": json.dumps(rng.bit_generator.state),
                "policy_sum": policy_sum,
                "value_sum": value_sum,
                "policy_first": stats.policy_first,
                "value_first": stats.value_first,
                "policy_last": stats.policy_last,
                "value_last": stats.value_last,
                "seconds": carried + time.perf_counter() - started,
            }
            replace_atomically(checkpoint, lambda tmp: torch.save(state, tmp))

        for step in range(start, steps):
            batch = buffer.sample_batch(rng, cfg.batch_size)
            x, target, value, mask = self._to_device(batch)

            logits, predicted = self.model(x)
            log_probs = torch.log_softmax(logits, dim=-1)
            total, counted = self._policy_loss(log_probs, target, mask)
            # `clamp` guards the batch in which every row happened to be capped: at a 25%
            # fraction and 512 rows that is astronomically unlikely, but a division by zero
            # here would poison the weights rather than raise.
            policy_loss = total / counted.clamp(min=1.0)
            # The value head trains on **every** row. That asymmetry is the entire point of
            # playout cap randomisation: one game, one outcome, however little search
            # produced the position.
            value_loss = nn.functional.mse_loss(predicted, value)
            loss = policy_loss + cfg.value_weight * value_loss

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            p, v = float(policy_loss.detach()), float(value_loss.detach())
            policy_sum += p
            value_sum += v
            if step == 0:
                stats.policy_first, stats.value_first = p, v
            stats.policy_last, stats.value_last = p, v
            stats.steps += 1

            if checkpoint is None:
                continue
            now = time.perf_counter()
            stopping = should_stop is not None and should_stop()
            if stopping or step + 1 == steps or now - last_save >= save_every:
                save(step + 1)
                last_save = now
            if stopping and step + 1 < steps:
                raise KeyboardInterrupt

        if stats.steps:
            stats.policy_mean = policy_sum / stats.steps
            stats.value_mean = value_sum / stats.steps
        stats.seconds = carried + time.perf_counter() - started
        return stats

    def _load_fit(self, path: Path | None, identity: dict) -> dict | None:
        """A fit checkpoint for exactly this fit, or ``None``."""
        if path is None or not Path(path).exists():
            return None
        try:
            saved = torch.load(path, map_location=self.device, weights_only=True)
        except Exception as exc:  # a checkpoint that will not load is one to redo, not a crash
            print(f"  train       could not read {path} ({exc}); starting this fit over", flush=True)
            return None
        if saved.get("identity") != identity:
            print(
                f"  train       {Path(path).name} is from a different fit (the buffer or [train] "
                f"changed); starting this fit over",
                flush=True,
            )
            return None
        return saved

    @torch.no_grad()
    def evaluate(self, holdout: Generation) -> EvalStats:
        """Score a held-out generation once, in batches, without touching the weights."""
        cfg = self.config.train
        was_training = self.model.training
        self.model.eval()
        stats = EvalStats()
        policy_sum = value_sum = policy_rows = 0.0
        try:
            for batch in holdout.batches(cfg.batch_size):
                x, target, value, mask = self._to_device(batch)
                logits, predicted = self.model(x)
                log_probs = torch.log_softmax(logits, dim=-1)
                n = len(value)
                # Summed, not meaned, so the last short batch does not get a full batch's
                # weight in the average.
                total, counted = self._policy_loss(log_probs, target, mask)
                policy_sum += float(total)
                policy_rows += float(counted)
                value_sum += float(((predicted - value) ** 2).sum())
                stats.samples += n
        finally:
            self.model.train(was_training)
        if stats.samples:
            # The two denominators differ, deliberately: the value head is scored on every
            # held-out row and the policy head only on the rows that have a target. Dividing
            # both by `samples` would make the held-out policy loss depend on the capping
            # fraction, and `PLAN.md` §4.2 change 7 exists so this number is comparable
            # between runs.
            stats.policy_loss = policy_sum / max(policy_rows, 1.0)
            stats.value_mse = value_sum / stats.samples
        return stats

    # --------------------------------------------------------------- checkpoints --

    def save_optimizer(self, path: str | Path) -> Path:
        """Persist AdamW's first and second moments beside the weights.

        ``--resume`` rebuilds this object from ``best.d52nn``, which carries weights and
        nothing else, so without this every interruption throws away the optimiser's
        momentum and costs a few hundred steps of re-warm. On a preemptible box that is
        paid for more than once. ``PLAN.md`` §4.2 change 5.
        """
        state = self.optimizer.state_dict()
        return replace_atomically(path, lambda tmp: torch.save(state, tmp))

    def load_optimizer(self, path: str | Path) -> bool:
        """Restore moments saved by :meth:`save_optimizer`. False if there is nothing to
        restore, or if what is there does not fit this model — a resumed run is worth more
        than its momentum, so a mismatch warns and carries on rather than stopping."""
        path = Path(path)
        if not path.exists():
            return False
        try:
            self.optimizer.load_state_dict(torch.load(path, map_location=self.device))
        except (ValueError, KeyError, RuntimeError) as exc:
            print(f"  optimiser state in {path} does not fit this model ({exc}); starting cold")
            return False
        return True

    def save(self, path: str | Path) -> Path:
        """Write a `.d52nn` the Rust side can load.

        Moved to CPU first: the checkpoint format is little-endian f32 and
        ``numpy.asarray`` on an MPS tensor is not a thing.
        """
        was = next(self.model.parameters()).device
        self.model.to("cpu")
        try:
            # Atomically, because the loop reads a candidate back after a pause and the engine
            # would otherwise be handed a truncated one.
            written = replace_atomically(
                path, lambda tmp: write_checkpoint(tmp, model=self.model, spec=self.spec)
            )
        finally:
            self.model.to(was)
        return written
