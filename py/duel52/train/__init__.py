"""AlphaZero-style training for Duel 52 — ``PLAN.md`` Phase 3, step 3.

The division of labour, from ``DESIGN.md`` §9: **search and inference are in Rust, training
is in Python.** This package owns the gradients and the loop; everything that plays a game,
enumerates a legal action or encodes an observation is a call into the engine.

    python -m duel52.train check --config configs/train-fast.toml
    python -m duel52.train run   --config configs/train-fast.toml --run-dir runs/first
"""

from __future__ import annotations

from .buffer import ReplayBuffer
from .config import TrainConfig, load_config
from .loop import TrainingLoop, run_loop
from .trainer import Trainer, resolve_device

__all__ = [
    "ReplayBuffer",
    "TrainConfig",
    "Trainer",
    "TrainingLoop",
    "load_config",
    "resolve_device",
    "run_loop",
]
