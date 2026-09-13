"""The R-NaD run configuration — ``PLAN.md`` item 8.

One TOML plus a seed describes a run, as for ``duel52.train``. ``[game]`` and ``[net]`` are
the *same* settings classes the AlphaZero loop uses, imported rather than copied, so the two
learners cannot disagree about what a game or a trunk is. ``[rnad]`` carries the learner and
``[run]`` the machine.

Defaults are the reference's (OpenSpiel ``rnad.py`` at ``d1dcdf5d``) wherever a reference
value exists.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..train.config import GameSettings, NetSettings, _build

__all__ = ["RNaDSettings", "RNaDRunSettings", "RNaDConfig", "load_config"]


@dataclass(frozen=True)
class RNaDSettings:
    #: ``"auto"`` picks MPS, then CUDA, then CPU — the whole of the laptop-to-GPU handoff.
    device: str = "auto"
    #: Games per learner step, played fresh with the current online net. The reference's
    #: ``batch_size``; its ``trajectory_max`` has no counterpart, since whole games are used.
    batch_games: int = 256
    #: Positions per forward pass in the learner. ``0`` evaluates the whole batch at once;
    #: set it when a batch does not fit in device memory. Gradients accumulate exactly.
    learner_chunk: int = 0
    #: Positions per forward pass in the actor. ``0`` evaluates every waiting game at once.
    actor_chunk: int = 0
    eta_reward_transform: float = 0.2
    #: Learner steps per regularisation iteration, and how many times each size repeats; the
    #: last size repeats forever. See ``core.EntropySchedule``.
    entropy_schedule_size: list[int] = field(default_factory=lambda: [20_000])
    entropy_schedule_repeats: list[int] = field(default_factory=lambda: [1])
    target_network_avg: float = 0.001
    learning_rate: float = 5e-5
    adam_b1: float = 0.0
    adam_b2: float = 0.999
    #: The reference writes ``10e-8``, which is ``1e-7``.
    adam_eps: float = 1e-7
    #: Global gradient-norm clip; ``0`` is off. The reference clips updates elementwise at
    #: 10,000, which never binds, so off is the faithful default.
    grad_clip: float = 0.0
    nerd_beta: float = 2.0
    nerd_clip: float = 10_000.0
    c_vtrace: float = 1.0
    #: The reference passes ``rho = inf``. On policy the ratio is 1 anyway.
    rho_vtrace: float = math.inf

    def __post_init__(self) -> None:
        if self.batch_games <= 0:
            raise ValueError(f"[rnad] batch_games must be positive; got {self.batch_games}")
        if len(self.entropy_schedule_size) != len(self.entropy_schedule_repeats):
            raise ValueError("[rnad] entropy_schedule_size and _repeats must be parallel lists")
        if not 0.0 < self.target_network_avg <= 1.0:
            raise ValueError("[rnad] target_network_avg must be in (0, 1]")


@dataclass(frozen=True)
class RNaDRunSettings:
    #: Wall-clock budget across every resume of the run.
    hours: float = 3.0
    #: A learner-step cap as well as the clock; ``0`` is none.
    max_steps: int = 0
    #: Step ``n`` plays games seeded ``seed + n·batch_games …`` and samples from
    #: ``default_rng([seed, n])``, so a resumed run continues the same streams.
    seed: int = 7_000_000
    #: Threads the engine uses to encode and apply a batch; ``0`` is every core.
    threads: int = 0
    engine: str = "./target/release/duel52"
    log_every: int = 10
    #: Learner steps between evaluation matches; ``0`` evaluates only at the end.
    eval_every: int = 250
    eval_games: int = 200
    eval_opponents: list[str] = field(default_factory=lambda: ["random", "greedy"])
    #: Threads for the evaluation match subprocess; ``0`` lets the engine decide.
    eval_threads: int = 0
    save_every_secs: float = 300.0


@dataclass(frozen=True)
class RNaDConfig:
    game: GameSettings = field(default_factory=GameSettings)
    net: NetSettings = field(default_factory=NetSettings)
    rnad: RNaDSettings = field(default_factory=RNaDSettings)
    run: RNaDRunSettings = field(default_factory=RNaDRunSettings)
    source: str = "<defaults>"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None) -> RNaDConfig:
    """Read an R-NaD TOML. Missing sections take the defaults; unknown keys are errors."""
    if path is None:
        return RNaDConfig()
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    sections = {"game": GameSettings, "net": NetSettings, "rnad": RNaDSettings, "run": RNaDRunSettings}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"{path}: unknown section(s) {sorted(unknown)}")
    built = {name: _build(cls, raw.get(name, {}), name) for name, cls in sections.items()}
    return RNaDConfig(source=str(path), **built)
