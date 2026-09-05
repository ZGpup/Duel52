"""The training configuration — every knob, and nothing hard-coded elsewhere.

``CLAUDE.md``: *"Config-driven, no hardcoded constants."* and *"Device-agnostic. Code must
run on MPS locally and CUDA on a rented box with no edits beyond a config value."* This
module is where both of those are kept honest: a run is fully described by one TOML file
plus a seed, and the file is copied into the run directory so a result can always name the
configuration that produced it.
"""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

__all__ = ["GameSettings", "NetSettings", "SelfPlaySettings", "TrainSettings", "GateSettings",
           "RunSettings", "TrainConfig", "load_config"]


def _build(cls: type, data: dict[str, Any], where: str) -> Any:
    """Construct a settings dataclass, rejecting keys it does not have.

    A typo in a TOML key is otherwise silent, and a silent typo in a training config is a
    run that took two hours to not do what you asked.
    """
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"[{where}] has unknown key(s) {sorted(unknown)}; expected some of {sorted(known)}"
        )
    return cls(**data)


@dataclass(frozen=True)
class GameSettings:
    """Which game is being learned. Passed to both the engine CLI and ``encoding_spec()``."""

    variant: str = "split"
    #: ``FINDINGS.md`` F3.1: 16 survives self-play and does *not* survive a ladder against
    #: `random`, which is the permanent anchor rung. 21 is the theoretical maximum, so the
    #: encoder provably cannot assert. Training runs pay the 30% tensor cost and sleep.
    encoding_slots: int = 21
    two_power: str | None = None
    stalemate: int | None = None
    #: What an engine-declared stalemate is worth **to the learner**, to both players.
    #: ``0.0`` makes refusing to play no better than losing. See ``FINDINGS.md`` F3.6 —
    #: at the engine default of 0.5 the loop walks straight into mutual refusal to attack,
    #: because a certain half point beats a risky fight for both sides. Scoring and Elo are
    #: untouched by this; a draw is still half a point there.
    stalemate_value: float = 0.0

    def cli_flags(self) -> list[str]:
        flags = [
            "--variant", self.variant,
            "--encoding-slots", str(self.encoding_slots),
            "--stalemate-value", str(self.stalemate_value),
        ]
        if self.two_power is not None:
            flags += ["--two-power", self.two_power]
        if self.stalemate is not None:
            flags += ["--stalemate", str(self.stalemate)]
        return flags


@dataclass(frozen=True)
class NetSettings:
    """Trunk size. Only consulted when a run starts from scratch — after that the shape
    comes from the checkpoint, which is the only thing that can be right about it."""

    width: int = 128
    blocks: int = 3
    value_hidden: int = 128


@dataclass(frozen=True)
class SelfPlaySettings:
    games: int = 3000
    sims: int = 64
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.3
    dirichlet_weight: float = 0.25
    temperature: float = 1.0
    temperature_decisions: int = 24

    def cli_flags(self) -> list[str]:
        return [
            "--games", str(self.games),
            "--sims", str(self.sims),
            "--c-puct", str(self.c_puct),
            "--dirichlet-alpha", str(self.dirichlet_alpha),
            "--dirichlet-weight", str(self.dirichlet_weight),
            "--temperature", str(self.temperature),
            "--temperature-decisions", str(self.temperature_decisions),
        ]


@dataclass(frozen=True)
class TrainSettings:
    #: ``"auto"`` picks MPS, then CUDA, then CPU. The handoff to a rented box is this string.
    device: str = "auto"
    batch_size: int = 512
    steps_per_generation: int = 1200
    lr: float = 2e-3
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    grad_clip: float = 1.0
    #: Keep one decision in N. Consecutive decisions inside a turn are nearly the same
    #: position, so this costs little signal and is what keeps the replay buffer in RAM.
    sample_stride: int = 2
    buffer_generations: int = 4
    buffer_samples: int = 900_000
    #: Piecewise learning-rate decay, as ``[[from_generation, multiplier], ...]`` ascending.
    #: The multiplier in force is the last entry whose ``from_generation`` the current
    #: generation has reached, so ``[[0, 1.0], [20, 0.25], [38, 0.0625]]`` is
    #: ``PLAN.md`` §4.2 change 5 for a 50-generation run. Empty means a constant ``lr``,
    #: which is what every Phase 3 run used.
    #:
    #: **Keyed to the generation index, not to an optimiser step count**: ``--resume``
    #: rebuilds the ``Trainer`` and the step count does not survive it, so a step-keyed
    #: schedule silently restarts at full learning rate after every interruption. The
    #: generation index is reconstructed from ``log.jsonl``, which does survive.
    lr_schedule: list[list[float]] = field(default_factory=list)
    #: Passes over the replay buffer per generation. When set, it **overrides**
    #: ``steps_per_generation``, which is a fixed step count and therefore a different
    #: number of epochs every time the buffer is a different size.
    #:
    #: That difference is not academic. A fixed 600 steps of batch 512 is 0.9 epochs of a
    #: 340k buffer and **4.1 epochs of the 74k buffer a run holds at generation 1** — and
    #: four passes over a quarter of a generation, applied to weights that are already
    #: good, memorises the shard. Measured, on the first generation of `runs/fourth`: the
    #: training batches fell to a value MSE of 0.225 while the held-out set rose to 0.870,
    #: from the 0.774 the starting checkpoint came in at. Both heads got worse at the job
    #: and better at the sample.
    #:
    #: The third run never hit this because it started from a random net, where the first
    #: generation's overfit is harmless — every candidate beats random — and by generation 3
    #: its buffer was full and it settled at 1.04 epochs for the rest of the run. A
    #: warm-started run has no such grace period, and this is what gives it one.
    epochs_per_generation: float = 0.0
    #: Relabel each drawn sample by one of the six lane permutations (``PLAN.md`` §4.2a).
    #:
    #: Duel 52 is invariant under any permutation of its three lanes — no rule names one —
    #: so this is an **exact** relabelling, not a noise transform: same value, same optimal
    #: policy, legal actions carried across by the same permutation. It costs nothing in
    #: compute or RAM (an index remap on two short arrays at batch-draw time) and it is the
    #: fix for ``FINDINGS.md`` F4.3, where gen022 spends real capacity on an arbitrary
    #: preference for lane 3.
    #:
    #: Off by default, so every earlier run reproduces unchanged. The tables come from the
    #: Rust encoder; nothing about the checkpoint or the shard format changes, so a net
    #: trained with this on is interchangeable with one trained without it.
    lane_augment: bool = False
    #: Samples carved out of the run's **first** shard and never trained on, scored every
    #: generation (``PLAN.md`` §4.2 change 7). 0 disables it.
    #:
    #: Fixed rather than rolling, deliberately. The per-generation value loss in
    #: ``runs/third/log.jsonl`` is computed on batches drawn from a replay window that
    #: slides underneath it, so a flat curve is ambiguous between "the value head learned
    #: all it can" and "the positions got harder". A holdout that never changes removes the
    #: second reading. The cost is that it drifts off-policy as the net improves — it
    #: measures the same question getting answered better, not the current question.
    holdout_samples: int = 0

    def steps_for(self, buffer_samples: int) -> int:
        """How many optimisation steps this generation gets.

        ``epochs_per_generation`` when it is set, so the fitting scales with the data
        available rather than with a constant chosen for a buffer that is not full yet;
        ``steps_per_generation`` otherwise, which is what every Phase 3 run used.
        """
        if self.epochs_per_generation <= 0:
            return self.steps_per_generation
        return max(1, round(self.epochs_per_generation * buffer_samples / self.batch_size))

    def __post_init__(self) -> None:
        if self.epochs_per_generation < 0:
            raise ValueError(
                f"[train] epochs_per_generation cannot be negative; got {self.epochs_per_generation}"
            )
        previous = -1
        for entry in self.lr_schedule:
            if len(entry) != 2:
                raise ValueError(
                    f"[train] lr_schedule entries are [from_generation, multiplier]; got {entry!r}"
                )
            start, multiplier = int(entry[0]), float(entry[1])
            if start <= previous:
                raise ValueError(
                    f"[train] lr_schedule must ascend by from_generation; {start} follows {previous}"
                )
            if multiplier <= 0:
                raise ValueError(f"[train] lr_schedule multiplier must be positive; got {multiplier}")
            previous = start


@dataclass(frozen=True)
class GateSettings:
    """Promotion: two independent tests, and a candidate must pass both.

    The first version of this gate was a single mirror match with a 0.5 threshold, and
    ``FINDINGS.md`` F3.6 is the record of it failing in the most expensive way available —
    passing three consecutive generations of a collapsing agent because the candidate and
    the incumbent stalled each other out. 199 draws in 200 games scores exactly 0.500, and
    0.500 cleared a 0.5 bar. Every promotion was made on no evidence at all.

    So:

    **1. The mirror match is scored on decisive games only.** ``W / (W + L)``, draws
    discarded. A stall then reads as *no evidence* rather than as a tie, which is what it
    actually is. Below ``min_decisive`` decisive games the mirror abstains and the reference
    panel decides alone.

    **2. A reference panel vetoes regression.** The candidate plays fixed opponents that do
    not cooperate with a stall, and may not score more than ``reference_tolerance`` below
    the incumbent on any of them. This is the test that would have caught F3.6 at generation
    2, where the mirror said 0.502 and `random` said 0.929 → 0.600. It is a veto, never a
    target: optimising to beat `random` is not the goal, and the panel only ever blocks.

    Both matches use the same seed every generation, so the comparison is paired and far
    tighter than the independent confidence intervals suggest.
    """

    games: int = 200
    sims: int = 64
    #: On decisive games only. AlphaGo Zero's 0.55 is meaningful again once draws are out
    #: of the denominator.
    threshold: float = 0.55
    #: Fewer decisive games than this and the mirror match abstains rather than reporting a
    #: number computed from three results.
    min_decisive: int = 20
    #: Opponents that will not cooperate with a stall. Also the readout's strength line.
    reference: list[str] = field(default_factory=lambda: ["random", "greedy"])
    reference_games: int = 150
    #: How far below the incumbent a candidate may fall on any reference opponent before it
    #: is refused. Wide enough to absorb paired-match noise at `reference_games`, narrow
    #: enough that F3.6's 0.33 collapse is nowhere near it.
    reference_tolerance: float = 0.05
    #: Consecutive refusals after which the run stops rather than grinding on. A gate that
    #: refuses forever is doing its job; a loop that hides that behind healthy-looking loss
    #: curves is not.
    max_consecutive_refusals: int = 3


@dataclass(frozen=True)
class RunSettings:
    generations: int = 12
    #: Wall-clock budget. The loop finishes the generation it is in and then stops, so a
    #: budget is never exceeded by more than one generation.
    hours: float = 2.5
    #: First self-play seed. Each generation advances it by `selfplay.games`, so no two
    #: generations replay the same deals.
    seed: int = 1_000_000
    #: 0 means every core.
    threads: int = 0
    engine: str = "./target/release/duel52"


@dataclass(frozen=True)
class TrainConfig:
    game: GameSettings = field(default_factory=GameSettings)
    net: NetSettings = field(default_factory=NetSettings)
    selfplay: SelfPlaySettings = field(default_factory=SelfPlaySettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    gate: GateSettings = field(default_factory=GateSettings)
    run: RunSettings = field(default_factory=RunSettings)
    source: str = "<defaults>"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None) -> TrainConfig:
    """Read a training TOML. Missing sections take the defaults above."""
    if path is None:
        return TrainConfig()
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    sections = {
        "game": GameSettings,
        "net": NetSettings,
        "selfplay": SelfPlaySettings,
        "train": TrainSettings,
        "gate": GateSettings,
        "run": RunSettings,
    }
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"{path}: unknown section(s) {sorted(unknown)}")
    built = {name: _build(cls, raw.get(name, {}), name) for name, cls in sections.items()}
    return TrainConfig(source=str(path), **built)
