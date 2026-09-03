"""The AlphaZero loop — ``PLAN.md`` Phase 3, step 3.

Per generation:

1. **Self-play.** ``duel52 selfplay`` plays the current best checkpoint against itself with
   root noise on, and writes a ``.d52sp`` trajectory shard.
2. **Replay.** The shard is replayed through the Rust encoder into the sliding buffer.
3. **Train.** A fixed number of optimisation steps over the whole buffer.
4. **Reference panel.** The candidate plays fixed opponents that will not cooperate with a
   stall — `random` and `greedy` by default.
5. **Gate.** The candidate plays the incumbent, colour-paired, noise off, and is promoted
   only if it clears *both* the reference veto and a decisive-games threshold.

Everything expensive is a subprocess call to the engine binary, for the reason
``DESIGN.md`` §9 gives: search and inference are in Rust and the Elo harness only takes an
``AgentSpec``. Python owns the gradients and nothing else.

Why gating rather than "newest wins"
------------------------------------

A generation can be worse than the one before it — a bad batch, an unlucky self-play sample,
a value head that collapses onto the draw. Without a gate that regression becomes the next
generation's teacher and the run quietly walks backwards, which is expensive to diagnose
after the fact and nearly free to prevent.

Why the gate looks the way it does
----------------------------------

Because the obvious version of it failed, and ``FINDINGS.md`` F3.6 is the record. A single
mirror match with a 0.5 threshold promoted three consecutive generations of a collapsing
agent, because the candidate and the incumbent stalled each other out: 199 draws in 200
games scores exactly 0.500, and 0.500 clears a 0.5 bar. The loss curves looked healthy
throughout. Two changes, both aimed at that:

* **The mirror match is scored on decisive games only**, so a stall reads as *no evidence*
  rather than as a tie.
* **The reference panel runs on the candidate, before the decision, and can veto.** It is
  the only measurement in the loop taken against an opponent with no incentive to stall, and
  it is what would have caught F3.6 at generation 2 — mirror 0.502, `random` 0.929 → 0.600.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..nn.model import spec_for
from .buffer import ReplayBuffer
from .config import TrainConfig
from .trainer import Trainer

__all__ = ["MatchResult", "TrainingLoop", "run_loop"]

#: `MatchStats::report` in `engine/src/ladder.rs`. Parsed rather than re-derived so the
#: score the loop gates on is literally the score the CLI prints.
_SCORE = re.compile(r"score for .*?:\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")
_WLD = re.compile(r"W(\d+)\s+L(\d+)\s+D(\d+)")


@dataclass
class MatchResult:
    score: float
    ci95: float
    wins: int
    losses: int
    draws: int

    @property
    def decisive(self) -> int:
        """Games that actually resolved. A stalled mirror match has none."""
        return self.wins + self.losses

    @property
    def decisive_score(self) -> float:
        """``W / (W + L)`` — the score with draws out of the denominator rather than
        counted as half a point each.

        This is the number the gate reads. ``FINDINGS.md`` F3.6: two agents that stall each
        other out score exactly 0.500 on the ordinary scale, which is indistinguishable from
        a dead-even fight and cleared a 0.5 threshold three generations running. Here the
        same match reports 0 decisive games, which is *no evidence* — a different thing, and
        the gate treats it as one. Returns 0.5 when nothing was decided, but callers should
        check :attr:`decisive` first rather than trust that.
        """
        return 0.5 if self.decisive == 0 else self.wins / self.decisive

    def __str__(self) -> str:
        return (
            f"{self.score:.3f} ± {self.ci95:.3f} (W{self.wins} L{self.losses} D{self.draws}"
            + (f", decisive {self.decisive_score:.3f})" if self.decisive else ", none decisive)")
        )


def say(*args: object) -> None:
    """``print``, flushed.

    The engine's progress lines go to stderr and this loop's go to stdout, and stdout is
    block-buffered the moment the run is piped into a file or a pager — which is exactly
    when someone is watching it. Unflushed, the readout arrives in the wrong order or not
    until the end.
    """
    print(*args, flush=True)


def _hms(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


class TrainingLoop:
    def __init__(self, config: TrainConfig, run_dir: Path, *, resume: bool = False):
        self.config = config
        self.run_dir = run_dir
        self.shards = run_dir / "shards"
        self.checkpoints = run_dir / "checkpoints"
        self.log_path = run_dir / "log.jsonl"
        for d in (self.run_dir, self.shards, self.checkpoints):
            d.mkdir(parents=True, exist_ok=True)

        self.engine = Path(config.run.engine)
        if not self.engine.exists():
            raise FileNotFoundError(
                f"engine binary {self.engine} not found — run `cargo build --release` first"
            )

        self.spec = spec_for(config.game.variant, config.game.encoding_slots)
        self.rng = np.random.default_rng(config.run.seed)
        self.buffer = ReplayBuffer(
            max_generations=config.train.buffer_generations,
            max_samples=config.train.buffer_samples,
            stride=config.train.sample_stride,
            threads=config.run.threads,
        )
        self.history: list[dict] = []
        self.generation = 0
        self.best = self.checkpoints / "best.d52nn"
        #: **High-water mark** per reference opponent — the best score any promoted
        #: checkpoint has managed, not the incumbent's current one.
        #:
        #: Measuring against the incumbent would let the run ratchet downwards: a candidate
        #: that gives up `reference_tolerance − ε` passes, becomes the new baseline, and the
        #: next one gives up as much again. Five generations of that is a collapse made of
        #: individually-legal steps, which is `FINDINGS.md` F3.6 in slow motion. Against a
        #: high-water mark the tolerance is a total budget rather than a per-generation one.
        self.reference_best: dict[str, float] = {}
        self.refusals = 0

        if resume and self.log_path.exists():
            self.history = [json.loads(line) for line in self.log_path.read_text().splitlines() if line.strip()]
            self.generation = max((h["generation"] for h in self.history), default=0)
            say(f"resuming {run_dir} at generation {self.generation}")
            # Refill the window from disk. Without this the first generation after a resume
            # trains on one shard, which is exactly the narrow-buffer failure the window
            # exists to prevent — and it would be invisible in the readout.
            recent = [h for h in self.history[-config.train.buffer_generations :]]
            for h in recent:
                shard = self.shards / f"gen{h['generation']:03d}.d52sp"
                if shard.exists():
                    self.buffer.add(shard, h["generation"])
            if self.buffer.generations:
                say(f"  refilled the buffer with {self.buffer.samples:,} samples from disk")
            # The veto needs the incumbent's reference scores; the last promoted generation
            # is where they are. Without this a resumed run would promote its first
            # candidate unconditionally, which is the hole the veto exists to close.
            for h in self.history:
                if h.get("promoted"):
                    for name, score in h.get("benchmarks", {}).items():
                        self.reference_best[name] = max(
                            self.reference_best.get(name, float(score)), float(score)
                        )
        if not resume:
            (run_dir / "train.toml.used").write_text(json.dumps(config.as_dict(), indent=2))

        self.trainer = Trainer(config, self.spec, self.best if self.best.exists() else None)
        if not self.best.exists():
            self.trainer.save(self.best)
            say(f"initialised {self.best} — {sum(p.numel() for p in self.trainer.model.parameters()):,} parameters")

    # ------------------------------------------------------------- engine calls --

    def _engine_args(self, *args: str) -> list[str]:
        out = [str(self.engine), *args, *self.config.game.cli_flags()]
        if self.config.run.threads:
            out += ["--threads", str(self.config.run.threads)]
        return out

    def selfplay(self, generation: int) -> tuple[Path, dict]:
        """Run one generation of self-play, streaming its progress to our stderr."""
        out = self.shards / f"gen{generation:03d}.d52sp"
        seed = self.config.run.seed + generation * self.config.selfplay.games
        args = self._engine_args(
            "selfplay",
            "--checkpoint", str(self.best),
            "--out", str(out),
            "--seed", str(seed),
            "--generation", str(generation),
            *self.config.selfplay.cli_flags(),
        )
        started = time.perf_counter()
        # Progress goes to the engine's stderr and straight through to ours, so the user
        # watching the run sees games/sec and an ETA while it happens.
        result = subprocess.run(args, stdout=subprocess.PIPE, text=True)
        if result.returncode < 0:
            # Killed by a signal, which on a terminal means the Ctrl-C was meant for the
            # whole run rather than for the engine — so it surfaces as an interrupt here
            # instead of as "self-play failed", which would read like a bug.
            raise KeyboardInterrupt
        if result.returncode != 0:
            raise RuntimeError(f"self-play failed ({result.returncode}): {' '.join(args)}")
        summary = _parse_selfplay(result.stdout)
        summary["seconds"] = time.perf_counter() - started
        summary["seed"] = seed
        return out, summary

    def play_match(self, a: str, b: str, games: int) -> MatchResult:
        args = self._engine_args("match", "--a", a, "--b", b, "--games", str(games), "--seed", "1")
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"match failed: {result.stderr.strip() or result.stdout.strip()}")
        score = _SCORE.search(result.stdout)
        wld = _WLD.search(result.stdout)
        if not score or not wld:
            raise RuntimeError(f"could not read a score out of:\n{result.stdout}")
        return MatchResult(
            score=float(score.group(1)),
            ci95=float(score.group(2)),
            wins=int(wld.group(1)),
            losses=int(wld.group(2)),
            draws=int(wld.group(3)),
        )

    # ------------------------------------------------------------------- the gate --

    def reference_scores(self, checkpoint: Path) -> dict[str, float]:
        """Score `checkpoint` against each fixed reference opponent.

        These are the opponents that will not cooperate with a stall, which is exactly why
        they are the veto rather than the readout.
        """
        gate = self.config.gate
        return {
            opponent: self.play_match(
                f"netmcts:{checkpoint}@{gate.sims}", opponent, gate.reference_games
            ).score
            for opponent in gate.reference
        }

    def judge(self, mirror: MatchResult, reference: dict[str, float]) -> tuple[bool, str]:
        """Decide whether to promote, and say why in one clause.

        Two tests, both of which must pass — see :class:`GateSettings`. Returning the reason
        rather than just the verdict is not decoration: "refused" with no reason is the state
        F3.6 spent three generations in.
        """
        gate = self.config.gate

        for name, score in reference.items():
            was = self.reference_best.get(name)
            if was is not None and score < was - gate.reference_tolerance:
                return (
                    False,
                    f"regressed vs {name}: {score:.3f} is more than "
                    f"{gate.reference_tolerance} below the best-ever {was:.3f}",
                )

        if mirror.decisive < gate.min_decisive:
            # The mirror abstains: too few games were decided to mean anything. The
            # reference panel has already had its say, so this is a pass on no objection.
            return True, f"mirror abstains ({mirror.decisive} decisive of {gate.games})"

        if mirror.decisive_score >= gate.threshold:
            return True, f"decisive score {mirror.decisive_score:.3f} ≥ {gate.threshold}"
        return False, f"decisive score {mirror.decisive_score:.3f} < {gate.threshold}"

    # -------------------------------------------------------------- one generation --

    def step(self) -> dict:
        self.generation += 1
        g = self.generation
        gate = self.config.gate
        started = time.perf_counter()
        say(f"\n── generation {g} " + "─" * 40)

        shard, sp = self.selfplay(g)
        say(
            f"  self-play   {sp['games']} games · {sp['games'] / max(sp['seconds'], 1e-9):.1f} g/s · "
            f"{_hms(sp['seconds'])} · {sp['samples']:,} decisions · "
            f"P0 {sp['p0']:.0f}% P1 {sp['p1']:.0f}% draw {sp['draw']:.0f}%"
        )

        gen = self.buffer.add(shard, g)
        won, drew, lost = self.buffer.value_target_mix()
        say(
            f"  buffer      {self.buffer.samples:,} samples over {len(self.buffer.generations)} "
            f"generation(s), {self.buffer.nbytes / 1e6:.0f} MB · "
            f"value targets win {won:.0%} draw {drew:.0%} loss {lost:.0%}"
        )

        stats = self.trainer.fit(self.buffer, self.rng, self.config.train.steps_per_generation)
        say(
            f"  train       {stats.steps} steps · policy {stats.policy_first:.3f} → {stats.policy_last:.3f} "
            f"(mean {stats.policy_mean:.3f}) · value {stats.value_first:.3f} → {stats.value_last:.3f} "
            f"(mean {stats.value_mean:.3f}) · {_hms(stats.seconds)}"
        )

        candidate = self.checkpoints / f"gen{g:03d}.d52nn"
        self.trainer.save(candidate)

        # The reference panel runs on the *candidate*, before the decision. Measuring the
        # winner afterwards — which is what this used to do — makes the strongest signal
        # available a report rather than a check. `FINDINGS.md` F3.6.
        reference = self.reference_scores(candidate)
        if reference:
            say(
                "  reference   "
                + " · ".join(
                    f"vs {name} {score:.3f}"
                    + (f" (best {self.reference_best[name]:.3f})" if name in self.reference_best else "")
                    for name, score in reference.items()
                )
            )

        mirror = self.play_match(
            f"netmcts:{candidate}@{gate.sims}", f"netmcts:{self.best}@{gate.sims}", gate.games
        )
        promoted, why = self.judge(mirror, reference)
        verdict = "PROMOTED" if promoted else "REFUSED"
        say(f"  gate        candidate vs best {mirror} · {why} → {verdict}")

        if promoted:
            shutil.copyfile(candidate, self.best)
            for name, score in reference.items():
                self.reference_best[name] = max(self.reference_best.get(name, score), score)
            self.refusals = 0
        else:
            # The optimiser state stays; only the weights roll back. Reloading the model is
            # what stops a bad generation from becoming the next generation's teacher.
            self.trainer.model.load_tensors(_tensors_of(self.best))
            self.refusals += 1
            say(
                f"              incumbent kept — {self.refusals} consecutive refusal(s) of "
                f"{gate.max_consecutive_refusals}"
            )

        record = {
            "generation": g,
            "seconds": time.perf_counter() - started,
            "selfplay": sp,
            "buffer_samples": self.buffer.samples,
            "value_mix": {"win": won, "draw": drew, "loss": lost},
            "policy_loss": stats.policy_mean,
            "value_loss": stats.value_mean,
            "gate_score": mirror.score,
            "gate_decisive_score": mirror.decisive_score,
            "gate_decisive_games": mirror.decisive,
            "gate_ci95": mirror.ci95,
            "gate_reason": why,
            "promoted": promoted,
            "refusals": self.refusals,
            "benchmarks": {k: round(v, 4) for k, v in reference.items()},
            "checkpoint": str(candidate),
            "samples_kept": gen.samples,
        }
        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        self.history.append(record)
        return record

    def run(self) -> None:
        cfg = self.config.run
        cfg_gate = self.config.gate
        budget = cfg.hours * 3600.0
        started = time.perf_counter()
        say(
            f"run {self.run_dir} · {cfg.generations} generations max · budget {_hms(budget)} · "
            f"device {self.trainer.device} · config {self.config.source}"
        )
        durations: list[float] = []
        while self.generation < cfg.generations:
            elapsed = time.perf_counter() - started
            # Stop before a generation that would overrun, not after. A budget the user set
            # to fit an afternoon should be a bound, and a generation is ~10 minutes.
            projected = elapsed + (sum(durations) / len(durations) if durations else 0.0)
            if elapsed >= budget or projected > budget:
                say(
                    f"\nstopping at generation {self.generation} after {_hms(elapsed)} — "
                    f"another generation would not fit the {_hms(budget)} budget"
                )
                break
            try:
                record = self.step()
            except KeyboardInterrupt:
                print("\ninterrupted — the best checkpoint and the log are on disk", file=sys.stderr, flush=True)
                break
            durations.append(record["seconds"])
            say(f"  elapsed     {_hms(time.perf_counter() - started)} of {_hms(budget)}")
            if self.refusals >= cfg_gate.max_consecutive_refusals:
                # Not a crash and not success. The gate is doing its job and the loop is
                # not making progress, and the worst thing to do with that is keep going
                # quietly for another two hours.
                say(
                    f"\nstopping: {self.refusals} candidates in a row were refused. The "
                    f"incumbent is still the best checkpoint. Something upstream of the "
                    f"gate needs changing — look at the self-play draw rate and the "
                    f"reference line before spending more compute."
                )
                break
        self.summary()

    def summary(self) -> None:
        say("\n" + "═" * 56)
        say(f"best checkpoint: {self.best}")
        if not self.history:
            return
        say(
            f"{'gen':>4} {'draw%':>6} {'decisive':>9} {'promo':>6} {'policy':>8} {'value':>7}"
            "  reference"
        )
        for h in self.history:
            marks = " ".join(f"{k}={v:.3f}" for k, v in h.get("benchmarks", {}).items())
            decisive = h.get("gate_decisive_games", 0)
            score = h.get("gate_decisive_score")
            cell = f"{score:.3f}" if decisive else "—"
            say(
                f"{h['generation']:>4} {h['selfplay'].get('draw', 0):>5.0f}% "
                f"{cell:>9} {'yes' if h['promoted'] else 'no':>6} "
                f"{h['policy_loss']:>8.3f} {h['value_loss']:>7.3f}  {marks}"
            )
        say(
            "\nNext: the real measurement is the frozen ladder —\n"
            f"  {self.engine} ladder --agents random,greedy,flatmc:600,pimc:8x1,ismcts:800,"
            f"netmcts:{self.best}@{self.config.gate.sims} \\\n"
            f"      --games 400 --markdown {' '.join(self.config.game.cli_flags())}"
        )


def _tensors_of(path: Path) -> list:
    from ..nn.checkpoint import read_checkpoint

    return read_checkpoint(path).tensors


def _parse_selfplay(text: str) -> dict:
    """Pull the numbers out of `SelfPlayReport::report`."""
    out = {"games": 0, "samples": 0, "p0": 0.0, "p1": 0.0, "draw": 0.0, "decisions_per_game": 0.0}
    m = re.search(r"—\s*(\d+) games, (\d+) samples", text)
    if m:
        out["games"], out["samples"] = int(m.group(1)), int(m.group(2))
    m = re.search(r"([0-9.]+) decisions/game · P0 ([0-9.]+)% P1 ([0-9.]+)% draw ([0-9.]+)%", text)
    if m:
        out["decisions_per_game"] = float(m.group(1))
        out["p0"], out["p1"], out["draw"] = (float(m.group(i)) for i in (2, 3, 4))
    return out


def run_loop(config: TrainConfig, run_dir: Path, *, resume: bool = False) -> None:
    TrainingLoop(config, run_dir, resume=resume).run()
