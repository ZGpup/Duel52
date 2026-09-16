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

Pausing
-------

The loop runs on a shared box that pauses a job by killing it, and a generation there is half
an hour. So a generation is saved as it goes, not only at its end, and ``--resume`` carries on
inside the generation it was paused in. What each step saves, and so the most a pause costs:

=================  ====================================================  ==================
step               saved                                                 a pause costs
=================  ====================================================  ==================
self-play          every game, the moment it ends (``--journal``)        the games in flight
replay             nothing — deterministic from the shard                the replay
fit                every ``run.save_every_secs`` (30), and on SIGTERM    ≤ 30 s of fitting
panel, gate        every game, the moment it ends (``--journal``)        the games in flight
the decision       before it is acted on                                 nothing
=================  ====================================================  ==================

At 128 cores the games in flight are worth ~100 s of self-play and ~20 s of a gate. The replay
and the buffer refill are paid again on every resume and are not saved, because they are
gigabytes that recompute exactly.

A resumed generation is **the same generation**. Self-play writes the same shard bytes, the
fit ends on the same weights, and the gate scores the same — each is tested — so its log
record matches an uninterrupted run's (``test_a_paused_generation_resumes_to_the_same_record``).

What makes that true, since it is easy to break:

* ``progress/genNNN.json`` holds each finished step's result **with the inputs it was
  computed from**. A result whose inputs no longer match — a ``--resume`` with a changed
  ``[selfplay]``, say, or a refitted candidate — is recomputed, and so is everything after it.
* **The decision is recorded before anything acts on it.** Promotion copies the candidate over
  ``best.d52nn``, and a gate replayed after that copy would be the candidate playing itself.
* **``log.jsonl`` is the commit.** A generation is done when its line is written; the optimiser
  moments and ``checkpoints/rng.json`` are written just before it, and ``progress/`` is cleared
  just after.
* SIGTERM and SIGHUP stop the loop at the next safe point, which is at most one optimiser step
  away. SIGKILL is fine too — it just costs up to ``save_every_secs`` of fitting.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..nn.model import spec_for
from .buffer import Generation, LaneAugmenter, ReplayBuffer, load_generation
from .config import TrainConfig
from .durable import append_line, copy_atomically, replace_atomically, write_json_atomically
from .trainer import Trainer

__all__ = ["MatchResult", "Progress", "TrainingLoop", "run_loop"]

#: Default generation the fixed holdout is carved from, when ``train.holdout_generation`` is
#: not set. One, because that is the only generation guaranteed to exist for the whole run and
#: ``--resume`` can rebuild the holdout from it without remembering anything.
#:
#: ⚠️ It is the wrong default for a **from-scratch** run, whose generation 1 is played by a
#: random init — see ``TrainSettings.holdout_generation`` and ``FINDINGS.md`` F4.6.
HOLDOUT_GENERATION = 1

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

    @property
    def decisive_ci95(self) -> float:
        """95% interval on :attr:`decisive_score`, from the decisive games alone.

        Conservative — the mirror match is colour-paired on a fixed seed, so the real
        interval is tighter than this — and reported anyway, because ``FINDINGS.md`` F3.7's
        three run-ending refusals were every one of them inside their own interval of even
        and nothing in the readout said so while the run was live. A gate whose interval
        straddles the threshold is a gate that is not deciding anything.
        """
        if self.decisive == 0:
            return 0.0
        p = self.wins / self.decisive
        return 1.96 * math.sqrt(max(p * (1.0 - p), 1e-12) / self.decisive)

    def __str__(self) -> str:
        return (
            f"{self.score:.3f} ± {self.ci95:.3f} (W{self.wins} L{self.losses} D{self.draws}"
            + (
                f", decisive {self.decisive_score:.3f} ± {self.decisive_ci95:.3f} "
                f"of {self.decisive})"
                if self.decisive
                else ", none decisive)"
            )
        )


def _canonical(value: Any) -> Any:
    """``value`` as JSON will give it back, so recorded inputs compare equal to rebuilt ones."""
    return json.loads(json.dumps(value))


class Progress:
    """What one generation has finished, on disk — see "Pausing" in the module docstring.

    Each entry is a result plus the inputs that produced it. :meth:`result` returns the result
    only while the inputs still match, and :meth:`record` forgets every entry recorded after the
    one it writes, so recomputing a step can never leave a later step's stale result behind.
    """

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {}
        if path.exists():
            try:
                self.data = json.loads(path.read_text())
            except json.JSONDecodeError:
                say(f"  {path.name} does not parse; this generation starts over")
        self.resumed = bool(self.data)
        if self.resumed:
            self.data["resumes"] = int(self.data.get("resumes", 0)) + 1
        self._carried = float(self.data.get("seconds", 0.0))
        self._started = time.perf_counter()
        # Written now, not at the first finished step, so an attempt paused before it finished
        # anything still counts as one.
        self.save()

    @property
    def resumes(self) -> int:
        return int(self.data.get("resumes", 0))

    def seconds(self) -> float:
        """Time spent on this generation up to now, across every attempt at it — counting each
        paused attempt up to the last step it saved, since the rest was not kept."""
        return self._carried + time.perf_counter() - self._started

    def done(self) -> list[str]:
        steps = [k for k, v in self.data.items() if isinstance(v, dict) and "inputs" in v]
        return steps + (["the decision"] if "decision" in self.data else [])

    def result(self, key: str, inputs: Any) -> Any | None:
        entry = self.data.get(key)
        if not isinstance(entry, dict) or entry.get("inputs") != _canonical(inputs):
            return None
        return entry["result"]

    def record(self, key: str, inputs: Any, result: Any) -> None:
        keys = list(self.data)
        if key in keys:
            for later in keys[keys.index(key) + 1 :]:
                if isinstance(self.data[later], dict) and "inputs" in self.data[later]:
                    del self.data[later]
            del self.data[key]
        self.data[key] = {"inputs": _canonical(inputs), "result": _canonical(result)}
        self.save()

    def save(self) -> None:
        self.data["seconds"] = self.seconds()
        write_json_atomically(self.path, self.data)


def _without_new_defaults(previous: dict, current: dict) -> dict:
    """`current` without the settings `previous` predates that still hold their defaults.

    Those are keys added to the code since the run began, not changes to the run, and without
    this every run in progress would record a "config changed" the first time it was resumed
    under a newer build.
    """
    defaults = TrainConfig().as_dict()
    out = {}
    for section, values in current.items():
        before = previous.get(section)
        if isinstance(values, dict) and isinstance(before, dict):
            values = {
                k: v
                for k, v in values.items()
                if k in before or _canonical(v) != _canonical(defaults[section].get(k))
            }
        out[section] = values
    return out


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_log(path: Path) -> list[dict]:
    """``log.jsonl``, tolerating the one kind of damage a pause can do to it.

    A kill mid-append can leave a torn last line. That generation never committed — its line is
    the commit — so the line is dropped and the file rewritten without it, and the resume redoes
    the generation from its progress file. A bad line anywhere else is real damage and raises.
    """
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    records = []
    for i, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
            say(f"  {path.name}: dropping a torn last line — that generation had not committed")
            kept = "".join(good + "\n" for good in lines[:-1])
            replace_atomically(path, lambda tmp: tmp.write_text(kept))
    return records


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
    def __init__(
        self,
        config: TrainConfig,
        run_dir: Path,
        *,
        resume: bool = False,
        init_from: str | Path | None = None,
    ):
        self.config = config
        self.run_dir = run_dir
        self.shards = run_dir / "shards"
        self.checkpoints = run_dir / "checkpoints"
        #: Work finished inside a generation that has not committed yet — "Pausing", above.
        self.progress_dir = run_dir / "progress"
        self.log_path = run_dir / "log.jsonl"
        self.baseline_path = run_dir / "baseline.json"
        for d in (self.run_dir, self.shards, self.checkpoints, self.progress_dir):
            d.mkdir(parents=True, exist_ok=True)
        #: Set by SIGTERM/SIGHUP; the loop stops at its next safe point. See :meth:`run`.
        self._stop_requested = False
        self._child: subprocess.Popen | None = None

        self.engine = Path(config.run.engine)
        if not self.engine.exists():
            raise FileNotFoundError(
                f"engine binary {self.engine} not found — run `cargo build --release` first"
            )

        self.spec = spec_for(
            config.game.variant, config.game.encoding_slots, config.game.rules_file
        )
        #: Whether the incumbent began as another run's checkpoint. If so it may have been
        #: trained on other rules, which ``duel52 match`` refuses unless told this is a
        #: gate — see :meth:`play_match`. On ``--resume`` the answer is in the recorded config.
        recorded = run_dir / "train.toml.used"
        self.warm_started = init_from is not None or (
            resume
            and recorded.exists()
            and json.loads(recorded.read_text()).get("init_from") is not None
        )
        self.rng = np.random.default_rng(config.run.seed)
        #: The six lane relabellings, or ``None``. Built once from the engine — never in
        #: Python (``CLAUDE.md``: one encoder, and a permutation table is a reading of it).
        self.augment = (
            LaneAugmenter.from_engine(
                config.game.variant, config.game.encoding_slots, config.game.rules_file
            )
            if config.train.lane_augment
            else None
        )
        self.buffer = ReplayBuffer(
            max_generations=config.train.buffer_generations,
            max_samples=config.train.buffer_samples,
            stride=config.train.sample_stride,
            threads=config.run.threads,
            augment=self.augment,
        )
        self.history: list[dict] = []
        self.generation = 0
        self.best = self.checkpoints / "best.d52nn"
        self.optimizer_state = self.checkpoints / "optimizer.pt"
        #: Samples carved off generation 1's shard and never trained on. ``None`` when
        #: ``train.holdout_samples`` is 0, or before generation 1 has been played.
        self.holdout: Generation | None = None
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
            self.history = _read_log(self.log_path)
            self.generation = max((h["generation"] for h in self.history), default=0)
            say(f"resuming {run_dir} at generation {self.generation}")
            # A streak is consecutive across a pause too. Before generations could be resumed
            # part way this reset to 0, which quietly let a stalled run grind on for another
            # `max_consecutive_refusals` generations after every restart.
            self.refusals = int(self.history[-1].get("refusals", 0)) if self.history else 0
            self._restore_rng()
            # Refill the window from disk. Without this the first generation after a resume
            # trains on one shard, which is exactly the narrow-buffer failure the window
            # exists to prevent — and it would be invisible in the readout.
            recent = [h for h in self.history[-config.train.buffer_generations :]]
            for h in recent:
                shard = self.shards / f"gen{h['generation']:03d}.d52sp"
                if shard.exists():
                    self._replay_into_buffer(shard, h["generation"])
            if self.buffer.generations:
                say(f"  refilled the buffer with {self.buffer.samples:,} samples from disk")
            # The holdout is derived from `train.holdout_generation`'s shard, so it survives
            # a resume without being stored — but only if that shard is still on disk, and
            # after a few generations it is no longer in the window that was just refilled.
            self._rebuild_holdout()
            # The veto needs the incumbent's reference scores; the last promoted generation
            # is where they are. Without this a resumed run would promote its first
            # candidate unconditionally, which is the hole the veto exists to close.
            for h in self.history:
                if h.get("promoted"):
                    for name, score in h.get("benchmarks", {}).items():
                        self.reference_best[name] = max(
                            self.reference_best.get(name, float(score)), float(score)
                        )
        if resume:
            # A warm-started run's veto baseline is the checkpoint it started from, which is in
            # `baseline.json` and in no history record. Read whether or not there is a log yet:
            # a pause during generation 1 leaves none, and resuming it without the baseline
            # would promote the first candidate on the mirror match alone.
            if self.baseline_path.exists():
                stored = json.loads(self.baseline_path.read_text()).get("reference", {})
                for name, score in stored.items():
                    self.reference_best[name] = max(
                        self.reference_best.get(name, float(score)), float(score)
                    )
            self._clear_stale_progress()
        if not resume:
            if init_from is not None:
                self._warm_start(Path(init_from))
            used = dict(config.as_dict())
            used["init_from"] = str(init_from) if init_from is not None else None
            (run_dir / "train.toml.used").write_text(json.dumps(used, indent=2))
        else:
            self._record_config_change()

        self.trainer = Trainer(config, self.spec, self.best if self.best.exists() else None)
        if not self.best.exists():
            self.trainer.save(self.best)
            say(f"initialised {self.best} — {sum(p.numel() for p in self.trainer.model.parameters()):,} parameters")
        if resume and self.trainer.load_optimizer(self.optimizer_state):
            say(f"  restored the optimiser moments from {self.optimizer_state}")

        # A warm start begins with an incumbent that has no history, so the reference panel
        # has no high-water mark to veto against and the first candidate would be promoted
        # on the mirror match alone. Measuring the incumbent once closes that, and it is
        # also the row every later generation's reference column is read against.
        if init_from is not None and not resume and self.config.gate.reference:
            self._measure_baseline(Path(init_from))
        elif resume and not self.history and self.config.gate.reference and not self.baseline_path.exists():
            # Paused while the baseline was being measured, before generation 1 began. The run
            # was warm-started if its first recorded config says so.
            recorded = self.run_dir / "train.toml.used"
            started_from = json.loads(recorded.read_text()).get("init_from") if recorded.exists() else None
            if started_from:
                self._measure_baseline(Path(started_from))

    # ------------------------------------------------------------------- pausing --

    def _restore_rng(self) -> None:
        """Put the batch RNG back where the last committed generation left it.

        The state rides in the log record rather than a file beside it, because the log line is
        the commit: a separate file could be a generation ahead of the log after a badly-timed
        pause. A run logged before this existed has no state to restore and re-seeds from
        ``run.seed``, which is what every resume used to do.
        """
        state = self.history[-1].get("rng_state") if self.history else None
        if state is not None:
            self.rng.bit_generator.state = state

    def _clear_stale_progress(self) -> None:
        """Delete progress for generations that committed, and half-written temporaries.

        A generation's progress is cleared just after its log line, so a pause between the two
        leaves some behind. It is finished work for a finished generation, and resuming from it
        would replay a generation that already happened.
        """
        for path in self.progress_dir.iterdir():
            match = re.match(r"gen(\d+)", path.name)
            if match and int(match.group(1)) <= self.generation:
                path.unlink()
            elif path.name.startswith("baseline") and self.baseline_path.exists():
                path.unlink()
        for directory in (self.progress_dir, self.checkpoints, self.shards):
            for tmp in directory.glob("*.tmp"):
                tmp.unlink()

    def _progress(self, generation: int) -> Progress:
        progress = Progress(self.progress_dir / f"gen{generation:03d}.json")
        if progress.resumed:
            done = progress.done()
            say(
                f"  resuming    paused {progress.resumes} time(s) · already done: "
                + (", ".join(done) if done else "nothing past the journals")
                + f" · {_hms(progress.seconds())} spent so far"
            )
        return progress

    def _fit_path(self, generation: int) -> Path:
        return self.progress_dir / f"gen{generation:03d}-fit.pt"

    def _journal(self, prefix: str, step: str) -> Path:
        return self.progress_dir / f"{prefix}-{step}.journal"

    def _request_stop(self, signum: int, _frame: object) -> None:
        """SIGTERM / SIGHUP. Flag it and stop the engine; the loop saves and exits at its next
        safe point. Nothing is printed here: a handler that prints can interrupt a print."""
        self._stop_requested = True
        child = self._child
        if child is not None and child.poll() is None:
            # The engine's journal already holds every finished game, so there is nothing to
            # wait for — and a grace period is usually seconds, not a gate's worth.
            child.terminate()

    def _check_stop(self) -> None:
        if getattr(self, "_stop_requested", False):
            raise KeyboardInterrupt

    def _record_config_change(self) -> None:
        """On a resume with a different config, record the new one beside the old.

        ``--resume`` re-reads the TOML, so a run can change shape halfway through — and
        this one did, at generation 2. ``CLAUDE.md``: an unreproducible finding is not a
        finding, and a run directory whose ``train.toml.used`` describes only the first
        generation cannot say what produced the rest. The original is never overwritten;
        each change is stamped with the generation it takes effect from.
        """
        recorded = sorted(self.run_dir.glob("train.toml.used*"))
        current = dict(self.config.as_dict())
        if recorded:
            previous = json.loads(recorded[-1].read_text())
            if {k: v for k, v in previous.items() if k != "init_from"} == _canonical(
                _without_new_defaults(previous, current)
            ):
                return
        path = self.run_dir / f"train.toml.used.from-gen{self.generation + 1:03d}"
        path.write_text(json.dumps(current, indent=2))
        say(f"  config changed since this run started — recorded as {path.name}")

    # ---------------------------------------------------------------- warm start --

    def _warm_start(self, checkpoint: Path) -> None:
        """Begin the run from an existing checkpoint instead of from a random init.

        The incumbent is a file, so this is a copy — but a checked one. A checkpoint whose
        trunk does not match ``[net]`` would load anyway (the shape comes from the
        checkpoint, which is the only thing that can be right about it) and the config's
        ``blocks``/``width`` would silently mean nothing, which is precisely the class of
        quiet mismatch the layout hashes exist to prevent elsewhere.
        """
        from ..nn.checkpoint import read_checkpoint

        if not checkpoint.exists():
            raise FileNotFoundError(f"--init-from {checkpoint} does not exist")
        ckpt = read_checkpoint(checkpoint)
        ckpt.check_against(self.spec)  # variant, encoding_slots, both layout hashes
        net = self.config.net
        if ckpt.arch != net.arch:
            # Checked before the trunk sizes because it is the more fundamental mismatch and
            # the more confusing one: `128 × 6` describes both architectures, so a size-only
            # message would look like the shapes agreed when they share no tensor at all.
            raise ValueError(
                f"--init-from {checkpoint} is a {ckpt.arch!r} network but [net] asks for "
                f"{net.arch!r}. The two share no tensor names, so there is nothing to carry "
                f"over — a change of architecture is a from-scratch run by construction."
            )
        have = (ckpt.width, ckpt.blocks, ckpt.value_hidden)
        want = (net.width, net.blocks, net.value_hidden)
        if have != want:
            raise ValueError(
                f"--init-from {checkpoint} is width={ckpt.width} blocks={ckpt.blocks} "
                f"value_hidden={ckpt.value_hidden}, but [net] asks for width={net.width} "
                f"blocks={net.blocks} value_hidden={net.value_hidden}. The checkpoint would "
                f"win and [net] would silently mean nothing — change one of them."
            )
        copy_atomically(checkpoint, self.best)
        say(f"warm start from {checkpoint} — {ckpt.width}×{ckpt.blocks} trunk, kept as the incumbent")

    def _measure_baseline(self, checkpoint: Path) -> None:
        """Score the warm-start checkpoint on the reference panel, once, before generation 1."""
        say(f"  scoring the starting checkpoint on the reference panel ({self.config.gate.reference_games} games each)")
        progress = Progress(self.progress_dir / "baseline.json")
        scores = self.reference_scores(self.best, progress=progress, prefix="baseline")
        for name, score in scores.items():
            self.reference_best[name] = max(self.reference_best.get(name, score), score)
        write_json_atomically(self.baseline_path, {"init_from": str(checkpoint), "reference": scores})
        progress.path.unlink(missing_ok=True)
        say("  baseline    " + " · ".join(f"vs {n} {s:.3f}" for n, s in scores.items()))

    # ------------------------------------------------------------------- holdout --

    def _replay_into_buffer(self, path: Path, generation: int) -> Generation:
        """Replay a shard into the training buffer, carving the fixed holdout off the front
        of generation 1's.

        The carve happens on the way in, on every path into the buffer — the live one and
        the refill after a ``--resume`` — so a holdout sample cannot reach a gradient step
        by a back door. That is the whole value of the number: it is only a held-out score
        if it was never trained on, and "it was held out except after a restart" is not a
        distinction anyone would notice in a log file.
        """
        cfg = self.config.train
        want = cfg.holdout_samples
        if want <= 0 or generation != cfg.holdout_generation:
            return self.buffer.add(path, generation)

        full = load_generation(path, generation, stride=cfg.sample_stride, threads=self.config.run.threads)
        keep = min(want, full.samples // 2)  # never hold out more than half a generation
        if keep < want:
            say(f"  holdout     {full.samples:,} samples in the shard; holding out {keep:,} rather than {want:,}")
        self.holdout = full.slice(0, keep)
        return self.buffer.append(full.slice(keep, full.samples))

    def _rebuild_holdout(self) -> None:
        """Re-derive the holdout after a resume. Deterministic — it is a prefix of a shard
        that is still on disk — so nothing about it needs storing."""
        cfg = self.config.train
        if cfg.holdout_samples <= 0 or self.holdout is not None:
            return
        at = cfg.holdout_generation
        shard = self.shards / f"gen{at:03d}.d52sp"
        if not shard.exists():
            # Two different situations, and the message says which: the run has not reached
            # the holdout generation yet (normal, and it will be carved on the way past), or
            # the shard is gone (the score is lost for good).
            if self.generation < at:
                say(f"  holdout     will be carved from generation {at}, not yet played")
            else:
                say(f"  no {shard.name} on disk — the held-out score is unavailable for this run")
            return
        full = load_generation(shard, at, stride=cfg.sample_stride, threads=self.config.run.threads)
        self.holdout = full.slice(0, min(cfg.holdout_samples, full.samples // 2))
        say(f"  rebuilt the {self.holdout.samples:,}-sample holdout from {shard.name}")

    # ------------------------------------------------------------- engine calls --

    def _engine_args(self, *args: str) -> list[str]:
        out = [str(self.engine), *args, *self.config.game.cli_flags()]
        if self.config.run.threads:
            out += ["--threads", str(self.config.run.threads)]
        return out

    def _run_engine(self, args: list[str], *, capture_stderr: bool) -> subprocess.CompletedProcess:
        """Run the engine as a child this loop can stop.

        Killed by a signal — on a terminal a Ctrl-C meant for the whole run, on the shared box a
        pause — surfaces as :class:`KeyboardInterrupt` rather than as "the engine failed", which
        would read like a bug. Whatever the engine finished is already in its journal.
        """
        self._check_stop()
        child = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE if capture_stderr else None,
            text=True,
        )
        self._child = child
        try:
            if self._stop_requested:  # the signal landed between the check and the launch
                child.terminate()
            out, err = child.communicate()
        except BaseException:
            child.kill()
            child.wait()
            raise
        finally:
            self._child = None
        if child.returncode < 0 or self._stop_requested:
            raise KeyboardInterrupt
        return subprocess.CompletedProcess(args, child.returncode, out, err)

    def selfplay(self, generation: int, journal: Path | None = None) -> tuple[Path, dict]:
        """Run one generation of self-play, streaming its progress to our stderr.

        With a ``journal`` every game is saved as it finishes and a re-run resumes, writing
        the same shard an uninterrupted run would.
        """
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
        if journal is not None:
            args += ["--journal", str(journal)]
        if self.config.run.eval_batch > 1:
            # ⚠️ **Self-play only, and that is a measurement rather than an oversight.**
            # Batching a *match* is a regression: measured on the 8-core laptop over a
            # 300-game gate between two lane nets at 256 sims, `--eval-batch 64` took
            # 401.2 s against 373.5 s unbatched — 7% slower, for identical scores. A gate
            # splits its in-flight games between two checkpoints, so it reaches only half
            # self-play's batch, and at that width the trunk saving no longer covers
            # interleaving 37 live search trees per worker. `FINDINGS.md` F4.8.
            args += ["--eval-batch", str(self.config.run.eval_batch)]
        started = time.perf_counter()
        # Progress goes to the engine's stderr and straight through to ours, so the user
        # watching the run sees games/sec and an ETA while it happens.
        result = self._run_engine(args, capture_stderr=False)
        if result.returncode != 0:
            raise RuntimeError(f"self-play failed ({result.returncode}): {' '.join(args)}")
        summary = _parse_selfplay(result.stdout)
        summary["seconds"] = time.perf_counter() - started
        summary["seed"] = seed
        return out, summary

    def play_match(self, a: str, b: str, games: int, journal: Path | None = None) -> MatchResult:
        args = self._engine_args("match", "--a", a, "--b", b, "--games", str(games), "--seed", "1")
        if journal is not None:
            args += ["--journal", str(journal)]
        if self.warm_started:
            # A warm start into another ruleset makes the incumbent a checkpoint trained on
            # other rules until a candidate replaces it, and `match` refuses to score one of
            # those. Here it is the loop's own gate and panel, not a result anyone reads, so
            # say so. Candidates are always stamped with this run's rules, which is why only a
            # warm-started run needs it. `MODULAR_RULES.md` §6.
            args += ["--warm-start-gate"]
        result = self._run_engine(args, capture_stderr=True)
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

    def reference_games_for(self, opponent: str) -> int:
        """Games for one reference row this generation — see :meth:`GateSettings.games_for`.

        Read off the *high-water mark* rather than the incumbent's latest score, for the
        same reason the veto is: a row that has saturated and then slipped is a row the
        panel should still be watching at full size.
        """
        return self.config.gate.games_for(opponent, self.reference_best.get(opponent))

    def reference_scores(
        self,
        checkpoint: Path,
        progress: Progress | None = None,
        prefix: str | None = None,
    ) -> dict[str, float]:
        """Score `checkpoint` against each fixed reference opponent.

        These are the opponents that will not cooperate with a stall, which is exactly why
        they are the veto rather than the readout. Rows that have saturated are re-run at a
        smaller size; the panel is a veto, and a veto only has to resolve a cliff.

        With `progress`, each row is journaled game by game while it plays and recorded once it
        is done, so a pause costs at most the games in flight of one row. A recorded row is
        reused only while the checkpoint's contents and the match are unchanged.
        """
        gate = self.config.gate
        digest = _sha256(checkpoint) if progress is not None else None
        scores: dict[str, float] = {}
        for i, opponent in enumerate(gate.reference):
            a = f"netmcts:{checkpoint}@{gate.sims}"
            games = self.reference_games_for(opponent)
            key = f"reference {i}"
            inputs = {"a": a, "b": opponent, "games": games, "checkpoint_sha256": digest,
                      "game": self.config.game.cli_flags()}
            if progress is not None and (done := progress.result(key, inputs)) is not None:
                scores[opponent] = float(done)
                continue
            journal = self._journal(prefix, f"ref{i}") if progress is not None else None
            scores[opponent] = self.play_match(a, opponent, games, journal=journal).score
            if progress is not None:
                progress.record(key, inputs, scores[opponent])
                journal.unlink(missing_ok=True)
        return scores

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
        """One generation, resumed from ``progress/`` if an earlier attempt was paused.

        See "Pausing" in the module docstring for what is saved where. The order of the steps
        at the end — decide, record the decision, act on it, write the log line, clear the
        progress — is load-bearing, and each comment below says why.
        """
        self.generation += 1
        g = self.generation
        gate = self.config.gate
        prefix = f"gen{g:03d}"
        say(f"\n── generation {g} " + "─" * 40)
        progress = self._progress(g)
        # A generation that got as far as deciding is finished in every way that matters, and
        # its commit is replayed from the record rather than re-derived — see below.
        decision = progress.data.get("decision")

        shard = self.shards / f"{prefix}.d52sp"
        sp_inputs = {
            "game": asdict(self.config.game),
            "selfplay": asdict(self.config.selfplay),
            "seed": self.config.run.seed,
        }
        sp = progress.result("selfplay", sp_inputs)
        if decision is not None:
            sp = progress.data["selfplay"]["result"]
        if sp is not None and shard.exists():
            restored = " · saved before the pause"
        else:
            # Recomputing self-play invalidates the fit that trained on the old shard. The panel
            # and the gate need no such help: they key on the candidate's contents.
            self._fit_path(g).unlink(missing_ok=True)
            journal = self._journal(prefix, "selfplay")
            shard, sp = self.selfplay(g, journal)
            progress.record("selfplay", sp_inputs, sp)
            # Only now. Deleted before the record, a pause in between would replay every game.
            journal.unlink(missing_ok=True)
            restored = ""
        say(
            f"  self-play   {sp['games']} games · {sp['games'] / max(sp['seconds'], 1e-9):.1f} g/s · "
            f"{_hms(sp['seconds'])} · {sp['samples']:,} decisions · "
            f"P0 {sp['p0']:.0f}% P1 {sp['p1']:.0f}% draw {sp['draw']:.0f}%{restored}"
        )
        self._check_stop()

        gen = self._replay_into_buffer(shard, g)
        won, drew, lost = self.buffer.value_target_mix()
        say(
            f"  buffer      {self.buffer.samples:,} samples over {len(self.buffer.generations)} "
            f"generation(s), {self.buffer.nbytes / 1e6:.0f} MB · "
            f"value targets win {won:.0%} draw {drew:.0%} loss {lost:.0%}"
        )
        self._check_stop()

        # Scaled to the buffer, not fixed: a constant step count is a different number of
        # passes over the data every time the window is a different size, and four passes
        # over a quarter-full buffer is how generation 1 of `runs/fourth` memorised its
        # shard. See `TrainSettings.epochs_per_generation`.
        steps = self.config.train.steps_for(self.buffer.samples)
        epochs = steps * self.config.train.batch_size / max(self.buffer.samples, 1)
        # Resumable, and kept until the generation commits: after a pause past this point it is
        # what restores the fitted weights, AdamW's moments and the batch RNG in one piece.
        stats = self.trainer.fit(
            self.buffer,
            self.rng,
            steps,
            generation=g,
            checkpoint=self._fit_path(g),
            save_every=self.config.run.save_every_secs,
            should_stop=lambda: self._stop_requested,
        )
        views = f" ×{self.augment.count} lane views" if self.augment is not None else ""
        say(
            f"  train       {stats.steps} steps · {epochs:.2f} epochs{views} · lr {stats.lr:.2e} · "
            f"policy {stats.policy_first:.3f} → {stats.policy_last:.3f} "
            f"(mean {stats.policy_mean:.3f}) · value {stats.value_first:.3f} → {stats.value_last:.3f} "
            f"(mean {stats.value_mean:.3f}) · {_hms(stats.seconds)}"
        )

        # The value head is the half that plateaued in the first run (`PLAN.md` §4.2 change
        # 7), and the training-batch number could not say so unambiguously because the
        # window slides underneath it. This one is fixed and was never trained on.
        held = self.trainer.evaluate(self.holdout) if self.holdout is not None else None
        if held is not None:
            say(
                f"  held-out    {held.samples:,} samples · value MSE {held.value_mse:.4f} · "
                f"policy {held.policy_loss:.3f}"
            )

        candidate = self.checkpoints / f"{prefix}.d52nn"
        self.trainer.save(candidate)
        self._check_stop()

        if decision is None:
            # The reference panel runs on the *candidate*, before the decision. Measuring the
            # winner afterwards — which is what this used to do — makes the strongest signal
            # available a report rather than a check. `FINDINGS.md` F3.6.
            reference = self.reference_scores(candidate, progress=progress, prefix=prefix)
        else:
            reference = decision["reference"]
        if reference:
            say(
                "  reference   "
                + " · ".join(
                    f"vs {name} {score:.3f}/{self.reference_games_for(name)}g"
                    + (f" (best {self.reference_best[name]:.3f})" if name in self.reference_best else "")
                    for name, score in reference.items()
                )
            )

        if decision is None:
            a, b = f"netmcts:{candidate}@{gate.sims}", f"netmcts:{self.best}@{gate.sims}"
            gate_inputs = {
                "a": a, "b": b, "games": gate.games, "game": self.config.game.cli_flags(),
                "candidate_sha256": _sha256(candidate), "best_sha256": _sha256(self.best),
            }
            played = progress.result("gate", gate_inputs)
            if played is None:
                journal = self._journal(prefix, "gate")
                mirror = self.play_match(a, b, gate.games, journal=journal)
                progress.record("gate", gate_inputs, asdict(mirror))
                journal.unlink(missing_ok=True)
            else:
                mirror = MatchResult(**played)
            promoted, why = self.judge(mirror, reference)
            # ⚠️ Recorded **before** it is acted on. Promotion overwrites `best.d52nn` with the
            # candidate, so a pause after the copy that re-ran the gate on resume would score
            # the candidate against itself — and the veto would be measured against a
            # high-water mark the copy had not yet been allowed to raise.
            decision = {
                "reference": reference,
                "mirror": asdict(mirror),
                "promoted": promoted,
                "why": why,
            }
            progress.data["decision"] = decision
            progress.save()
        mirror = MatchResult(**decision["mirror"])
        promoted, why = bool(decision["promoted"]), str(decision["why"])
        verdict = "PROMOTED" if promoted else "REFUSED"
        say(f"  gate        candidate vs best {mirror} · {why} → {verdict}")

        # From here to the log line nothing checks for a stop. It is a copy and three small
        # writes, and every one of them is safe to repeat if a SIGKILL lands in the middle.
        if promoted:
            copy_atomically(candidate, self.best)
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

        # The optimiser's moments are saved whether or not the candidate was promoted: a
        # refusal rolls back the weights and deliberately keeps the momentum.
        self.trainer.save_optimizer(self.optimizer_state)

        record = {
            "generation": g,
            # Across attempts, so a paused generation is not logged as a quick one.
            "seconds": progress.seconds(),
            "resumes": progress.resumes,
            "selfplay": sp,
            "buffer_samples": self.buffer.samples,
            "value_mix": {"win": won, "draw": drew, "loss": lost},
            "lr": stats.lr,
            "steps": stats.steps,
            "epochs": epochs,
            "policy_loss": stats.policy_mean,
            "value_loss": stats.value_mean,
            "holdout_samples": held.samples if held else 0,
            "holdout_policy_loss": held.policy_loss if held else None,
            "holdout_value_mse": held.value_mse if held else None,
            "gate_score": mirror.score,
            "gate_decisive_score": mirror.decisive_score,
            "gate_decisive_games": mirror.decisive,
            "gate_decisive_ci95": mirror.decisive_ci95,
            "gate_ci95": mirror.ci95,
            "gate_reason": why,
            "promoted": promoted,
            "refusals": self.refusals,
            "benchmarks": {k: round(v, 4) for k, v in reference.items()},
            "checkpoint": str(candidate),
            "samples_kept": gen.samples,
            # In the record and not beside it, because this line is the commit — see
            # `_restore_rng`. It is the state the next generation's fit starts from.
            "rng_state": self.rng.bit_generator.state,
        }
        # **The commit.** Before this line a resume replays the generation from `progress/`;
        # after it, the generation is history and its progress is garbage.
        append_line(self.log_path, json.dumps(record))
        self.history.append(record)
        progress.path.unlink(missing_ok=True)
        for leftover in self.progress_dir.glob(f"{prefix}-*"):
            leftover.unlink()
        return record

    def run(self) -> None:
        """Run generations until the budget, the cap or the refusal streak says stop.

        SIGTERM and SIGHUP — what a scheduler sends before it reclaims a box, and what a dropped
        SSH session sends — stop the run at the next safe point instead of wherever the signal
        happened to land: the engine is told to stop, a fit saves the step it is on, and the
        progress file has everything else. Ctrl-C keeps its old, immediate meaning.
        """
        handlers = {}
        for name in ("SIGTERM", "SIGHUP"):
            sig = getattr(signal, name, None)
            # An ignored signal stays ignored: under `nohup` SIGHUP is, and a run started that
            # way is meant to outlive the SSH session rather than stop politely when it drops.
            if sig is None or signal.getsignal(sig) is signal.SIG_IGN:
                continue
            try:
                handlers[sig] = signal.signal(sig, self._request_stop)
            except ValueError:  # not the main thread, e.g. under a test runner's worker
                pass
        try:
            self._run()
        finally:
            for sig, previous in handlers.items():
                signal.signal(sig, previous)

    def _run(self) -> None:
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
                print(
                    f"\n{'stopped by signal' if self._stop_requested else 'interrupted'} in "
                    f"generation {self.generation} — what it finished is saved in "
                    f"{self.progress_dir}, and --resume carries on from there",
                    file=sys.stderr,
                    flush=True,
                )
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
            f"{'gen':>4} {'draw%':>6} {'decisive':>16} {'promo':>6} {'policy':>8} {'value':>7} "
            f"{'held-out':>9}  reference"
        )
        for h in self.history:
            marks = " ".join(f"{k}={v:.3f}" for k, v in h.get("benchmarks", {}).items())
            decisive = h.get("gate_decisive_games", 0)
            score = h.get("gate_decisive_score")
            # Score and interval together: a gate whose interval straddles the threshold
            # did not decide anything, and F3.7 is three of those in a row.
            cell = f"{score:.3f}±{h.get('gate_decisive_ci95', 0.0):.3f}" if decisive else "—"
            mse = h.get("holdout_value_mse")
            say(
                f"{h['generation']:>4} {h['selfplay'].get('draw', 0):>5.0f}% "
                f"{cell:>16} {'yes' if h['promoted'] else 'no':>6} "
                f"{h['policy_loss']:>8.3f} {h['value_loss']:>7.3f} "
                f"{(f'{mse:.4f}' if mse is not None else '—'):>9}  {marks}"
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


def run_loop(
    config: TrainConfig,
    run_dir: Path,
    *,
    resume: bool = False,
    init_from: str | Path | None = None,
) -> None:
    TrainingLoop(config, run_dir, resume=resume, init_from=init_from).run()
