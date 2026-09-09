"""``python -m duel52.train`` — the Phase 3 training loop.

    python -m duel52.train run --config configs/train-fast.toml --run-dir runs/first
    python -m duel52.train run --config configs/train-fast.toml --run-dir runs/first --resume
    python -m duel52.train check --config configs/train-fast.toml

    # continue from a shipped checkpoint rather than from a random init
    python -m duel52.train run --config configs/train-2h.toml --run-dir runs/fourth \
        --init-from models/duel52-split-gen016.d52nn

``check`` is the five-second version of the run: it validates the config, resolves the
device, confirms the engine binary is there and agrees about the encoding layout, and prints
what one generation is expected to cost. Worth running before a two-hour session.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from ..nn.model import spec_for
from .config import TrainConfig, load_config
from .loop import TrainingLoop, run_loop
from .trainer import resolve_device


def _check(args: argparse.Namespace) -> int:
    config: TrainConfig = load_config(args.config)
    spec = spec_for(config.game.variant, config.game.encoding_slots)
    engine = Path(config.run.engine)

    print(f"config      {config.source}")
    print(f"device      {resolve_device(config.train.device)}")
    print(f"engine      {engine}{'' if engine.exists() else '   *** MISSING — cargo build --release'}")
    print(f"game        variant={config.game.variant} encoding_slots={config.game.encoding_slots}")
    print(f"encoding    obs_dim={spec['obs_dim']} action_dim={spec['action_dim']}")
    print(f"            obs_layout_hash={spec['obs_layout_hash']}")
    print(f"            action_layout_hash={spec['action_layout_hash']}")
    print(
        f"net         arch={config.net.arch} width={config.net.width} "
        f"blocks={config.net.blocks} value_hidden={config.net.value_hidden}"
    )
    if config.net.arch == "lane":
        # The equivariance is the reason the architecture exists, so `check` says so rather
        # than leaving "lane" to be looked up.
        print(
            "            lane-equivariant: the trunk runs once per lane with shared weights,"
        )
        print(
            "            so a lane preference (FINDINGS.md F4.3) is unrepresentable rather"
        )
        print("            than merely small. Warm-starting from an 'mlp' checkpoint is refused.")
    if engine.exists():
        version = subprocess.run([str(engine), "version"], capture_output=True, text=True)
        print(f"            {version.stdout.strip()}")

    sp = config.selfplay
    tr = config.train
    fitting = (
        f"{tr.epochs_per_generation:.2f} epochs of the buffer"
        if tr.epochs_per_generation > 0
        else f"{tr.steps_per_generation} training steps"
    )
    search = (
        f"{sp.sims} sims"
        if sp.full_search_fraction >= 1.0
        else f"{sp.mean_sims:.0f} sims on average"
    )
    print(
        f"\nper generation: {sp.games} self-play games at {search}, "
        f"{fitting} in batches of {tr.batch_size}, a {config.gate.games}-game gate"
    )
    if sp.full_search_fraction < 1.0:
        # The two numbers that decide whether the technique is paying: how much cheaper
        # self-play got, and how many policy targets are left to pay for it with.
        speedup = sp.sims / sp.mean_sims
        print(
            f"                playout cap: {sp.full_search_fraction:.0%} of decisions get "
            f"{sp.sims} sims and a policy target,"
        )
        print(
            f"                the rest get {sp.cap_sims} and a value target only — "
            f"{speedup:.1f}x cheaper search,"
        )
        print(
            f"                ~{sp.games * 136 // tr.sample_stride * sp.full_search_fraction:,.0f} "
            f"policy targets a generation against "
            f"{sp.games * 136 // tr.sample_stride:,} value targets"
        )
    if sp.eval_batch > 1:
        # `selfplay.rs` clamps the batch to the games a worker actually owns, so a small
        # generation on many cores silently gets a smaller batch than the config asks for.
        # Say the effective number, because the clamp is invisible in the run's output.
        workers = config.run.threads or os.cpu_count() or 1
        per_worker = max(sp.games // max(workers, 1), 1)
        effective = min(sp.eval_batch, per_worker)
        line = (
            f"eval batch:     {sp.eval_batch} games in flight per worker — batched "
            f"evaluation, and byte-identical to 1"
        )
        print(f"                {line}")
        if effective < sp.eval_batch:
            print(
                f"                ⚠️  only {effective} in practice: {sp.games} games over "
                f"{workers} threads is {per_worker} each"
            )
    # What a fixed step count actually means depends on how full the buffer is, and the
    # answer at generation 1 is what cost `runs/fourth` its first generation.
    if tr.epochs_per_generation <= 0:
        per_gen = sp.games * 136 // tr.sample_stride
        full = min(per_gen * tr.buffer_generations, tr.buffer_samples)
        presentations = tr.steps_per_generation * tr.batch_size
        print(
            f"                which is {presentations / max(per_gen - tr.holdout_samples, 1):.1f} epochs "
            f"at generation 1 and {presentations / max(full, 1):.1f} once the buffer is full"
        )
    print(
        f"budget:         {config.run.generations} generations or {config.run.hours} hours, "
        f"whichever comes first"
    )
    schedule = config.train.lr_schedule
    print(
        "lr:             "
        + (
            f"{config.train.lr:.2e} constant"
            if not schedule
            else " · ".join(f"gen {int(a)}+ → {config.train.lr * float(b):.2e}" for a, b in schedule)
        )
    )
    print(
        "held-out:       "
        + (
            "none — the value loss is a training-batch number only"
            if config.train.holdout_samples <= 0
            else f"{config.train.holdout_samples:,} samples off generation "
            f"{config.train.holdout_generation}, scored from then on"
        )
    )
    # ⚠️ `FINDINGS.md` F4.6: a from-scratch run's generation 1 is played by a random init, and
    # its holdout becomes noise within a few generations — `runs/sixth` ended at a held-out
    # value MSE of 1.002, which is what predicting zero scores. Five seconds is the right
    # place to catch that, since the alternative is finding out at the end of a 24-hour run.
    if config.train.holdout_samples > 0 and config.train.holdout_generation == 1:
        print(
            "                ⚠️  generation 1 — right for a warm start, and noise for a\n"
            "                    from-scratch run (FINDINGS.md F4.6). If this run has no\n"
            "                    --init-from, set train.holdout_generation later."
        )

    # Building the tables here rather than only describing them is deliberate: a wrong or
    # mis-shaped permutation table does not crash a run, it just trains the net on somebody
    # else's targets (`PLAN.md` §4.2a). Five seconds is the right place to find that out.
    if not config.train.lane_augment and config.net.arch == "lane":
        # Off *because* the architecture already has it, which is a different fact from
        # having forgotten to turn it on — and the readout should not make the two look
        # alike. See `test_lane_augmentation_is_a_no_op_on_the_lane_equivariant_network`.
        print(
            "augmentation:   off, and correctly so — the lane-equivariant network satisfies\n"
            "                f(σ·x) = σ·f(x), so relabelling a sample gives the identical\n"
            "                loss and the identical gradient. It would be a gather for nothing."
        )
    elif not config.train.lane_augment:
        print("augmentation:   off — every sample is seen in one lane labelling only")
    else:
        from .buffer import LaneAugmenter

        aug = LaneAugmenter.from_engine(config.game.variant, config.game.encoding_slots)
        aug.check(int(spec["obs_dim"]), int(spec["action_dim"]))
        print(
            f"augmentation:   lane permutations on — {aug.count} exact relabellings per sample, "
            f"one drawn per draw\n"
            f"                (PLAN.md §4.2a; the holdout stays unaugmented)"
        )

    # `PLAN.md` §4.2 change 1: the first run ended on three refusals that were every one of
    # them inside their own confidence interval of even. What a gate can and cannot resolve
    # is arithmetic, and it belongs here rather than in a table in a plan that drifts.
    gate = config.gate
    print(
        f"gate power:     at {gate.games} decisive games, promoting at ≥ {gate.threshold} —\n"
        f"                a candidate that is really 0.500 passes {_passes(gate.threshold, 0.500, gate.games):.0%} "
        f"of the time (a wasted promotion, which costs ~nothing)\n"
        f"                a candidate that is really 0.540 passes {_passes(gate.threshold, 0.540, gate.games):.0%} "
        f"of the time (a refused improvement, and "
        f"{gate.max_consecutive_refusals} in a row ends the run)"
    )
    # A panel row that has saturated gets fewer games (`GateSettings.games_for`). Whether
    # that is still a veto is arithmetic, and it belongs next to the gate's power rather
    # than in a comment in a config.
    if not gate.reference:
        print("panel:          none — the mirror gate decides alone")
    elif gate.reference_games_saturated <= 0:
        print(f"panel:          {len(gate.reference)} rows at {gate.reference_games} games each")
    else:
        n = gate.reference_games_saturated
        tol = gate.reference_tolerance
        print(
            f"panel:          {len(gate.reference)} rows at {gate.reference_games} games, "
            f"{n} once a row's best reaches {gate.reference_saturated_at:.2f} —\n"
            f"                a trimmed row on a 1.000 best false-vetoes "
            f"{_veto(n, 1.0 - 0.005, 1.0, tol):.2%} of the time if it is really 0.995\n"
            f"                and catches a fall to 0.900 {_veto(n, 0.900, 1.0, tol):.0%} of the "
            f"time, to 0.600 {_veto(n, 0.600, 1.0, tol):.0%}"
        )

    if not engine.exists():
        return 1
    return 0


def _veto(games: int, true_score: float, best: float, tolerance: float) -> float:
    """P(a reference row vetoes) at `games`, when it is really `true_score`.

    The veto fires below ``best - tolerance``, so this is one binomial tail. It is what says
    whether a trimmed row is still a catastrophe detector or has become a coin flip.
    """
    from math import comb

    threshold = best - tolerance
    return sum(
        comb(games, k) * true_score**k * (1.0 - true_score) ** (games - k)
        for k in range(games + 1)
        if k / games < threshold - 1e-12
    )


def _passes(threshold: float, true_score: float, games: int) -> float:
    """P(the measured decisive score clears `threshold`) for a candidate that is really
    `true_score`, normal approximation, assuming the mirror match decides every game —
    which every generation of the third run did."""
    import math

    se = math.sqrt(max(true_score * (1.0 - true_score), 1e-12) / max(games, 1))
    return 0.5 * math.erfc((threshold - true_score) / (se * math.sqrt(2.0)))


def _run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    overrides = {}
    if args.hours is not None:
        overrides["hours"] = args.hours
    if args.generations is not None:
        overrides["generations"] = args.generations
    if overrides:
        from dataclasses import replace

        config = replace(config, run=replace(config.run, **overrides))

    run_dir = Path(args.run_dir)
    if run_dir.exists() and not args.resume and any(run_dir.iterdir()):
        print(
            f"{run_dir} already has a run in it. Pass --resume to continue it, or pick "
            f"another --run-dir.",
            file=sys.stderr,
        )
        return 1
    if args.init_from is not None and args.resume:
        print(
            "--init-from starts a run from a checkpoint; --resume continues one that "
            "already has an incumbent. Pass one or the other.",
            file=sys.stderr,
        )
        return 1
    run_loop(config, run_dir, resume=args.resume, init_from=args.init_from)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m duel52.train", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the self-play → train → gate loop")
    run.add_argument("--config", type=Path, default=None, help="training TOML (default: built-in defaults)")
    run.add_argument("--run-dir", type=Path, required=True, help="where checkpoints, shards and the log go")
    run.add_argument("--resume", action="store_true", help="continue an existing run directory")
    run.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="start from this .d52nn instead of a random init, keeping it as the first "
        "incumbent (it must match [net] and the encoding layout)",
    )
    run.add_argument("--hours", type=float, default=None, help="override the wall-clock budget")
    run.add_argument("--generations", type=int, default=None, help="override the generation cap")
    run.set_defaults(func=_run)

    check = sub.add_parser("check", help="validate the config and print what a run would cost")
    check.add_argument("--config", type=Path, default=None)
    check.set_defaults(func=_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
