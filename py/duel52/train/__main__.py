"""``python -m duel52.train`` — the Phase 3 training loop.

    python -m duel52.train run --config configs/train-fast.toml --run-dir runs/first
    python -m duel52.train run --config configs/train-fast.toml --run-dir runs/first --resume
    python -m duel52.train check --config configs/train-fast.toml

``check`` is the five-second version of the run: it validates the config, resolves the
device, confirms the engine binary is there and agrees about the encoding layout, and prints
what one generation is expected to cost. Worth running before a two-hour session.
"""

from __future__ import annotations

import argparse
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
        f"net         width={config.net.width} blocks={config.net.blocks} "
        f"value_hidden={config.net.value_hidden}"
    )
    if engine.exists():
        version = subprocess.run([str(engine), "version"], capture_output=True, text=True)
        print(f"            {version.stdout.strip()}")

    sp = config.selfplay
    print(
        f"\nper generation: {sp.games} self-play games at {sp.sims} sims, "
        f"{config.train.steps_per_generation} training steps of {config.train.batch_size}, "
        f"a {config.gate.games}-game gate"
    )
    print(
        f"budget:         {config.run.generations} generations or {config.run.hours} hours, "
        f"whichever comes first"
    )
    if not engine.exists():
        return 1
    return 0


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
    run_loop(config, run_dir, resume=args.resume)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m duel52.train", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the self-play → train → gate loop")
    run.add_argument("--config", type=Path, default=None, help="training TOML (default: built-in defaults)")
    run.add_argument("--run-dir", type=Path, required=True, help="where checkpoints, shards and the log go")
    run.add_argument("--resume", action="store_true", help="continue an existing run directory")
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
