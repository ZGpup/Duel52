"""``python -m duel52.rnad`` — the R-NaD learner. ``PLAN.md`` item 8.

    python -m duel52.rnad check --config configs/rnad-3h.toml
    python -m duel52.rnad bench --config configs/rnad-3h.toml --steps 3
    python -m duel52.rnad run   --config configs/rnad-3h.toml --run-dir runs/rnad-3h
    python -m duel52.rnad run   --config configs/rnad-3h.toml --run-dir runs/rnad-3h --resume

``check`` validates the config and prints what a run will do, in seconds. ``bench`` times a
few real steps on this machine and says how many regularisation iterations the run's clock
buys at the configured schedule — run it first on any new box, because the schedule is sized
in learner steps and a GPU and a laptop take very different numbers of them per hour.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .config import RNaDConfig, load_config

#: A trained Duel 52 game offers ~140 non-forced decisions (a random one ~131); used only to
#: estimate memory before anything has been played.
DECISIONS_PER_GAME = 140


def _describe(config: RNaDConfig) -> None:
    from ..nn.model import spec_for
    from ..train.trainer import resolve_device
    from .core import EntropySchedule
    from .loop import build_model

    spec = spec_for(config.game.variant, config.game.encoding_slots, config.game.rules_file)
    model = build_model(config, spec)
    params = sum(p.numel() for p in model.parameters())
    r, run = config.rnad, config.run
    engine = Path(run.engine)

    print(f"config      {config.source}")
    print(f"device      {resolve_device(r.device)}")
    game = f"rules_file={config.game.rules_file}" if config.game.rules_file else f"variant={config.game.variant}"
    print(f"game        {game} encoding_slots={config.game.encoding_slots} stalemate_value={config.game.stalemate_value}")
    print(f"rules       {spec['rules_name']}/{spec['rules_hash']}")
    print(f"encoding    obs_dim={spec['obs_dim']} action_dim={spec['action_dim']} obs_layout_hash={spec['obs_layout_hash']}")
    print(f"net         arch={config.net.arch} width={config.net.width} blocks={config.net.blocks} "
          f"value_hidden={config.net.value_hidden} value_head=linear · {params:,} parameters")
    print(f"engine      {engine}{'' if engine.exists() else '   *** MISSING — cargo build --release'}")
    if engine.exists():
        version = subprocess.run([str(engine), "version"], capture_output=True, text=True)
        print(f"            {version.stdout.strip()}")

    positions = r.batch_games * DECISIONS_PER_GAME
    obs_mb = positions * spec["obs_dim"] * 4 / 1e6
    logits_mb = positions * spec["action_dim"] * 4 / 1e6
    chunk = r.learner_chunk if r.learner_chunk > 0 else positions
    print(
        f"\nper step    {r.batch_games} games ≈ {positions:,} positions · observations ≈ {obs_mb:,.0f} MB, "
        f"one logit tensor ≈ {logits_mb:,.0f} MB on the device"
    )
    print(f"            learner forward chunk {min(chunk, positions):,} positions")
    print(
        f"learner     eta={r.eta_reward_transform} lr={r.learning_rate} adam=({r.adam_b1}, {r.adam_b2}, {r.adam_eps}) "
        f"target_avg={r.target_network_avg} nerd_beta={r.nerd_beta} c={r.c_vtrace} rho={r.rho_vtrace}"
    )
    schedule = EntropySchedule(sizes=r.entropy_schedule_size, repeats=r.entropy_schedule_repeats)
    horizon = 5 * sum(s * n for s, n in zip(r.entropy_schedule_size, r.entropy_schedule_repeats))
    updates = [n for n in range(horizon) if schedule(n)[1]][:5]
    print(f"schedule    the regularisation policy moves after learner steps {updates} …")
    _warn_coupling(r)
    cap = f", or {run.max_steps} steps" if run.max_steps else ""
    print(f"run         {run.hours:g} h{cap} · seed {run.seed} · log every {run.log_every} steps · save every {run.save_every_secs:g} s")
    evals = f"every {run.eval_every} steps" if run.eval_every else "at the end"
    print(f"eval        {evals}: netsample vs {', '.join(run.eval_opponents)}, {run.eval_games} games each")


def _warn_coupling(r) -> None:
    """How far the target net can travel in one regularisation iteration.

    Each update copies the *target* net into the regularisation policy, and the target moves
    by ``target_network_avg`` of the gap per step — so over an iteration it covers roughly
    ``1 − exp(−τ·size)`` of the way to the online net. The reference runs at τ·size = 20.
    Shrink the schedule without raising τ and the regularisation policy stays pinned near
    the random initial network, pulling the policy back to it every update.
    """
    coupling = r.target_network_avg * min(r.entropy_schedule_size)
    print(f"            target_network_avg × entropy_schedule_size = {coupling:g} (the reference runs at 20)")
    if coupling < 5:
        print(
            "            ⚠️ the target net covers only "
            f"{1 - np.exp(-coupling):.0%} of the way to the online net per iteration, so the "
            "regularisation policy stays near its start — raise target_network_avg"
        )


def _check(args: argparse.Namespace) -> int:
    _describe(load_config(args.config))
    return 0


def _bench(args: argparse.Namespace) -> int:
    import torch

    from ..nn.model import spec_for
    from ..train.trainer import resolve_device
    from .actor import play_batch
    from .core import EntropySchedule
    from .learner import RNaDLearner
    from .loop import build_model

    config = load_config(args.config)
    r, run = config.rnad, config.run
    spec = spec_for(config.game.variant, config.game.encoding_slots, config.game.rules_file)
    device = resolve_device(r.device)
    torch.manual_seed(run.seed)
    learner = RNaDLearner(build_model(config, spec), r, device)
    print(f"bench       {config.source} · device {device} · {r.batch_games} games/step · {args.steps} timed steps after 1 warm-up")

    acts, learns, decisions = [], [], []
    for n in range(args.steps + 1):
        t0 = time.perf_counter()
        traj = play_batch(
            learner.online, games=r.batch_games, seed=run.seed + n * r.batch_games,
            rng=np.random.default_rng([run.seed, n]), game=config.game, device=device,
            threads=run.threads, chunk=r.actor_chunk,
        )
        if device.type != "cpu":
            getattr(torch, device.type).synchronize()
        t1 = time.perf_counter()
        learner.step(traj)
        if device.type != "cpu":
            getattr(torch, device.type).synchronize()
        t2 = time.perf_counter()
        if n:
            acts.append(t1 - t0)
            learns.append(t2 - t1)
            decisions.append(traj.decisions)
        print(f"  step {n}{' (warm-up)' if n == 0 else ''}: act {t1 - t0:.2f}s · learn {t2 - t1:.2f}s · {traj.decisions:,} decisions")

    act, learn = float(np.mean(acts)), float(np.mean(learns))
    per_step = act + learn
    steps_per_hour = 3600.0 / per_step
    total = int(steps_per_hour * run.hours)
    schedule = EntropySchedule(sizes=r.entropy_schedule_size, repeats=r.entropy_schedule_repeats)
    iterations = schedule.iterations_by(min(total, 2_000_000))
    print(
        f"\nper step    act {act:.2f}s ({act / per_step:.0%}) · learn {learn:.2f}s ({learn / per_step:.0%}) · "
        f"{np.mean(decisions) / act:,.0f} decisions/s acting · {r.batch_games / per_step:.1f} games/s end to end"
    )
    print(f"clock       {steps_per_hour:,.0f} steps/hour → ~{total:,} steps in the run's {run.hours:g} h")
    print(f"schedule    {iterations} regularisation updates at entropy_schedule_size={r.entropy_schedule_size}")
    _warn_coupling(r)
    if total > 0:
        suggested = max(1, total // 10)
        tau = min(0.1, max(r.target_network_avg, 10.0 / suggested))
        print(
            f"            for ~10 updates in {run.hours:g} h: entropy_schedule_size = [{suggested}], "
            f"target_network_avg = {tau:.3g} (τ·size = {tau * suggested:.0f})"
        )
    return 0


def _run(args: argparse.Namespace) -> int:
    from .loop import RNaDRun

    config = load_config(args.config)
    RNaDRun(
        config, args.run_dir, resume=args.resume, hours=args.hours, max_steps=args.max_steps
    ).run()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m duel52.rnad", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="act → learn, until the clock or the step cap")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--run-dir", type=Path, required=True)
    run.add_argument("--resume", action="store_true", help="continue the run in --run-dir")
    run.add_argument("--hours", type=float, default=None, help="override run.hours")
    run.add_argument("--max-steps", type=int, default=None, help="override run.max_steps")
    run.set_defaults(func=_run)

    check = sub.add_parser("check", help="validate the config and print what a run will do")
    check.add_argument("--config", type=Path, required=True)
    check.set_defaults(func=_check)

    bench = sub.add_parser("bench", help="time real steps on this machine and size the schedule")
    bench.add_argument("--config", type=Path, required=True)
    bench.add_argument("--steps", type=int, default=3)
    bench.set_defaults(func=_bench)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
