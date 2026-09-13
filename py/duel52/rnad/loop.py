"""The R-NaD run — ``PLAN.md`` item 8.

Per learner step, in one process on one device:

1. **Act.** Play ``rnad.batch_games`` fresh games through the engine's ``GameBatch`` with the
   online network on both seats (``actor.play_batch``).
2. **Learn.** One R-NaD step on exactly those games (``learner.RNaDLearner.step``).

No promotion gate: R-NaD follows its own iterates. Every ``run.eval_every`` steps the target
network is written as a ``.d52nn`` checkpoint and plays ``run.eval_opponents`` through
``duel52 match --a netsample:<checkpoint>``, and the scores go into the log beside the
losses.

Run directory::

    config.toml        the config the run was started with
    log.jsonl          one line per step, and one per evaluation
    state.pt           all four networks, the optimiser, the step counter, elapsed time
    checkpoints/       stepNNNNNN.d52nn at each evaluation, and latest.d52nn

**Stopping and resuming.** Ctrl-C, SIGTERM or SIGHUP finishes the step in progress, saves
``state.pt`` and exits; ``--resume`` continues from it. Step ``n``'s games and samples are
seeded from ``(run.seed, n)``, so a resumed run plays the same streams it would have. On a
CPU that makes a resumed run identical to an uninterrupted one; on a GPU the network's
arithmetic is not bit-deterministic, so it is the same experiment rather than the same bits.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import signal
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..nn.checkpoint import write_checkpoint
from ..nn.model import NetConfig, build_net, lane_spec_for, spec_for
from ..train.durable import append_line, copy_atomically, replace_atomically
from ..train.trainer import resolve_device
from .actor import play_batch
from .config import RNaDConfig
from .learner import RNaDLearner

__all__ = ["RNaDRun", "build_model", "match_score"]

_SCORE = re.compile(r"score for .*?:\s*([0-9.]+)\s*\+/-\s*([0-9.]+)")
_WLD = re.compile(r"W(\d+)\s+L(\d+)\s+D(\d+)")


def say(*args: object) -> None:
    print(*args, flush=True)


def clock(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m{secs:02d}s"


def build_model(config: RNaDConfig, spec: dict) -> torch.nn.Module:
    """The trunk ``[net]`` names, with R-NaD's linear value head."""
    net = NetConfig(
        obs_dim=spec["obs_dim"],
        action_dim=spec["action_dim"],
        width=config.net.width,
        blocks=config.net.blocks,
        value_hidden=config.net.value_hidden,
        arch=config.net.arch,
        value_head="linear",
    )
    lanes = (
        lane_spec_for(config.game.variant, config.game.encoding_slots, config.game.rules_file)
        if net.arch == "lane"
        else None
    )
    return build_net(net, lanes)


def match_score(stdout: str) -> dict:
    score, wld = _SCORE.search(stdout), _WLD.search(stdout)
    if not score or not wld:
        raise RuntimeError(f"could not read a score out of:\n{stdout}")
    return {
        "score": float(score.group(1)),
        "ci95": float(score.group(2)),
        "wins": int(wld.group(1)),
        "losses": int(wld.group(2)),
        "draws": int(wld.group(3)),
    }


class RNaDRun:
    def __init__(
        self,
        config: RNaDConfig,
        run_dir: Path,
        *,
        resume: bool = False,
        hours: float | None = None,
        max_steps: int | None = None,
    ):
        self.config = config
        self.run_dir = Path(run_dir)
        self.hours = config.run.hours if hours is None else hours
        self.max_steps = config.run.max_steps if max_steps is None else max_steps
        self.state_path = self.run_dir / "state.pt"
        self.log_path = self.run_dir / "log.jsonl"
        self.checkpoints = self.run_dir / "checkpoints"

        if resume and not self.state_path.exists():
            raise FileNotFoundError(f"--resume, but {self.state_path} does not exist")
        if not resume and self.state_path.exists():
            raise FileExistsError(f"{self.run_dir} already holds a run; pass --resume to continue it")

        self.spec = spec_for(config.game.variant, config.game.encoding_slots, config.game.rules_file)
        self.device = resolve_device(config.rnad.device)
        torch.manual_seed(config.run.seed)
        self.learner = RNaDLearner(build_model(config, self.spec), config.rnad, self.device)
        self.elapsed = 0.0
        self.last_eval_step = -1
        self._stop = False

        self.run_dir.mkdir(parents=True, exist_ok=True)
        if resume:
            state = torch.load(self.state_path, map_location=self.device, weights_only=False)
            self.learner.load_state_dict(state["learner"])
            self.elapsed = float(state["elapsed"])
            self.last_eval_step = int(state.get("last_eval_step", -1))
        elif config.source != "<defaults>":
            shutil.copyfile(config.source, self.run_dir / "config.toml")

    # ------------------------------------------------------------------ plumbing --

    def _install_signals(self) -> None:
        def request_stop(signum, _frame):
            if self._stop:
                raise KeyboardInterrupt
            self._stop = True
            say(f"\n{signal.Signals(signum).name}: finishing this step, then saving and stopping")

        for name in ("SIGINT", "SIGTERM", "SIGHUP"):
            sig = getattr(signal, name, None)
            if sig is not None and signal.getsignal(sig) is not signal.SIG_IGN:
                signal.signal(sig, request_stop)

    def _log(self, record: dict) -> None:
        append_line(self.log_path, json.dumps(record))

    def save(self) -> None:
        state = {
            "learner": self.learner.state_dict(),
            "elapsed": self.elapsed,
            "last_eval_step": self.last_eval_step,
            "config": self.config.as_dict(),
        }
        replace_atomically(self.state_path, lambda tmp: torch.save(state, tmp))

    def write_target(self) -> Path:
        path = self.checkpoints / f"step{self.learner.steps:06d}.d52nn"
        target = copy.deepcopy(self.learner.target).cpu()
        replace_atomically(path, lambda tmp: write_checkpoint(tmp, model=target, spec=self.spec, learner="rnad"))
        copy_atomically(path, self.checkpoints / "latest.d52nn")
        return path

    def evaluate(self) -> dict:
        run = self.config.run
        checkpoint = self.write_target()
        results = {}
        engine = Path(run.engine)
        if not engine.exists():
            say(f"  eval       skipped — {engine} is not built (cargo build --release)")
            return results
        for opponent in run.eval_opponents:
            args = [
                str(engine), "match", "--a", f"netsample:{checkpoint}", "--b", opponent,
                "--games", str(run.eval_games), "--seed", str(run.seed),
                *self.config.game.cli_flags(),
            ]
            if run.eval_threads > 0:
                args += ["--threads", str(run.eval_threads)]
            done = subprocess.run(args, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"evaluation match failed: {done.stderr.strip() or done.stdout.strip()}")
            results[opponent] = match_score(done.stdout)
        self.last_eval_step = self.learner.steps
        line = " · ".join(f"vs {o} {r['score']:.3f}±{r['ci95']:.3f}" for o, r in results.items())
        say(f"  eval       step {self.learner.steps} · {checkpoint.name} · {line}")
        self._log({"kind": "eval", "step": self.learner.steps, "checkpoint": str(checkpoint), **results})
        return results

    # ----------------------------------------------------------------------- run --

    def step(self) -> dict:
        cfg, run = self.config, self.config.run
        n = self.learner.steps
        games = cfg.rnad.batch_games
        started = time.perf_counter()
        traj = play_batch(
            self.learner.online,
            games=games,
            seed=run.seed + n * games,
            rng=np.random.default_rng([run.seed, n]),
            game=cfg.game,
            device=self.device,
            threads=run.threads,
            chunk=cfg.rnad.actor_chunk,
        )
        acted = time.perf_counter()
        stats = self.learner.step(traj)
        learned = time.perf_counter()

        outcomes = traj.outcomes
        record = {
            "kind": "step",
            **asdict(stats),
            "games": traj.games,
            "decisions": traj.decisions,
            "decisions_per_game": traj.decisions / max(1, traj.games),
            "p0_wins": outcomes.count("p0_wins") / games,
            "p1_wins": outcomes.count("p1_wins") / games,
            "draws": 1.0 - (outcomes.count("p0_wins") + outcomes.count("p1_wins")) / games,
            "actor_secs": acted - started,
            "learner_secs": learned - acted,
        }
        self.elapsed += learned - started
        record["elapsed"] = self.elapsed
        self._log(record)
        return record

    def run(self) -> None:
        cfg, run = self.config, self.config.run
        self._install_signals()
        say(
            f"R-NaD run {self.run_dir} · device {self.device} · {cfg.net.arch} "
            f"{cfg.net.width}x{cfg.net.blocks} · {cfg.rnad.batch_games} games/step · "
            f"budget {self.hours:g} h" + (f", {self.max_steps} steps" if self.max_steps else "")
        )
        if self.learner.steps:
            say(f"resuming at step {self.learner.steps}, {clock(self.elapsed)} elapsed")

        last_save = time.monotonic()
        while not self._stop:
            if self.max_steps and self.learner.steps >= self.max_steps:
                break
            if self.elapsed >= self.hours * 3600:
                break
            r = self.step()
            step = self.learner.steps
            if step % max(1, run.log_every) == 0 or step == 1:
                say(
                    f"step {step:>6} · {r['decisions_per_game']:.0f} dec/game · "
                    f"P0 {r['p0_wins']:.0%} draw {r['draws']:.0%} · "
                    f"act {r['actor_secs']:.2f}s learn {r['learner_secs']:.2f}s · "
                    f"loss v {r['loss_value']:.3f} nerd {r['loss_nerd']:+.4f} · "
                    f"v̂ [{r['v_target_min']:+.2f}, {r['v_target_max']:+.2f}] · "
                    f"H {r['policy_entropy']:.2f} · KL {r['kl_to_reg']:.4f} · "
                    f"α {r['alpha']:.2f} · reg {r['reg_updates']} · {clock(self.elapsed)}"
                )
            if run.eval_every and step % run.eval_every == 0:
                self.evaluate()
            if time.monotonic() - last_save >= run.save_every_secs:
                self.save()
                last_save = time.monotonic()

        self.save()
        if self.learner.steps and self.last_eval_step != self.learner.steps and not self._stop:
            self.evaluate()
            self.save()
        elif self.learner.steps:
            self.write_target()
        state = "stopped" if self._stop else "finished"
        say(
            f"{state} at step {self.learner.steps} · {clock(self.elapsed)} · "
            f"state {self.state_path} · checkpoint {self.checkpoints / 'latest.d52nn'}"
        )
