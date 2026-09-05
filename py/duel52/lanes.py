"""Is the network equivariant under the three lanes? — ``FINDINGS.md`` F4.3, as a command.

    .venv/bin/python -m duel52.lanes --checkpoint models/duel52-split-gen031.d52nn

``PLAN.md`` §4.2a: Duel 52 is invariant under any permutation of its lanes. No rule in
``game_rules.md`` names a lane, orders them, or tells one from another, so a position and
its relabellings have the same value and the same optimal policy. A network that has not
learned that is spending parameters on an arbitrary preference — F4.3 measured gen022
preferring lane 3 in 24 of 24 seeds — and §4.2a's augmentation is the fix.

This is the **regression metric** that says whether the fix took. It needs no opponent, no
ladder and about two minutes, and ``PLAN.md`` §4.2a is explicit about when to run it:
⚠️ *on generation 1's checkpoint, not at the end* — if the numbers have not moved, the
augmentation is not wired up and the run is a bug report rather than a result.

⚠️ **Read the policy-TV row at generation 1, not the agreement row.** ``FINDINGS.md`` F4.5:
one generation into the run that fixed this, agreement still read 86/128 against gen022's
82/128 — inside noise, and it would have been called a failure — while TV had already gone
0.152 to 0.113. Agreement is an argmax over near-ties and breaks late; TV is continuous and
moves first. By generation 9 agreement was 114/128 and TV 0.039.

How the comparison is made honest
---------------------------------

After ``PLAY rank → lane L``, the three resulting positions are *exact* relabellings of one
another: the same cards, differing only in which lane the played one went to. So this does
not compare the three policies directly — it finds the permutation σ that carries one
position's observation onto another's, **bit for bit**, and then asks whether the policy
moved the same way. The σ comes from :func:`duel52._engine.lane_permutations`, i.e. from the
Rust encoder (``CLAUDE.md``: one encoder, and a permutation table is a reading of it).

Two σ always qualify, because the two lanes that stayed empty are indistinguishable. The
score takes the kinder of them: the question is whether the policies agree under *some*
relabelling, and holding a network to a choice between two identical lanes would be
measuring nothing.

The forward pass here is PyTorch's, which ``py/tests/test_parity.py`` pins to the engine's
own to 1e-5. Search is deliberately absent: F4.3 established that search does not repair the
bias (``netmcts@1024`` agreed 21/32 against the raw policy's 19/32), and the raw policy is
where the weights speak for themselves.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["LaneReport", "measure"]


@dataclass
class LaneReport:
    """What F4.3's table is made of. Every list has one entry per measured pair."""

    checkpoint: str
    seeds: int
    #: Prior mass on each lane at the opening decision, one row of three per seed.
    opening: list[list[float]]
    #: Spread of the value head across the three post-opening positions.
    value_spread: list[float]
    #: Total-variation distance between two relabelled policies, one per lane pair.
    policy_tv: list[float]
    #: Did the top second action agree across all three lanes, per (seed, rank) pair.
    argmax_agrees: list[bool]
    #: Pairs where no permutation carried one observation exactly onto the other. Any of
    #: these and the measurement below is not measuring what it says it is.
    unmatched: int

    @property
    def pairs(self) -> int:
        return len(self.argmax_agrees)

    def table(self) -> str:
        opening = np.array(self.opening).mean(axis=0) if self.opening else np.zeros(3)
        spread = np.array(self.value_spread) if self.value_spread else np.zeros(1)
        tv = np.array(self.policy_tv) if self.policy_tv else np.zeros(1)
        agreed = sum(self.argmax_agrees)
        rows = [
            f"| {Path(self.checkpoint).name}, {self.pairs} pairs | measured | equivariant |",
            "|---|---:|---:|",
            "| opening prior on lane 1 / 2 / 3 | "
            + " / ".join(f"{v:.3f}" for v in opening)
            + " | .333 each |",
            f"| value-head spread at node 2 (median / max) | "
            f"{np.median(spread):.3f} / {spread.max():.3f} | 0 |",
            f"| policy TV between lane pairs (median / max) | "
            f"{np.median(tv):.3f} / {tv.max():.3f} | 0 |",
            f"| top second action agrees across all three lanes | "
            f"{agreed}/{self.pairs} | {self.pairs}/{self.pairs} |",
        ]
        return "\n".join(rows)


def _model(checkpoint: str | Path):
    from .nn.checkpoint import read_checkpoint
    from .nn.model import Duel52Net, NetConfig

    ckpt = read_checkpoint(checkpoint)
    model = Duel52Net(
        NetConfig(
            obs_dim=ckpt.obs_dim,
            action_dim=ckpt.action_dim,
            width=ckpt.width,
            blocks=ckpt.blocks,
            value_hidden=ckpt.value_hidden,
        )
    )
    model.load_tensors(ckpt.tensors)
    model.eval()
    return model, ckpt


def _permutations(variant: str, encoding_slots: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from ._engine import lane_permutations

    return [
        (
            np.frombuffer(p["obs"], dtype="<u4").astype(np.int64),
            np.frombuffer(p["action"], dtype="<u4").astype(np.int64),
        )
        for p in lane_permutations(variant=variant, encoding_slots=encoding_slots)
    ]


def _policy(model, game, observer: str) -> tuple[np.ndarray, float]:
    """The masked policy and the value head at ``game``, from ``observer``'s seat."""
    import torch

    x = torch.tensor([game.encode_observation(observer)], dtype=torch.float32)
    with torch.no_grad():
        logits, value = model(x)
    logits = logits[0].numpy().astype(np.float64)
    legal = np.array(game.legal_mask(), dtype=bool)
    logits[~legal] = -np.inf
    shifted = np.exp(logits - logits.max())
    return shifted / shifted.sum(), float(value[0])


def measure(
    checkpoint: str | Path,
    *,
    seeds: int = 24,
    first_seed: int = 1,
    variant: str = "split",
    encoding_slots: int = 21,
) -> LaneReport:
    """Play the first card into each lane in turn and compare the three resulting policies."""
    from . import Game

    model, ckpt = _model(checkpoint)
    perms = _permutations(variant, encoding_slots)
    if ckpt.obs_dim != len(perms[0][0]) or ckpt.action_dim != len(perms[0][1]):
        raise ValueError(
            f"{checkpoint} is {ckpt.obs_dim}/{ckpt.action_dim} wide but this variant at "
            f"encoding_slots={encoding_slots} is {len(perms[0][0])}/{len(perms[0][1])} — "
            f"pass the --encoding-slots the checkpoint was trained at"
        )

    report = LaneReport(str(checkpoint), seeds, [], [], [], [], 0)
    for seed in range(first_seed, first_seed + seeds):
        game = Game(variant=variant, seed=seed, encoding_slots=encoding_slots)
        opener = game.to_move
        prior, _ = _policy(model, game, opener)

        plays = [a for a in game.legal_actions() if a["kind"] == "play"]
        lanes = sorted({a["lane"] for a in plays})
        by_lane = np.zeros(len(lanes))
        for a in plays:
            by_lane[a["lane"]] += prior[game.encode_action(a)]
        report.opening.append([float(v) for v in by_lane / max(by_lane.sum(), 1e-12)])

        for rank in sorted({a["rank"] for a in plays}):
            after: list[tuple[np.ndarray, np.ndarray, float]] = []
            for lane in lanes:
                child = game.clone_state()
                child.apply({"kind": "play", "rank": rank, "lane": lane})
                policy, value = _policy(model, child, opener)
                after.append((np.array(child.encode_observation(opener)), policy, value))

            values = [v for _, _, v in after]
            report.value_spread.append(max(values) - min(values))

            agrees = True
            for i in range(len(after)):
                for j in range(i + 1, len(after)):
                    obs_i, policy_i, _ = after[i]
                    obs_j, policy_j, _ = after[j]
                    # Every σ that carries position i onto position j exactly. Two always
                    # do — the two still-empty lanes are indistinguishable — and the score
                    # takes the kinder of them.
                    tvs, matches = [], []
                    for obs_perm, action_perm in perms:
                        moved = np.zeros_like(obs_i)
                        moved[obs_perm] = obs_i
                        if not np.array_equal(moved, obs_j):
                            continue
                        turned = np.zeros_like(policy_i)
                        turned[action_perm] = policy_i
                        tvs.append(0.5 * float(np.abs(turned - policy_j).sum()))
                        matches.append(int(turned.argmax()) == int(policy_j.argmax()))
                    if not tvs:
                        report.unmatched += 1
                        continue
                    report.policy_tv.append(min(tvs))
                    agrees = agrees and any(matches)
            report.argmax_agrees.append(agrees)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m duel52.lanes",
        description="FINDINGS.md F4.3's lane-symmetry metric, for one checkpoint.",
    )
    parser.add_argument("--checkpoint", required=True, help="the .d52nn to measure")
    parser.add_argument("--seeds", type=int, default=24, help="deals to measure (default 24)")
    parser.add_argument("--first-seed", type=int, default=1)
    parser.add_argument("--variant", default="split")
    parser.add_argument(
        "--encoding-slots",
        type=int,
        default=21,
        help="must be what the checkpoint was trained at; it is what fixes obs_dim",
    )
    args = parser.parse_args(argv)

    report = measure(
        args.checkpoint,
        seeds=args.seeds,
        first_seed=args.first_seed,
        variant=args.variant,
        encoding_slots=args.encoding_slots,
    )
    print(report.table())
    if report.unmatched:
        # Not a soft warning: the whole comparison rests on the three positions being exact
        # relabellings of one another, so if they are not, the table above means nothing.
        print(
            f"\n⚠️  {report.unmatched} lane pair(s) were not exact relabellings of each "
            f"other — the encoder and the permutation tables disagree, and the numbers "
            f"above are not measuring lane symmetry. Run `cargo test phase4_lane`.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
