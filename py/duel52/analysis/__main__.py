"""The one command: play what is missing, then render the document.

    # play 2,000 games for each of two agents, in chunks, and render
    .venv/bin/python -m duel52.analysis \\
        --agents netmcts:models/duel52-split-gen031.d52nn@1000,\\
netmcts:models/duel52-split-lane-gen032.d52nn@1000 \\
        --games 2000 --chunk 250 --encoding-slots 21 --eval-batch 32

    # render whatever corpora are already on disk, playing nothing
    .venv/bin/python -m duel52.analysis

Extraction is **chunked and resumable**: each chunk is its own directory named for the seed
range it covers, an existing chunk is skipped rather than replayed, and the reader merges
them. So an interrupted run costs the current chunk and nothing else, and adding a third
model later plays only that model's games.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from . import corpus as corpus_mod
from . import report
from .metrics import Context

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BINARY = REPO_ROOT / "target" / "release" / "duel52"


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m duel52.analysis",
        description="Play the self-play corpora and render the analysis document.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dir", default="analysis", type=Path,
        help="where the corpora and the document live (default: analysis)",
    )
    parser.add_argument(
        "--agents", default=None,
        help="comma-separated agents to play, each against itself. Omit to render only.",
    )
    parser.add_argument("--games", type=int, default=2000, help="games per agent")
    parser.add_argument(
        "--chunk", type=int, default=0,
        help="play in chunks of this many games, so an interruption costs one chunk "
        "(default: one chunk of --games)",
    )
    parser.add_argument("--seed", type=int, default=1, help="first deal seed")
    parser.add_argument("--threads", type=int, default=0, help="0 means all cores")
    parser.add_argument(
        "--eval-batch", type=int, default=32,
        help="games in flight per worker. Unlike a gate, self-play holds one checkpoint, so "
        "both seats wait on the same network and the batch does not split (default 32)",
    )
    parser.add_argument("--variant", default=None, help="base | split | mirrored")
    parser.add_argument("--config", default=None, help="a config file, for a modded ruleset")
    parser.add_argument("--encoding-slots", type=int, default=None)
    parser.add_argument(
        "--dataset", default=None,
        help="which dataset to work on: extraction writes into it, and rendering is limited "
        "to it. Defaults to the variant's own name, and to rendering every dataset found. "
        "Name it when a corpus differs in something the rules hash does not capture — the "
        "search budget, most usefully — so it gets its own document instead of being merged "
        "into the variant's",
    )
    parser.add_argument(
        "--card-value-games", type=int, default=400,
        help="positions for `duel52 card-value`, the one measurement that is not a fold over "
        "the corpus (default 400; 0 turns it off)",
    )
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument(
        "--force", action="store_true", help="replay chunks that already exist"
    )
    parser.add_argument(
        "--render-only", action="store_true",
        help="render without playing anything, even though --agents was given. `--agents` "
        "also fixes the column order, so this is how you re-render an existing document in "
        "the order you named the models rather than alphabetically",
    )
    return parser.parse_args(argv)


def engine_args(args: argparse.Namespace) -> List[str]:
    """The flags that select the ruleset — forwarded to every engine call so the corpus and
    the card-value table describe the same game."""
    out: List[str] = []
    if args.config:
        out += ["--config", args.config]
    if args.variant:
        out += ["--variant", args.variant]
    if args.encoding_slots:
        out += ["--encoding-slots", str(args.encoding_slots)]
    return out


def extract(args: argparse.Namespace) -> int:
    if not args.binary.exists():
        print(
            f"error: no engine at {args.binary} — run `cargo build --release`",
            file=sys.stderr,
        )
        return 1
    total = args.games + (args.games % 2)
    chunk = args.chunk or total
    chunk += chunk % 2
    starts = list(range(0, total, chunk))
    print(
        f"Playing {total:,} games per agent in {len(starts)} chunk(s) of up to {chunk:,}.",
        flush=True,
    )
    began = time.time()
    for index, offset in enumerate(starts):
        size = min(chunk, total - offset)
        # A deal is two games, so a chunk of `size` games advances the seed by size / 2.
        seed = args.seed + offset // 2
        command = [
            str(args.binary), "analyze",
            "--agents", args.agents,
            "--games", str(size),
            "--seed", str(seed),
            "--eval-batch", str(args.eval_batch),
            "--out", str(args.dir),
            *engine_args(args),
        ]
        if args.threads:
            command += ["--threads", str(args.threads)]
        if args.dataset:
            command += ["--dataset", args.dataset]
        if args.force:
            command.append("--force")
        print(f"\n── chunk {index + 1}/{len(starts)}: {size:,} games from seed {seed}",
              flush=True)
        done = subprocess.run(command)
        if done.returncode != 0:
            print(
                f"error: the engine exited {done.returncode} on chunk {index + 1}. "
                f"Chunks already written are kept — re-run to continue.",
                file=sys.stderr,
            )
            return done.returncode
    print(f"\nAll chunks done in {(time.time() - began) / 3600:.2f} h.", flush=True)
    return 0


def render(args: argparse.Namespace) -> int:
    root: Path = args.dir
    found = corpus_mod.datasets(root)
    if not found:
        print(f"error: no corpora under {root}. Pass --agents to play some.", file=sys.stderr)
        return 1
    names = [p.name for p in found]
    if args.dataset:
        found = [p for p in found if p.name == args.dataset]
        if not found:
            print(
                f"error: no dataset {args.dataset!r} under {root}; found {', '.join(names)}",
                file=sys.stderr,
            )
            return 1
    status = 0
    for dataset_dir in found:
        try:
            corpora = corpus_mod.load_dataset(dataset_dir)
        except ValueError as error:
            print(f"error: {dataset_dir.name}: {error}", file=sys.stderr)
            status = 1
            continue
        if not corpora:
            continue
        # Columns follow the order the agents were named in, which for a lineage is the
        # order worth reading them in. Without `--agents` there is nothing to go on and the
        # directory order (alphabetical) stands.
        if args.agents:
            wanted = [a.strip() for a in args.agents.split(",") if a.strip()]
            rank = {name: i for i, name in enumerate(wanted)}
            corpora.sort(key=lambda c: (rank.get(c.agent, len(rank)), c.agent))
        ctx = Context(
            dataset=dataset_dir.name,
            root=root,
            binary=args.binary if args.binary.exists() and args.card_value_games else None,
            engine_args=engine_args(args),
            card_value_games=args.card_value_games,
        )
        written = report.write(corpora, ctx, root)
        for path in written:
            print(f"wrote {path}")
    return status


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.agents and not args.render_only:
        code = extract(args)
        if code != 0:
            return code
    return render(args)


if __name__ == "__main__":
    raise SystemExit(main())
