"""Reading the corpora `duel52 analyze` writes.

One :class:`Corpus` is one agent's self-play, possibly assembled from several chunks. A
chunk is a directory ``<dataset>/<agent>/s<seed>-g<games>/`` holding ``games.csv``,
``cards.csv`` and ``meta.json``; chunks are merged by the agent name in their metadata, so
adding games later and resuming after a crash are the same operation.

Two things are checked rather than assumed, because both produce a plausible-looking table
rather than an error:

* **Chunks must not overlap in seeds.** Two chunks over the same deals would double-count
  those games and quietly narrow every confidence interval.
* **Every corpus in one document must share a ``rules_hash``.** ``MODULAR_RULES.md`` §6: the
  encoder is rank-agnostic, so nothing about a corpus's *shape* says which ruleset produced
  it. A cross-ruleset comparison is refused rather than printed.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = 1

#: Columns read as integers. Everything else is a string, and an empty field is ``None``.
_INT_COLUMNS = {
    "game",
    "seed",
    "seat",
    "plies",
    "decisions",
    "unlock_ply",
    "hand_at_unlock",
    "opp_hand_at_unlock",
    "hand_at_end",
    "draws_taken",
    "stuck_turns",
    "plays",
    "attacks",
    "pairs",
    "owner",
    "rank",
    "base",
    "enter_ply",
    "faceup_ply",
    "death_ply",
    "died_face_up",
    "paired",
}
_FLOAT_COLUMNS = {"lane_conc", "attack_conc"}


class Table:
    """A CSV as columns, with ``None`` for a field that was written empty.

    Columnar because the card table runs to hundreds of thousands of rows and every metric
    wants two or three of its ten columns.
    """

    def __init__(self, columns: Dict[str, list]):
        self.columns = columns
        self.n = len(next(iter(columns.values()))) if columns else 0

    def __len__(self) -> int:
        return self.n

    def col(self, name: str) -> list:
        try:
            return self.columns[name]
        except KeyError:
            raise KeyError(
                f"no column {name!r}; the corpus has {sorted(self.columns)}"
            ) from None

    def has(self, name: str) -> bool:
        return name in self.columns

    def extend(self, other: "Table") -> None:
        if set(self.columns) != set(other.columns):
            missing = set(self.columns) ^ set(other.columns)
            raise ValueError(f"chunks disagree about columns: {sorted(missing)}")
        for name, values in other.columns.items():
            self.columns[name].extend(values)
        self.n += other.n


def _read_csv(path: Path, prefixed_ints: Sequence[str] = ()) -> Table:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"{path} is empty") from None
        kinds = []
        for name in header:
            if name in _FLOAT_COLUMNS:
                kinds.append(float)
            elif name in _INT_COLUMNS or any(
                name.startswith(p) for p in prefixed_ints
            ):
                kinds.append(int)
            else:
                kinds.append(str)
        columns: Dict[str, list] = {name: [] for name in header}
        for row in reader:
            if len(row) != len(header):
                raise ValueError(
                    f"{path}: a row has {len(row)} fields, the header has {len(header)}"
                )
            for name, kind, value in zip(header, kinds, row):
                columns[name].append(kind(value) if value != "" else None)
    return Table(columns)


@dataclass
class Chunk:
    directory: Path
    meta: dict

    @property
    def seeds(self) -> range:
        return range(self.meta["first_seed"], self.meta["first_seed"] + self.meta["deals"])


@dataclass
class Corpus:
    """One agent's self-play corpus: the games, the cards, and where they came from."""

    agent: str
    meta: dict
    games: Table
    cards: Table
    chunks: List[Chunk] = field(default_factory=list)

    # ----------------------------------------------------------------- naming --
    @property
    def label(self) -> str:
        """A short column heading: the checkpoint's name and its search budget."""
        name = self.agent
        if ":" not in name:
            return name
        kind, rest = name.split(":", 1)
        budget = ""
        if "@" in rest:
            rest, budget = rest.rsplit("@", 1)
            budget = f"@{budget}"
        stem = Path(rest).stem
        for prefix in ("duel52-split-", "duel52-"):
            if stem.startswith(prefix):
                stem = stem[len(prefix) :]
                break
        return f"{stem}{budget}"

    @property
    def ranks(self) -> List[str]:
        return list(self.meta["ranks"])

    @property
    def powers(self) -> List[str]:
        return list(self.meta["powers"])

    @property
    def n_games(self) -> int:
        return len(self.games) // 2

    # ------------------------------------------------------------------ joins --
    def game_index(self) -> Dict[Tuple[int, int], int]:
        """``(game, seat) -> row`` in the games table."""
        if self._index is None:
            game, seat = self.games.col("game"), self.games.col("seat")
            self._index = {(g, s): i for i, (g, s) in enumerate(zip(game, seat))}
        return self._index

    _index: Optional[Dict[Tuple[int, int], int]] = None

    def plies_by_game(self) -> Dict[int, int]:
        """``game -> total turns``, for the cards that never left the board."""
        if self._plies is None:
            self._plies = dict(zip(self.games.col("game"), self.games.col("plies")))
        return self._plies

    _plies: Optional[Dict[int, int]] = None

    def deal_of_game(self) -> Dict[int, int]:
        """``game -> seed``. The seed is the *deal*, and both games of a pair share it —
        which is what every clustered confidence interval in `metrics` is clustered on."""
        if self._deal is None:
            self._deal = dict(zip(self.games.col("game"), self.games.col("seed")))
        return self._deal

    _deal: Optional[Dict[int, int]] = None

    def scores(self) -> List[float]:
        """One score per row of the games table: 1 win, 0.5 draw, 0 loss."""
        return [
            1.0 if r == "win" else 0.0 if r == "loss" else 0.5
            for r in self.games.col("result")
        ]


def load_chunk(directory: Path) -> Corpus:
    meta = json.loads((directory / "meta.json").read_text())
    if meta.get("schema") != SCHEMA:
        raise ValueError(
            f"{directory} is corpus schema {meta.get('schema')}, this reader is {SCHEMA}. "
            "Re-run `duel52 analyze` for it."
        )
    games = _read_csv(directory / "games.csv", prefixed_ints=("start_", "unlock_", "pairs_"))
    cards = _read_csv(directory / "cards.csv")
    return Corpus(
        agent=meta["agent"],
        meta=meta,
        games=games,
        cards=cards,
        chunks=[Chunk(directory, meta)],
    )


def _merge(into: Corpus, other: Corpus) -> None:
    for key in ("rules_hash", "variant", "ranks", "config_summary"):
        if into.meta[key] != other.meta[key]:
            raise ValueError(
                f"{other.chunks[0].directory} disagrees with "
                f"{into.chunks[0].directory} about {key}"
            )
    covered = set()
    for chunk in into.chunks:
        covered.update(chunk.seeds)
    overlap = covered.intersection(other.chunks[0].seeds)
    if overlap:
        raise ValueError(
            f"{other.chunks[0].directory} re-plays {len(overlap)} deal(s) already covered by "
            f"another chunk of {into.agent}. Overlapping chunks would count those games "
            "twice; move --seed on, or delete one."
        )
    # Game indices are per chunk and would collide, so the later chunk's are shifted past
    # the earlier one's. The number itself means nothing downstream — it is a join key.
    shift = max(into.games.col("game")) + 1
    for table in (other.games, other.cards):
        column = table.col("game")
        for i, value in enumerate(column):
            column[i] = value + shift
    into.games.extend(other.games)
    into.cards.extend(other.cards)
    into.chunks.extend(other.chunks)
    into.meta = dict(into.meta)
    into.meta["games"] += other.meta["games"]
    into.meta["deals"] += other.meta["deals"]
    into.meta["elapsed_secs"] += other.meta["elapsed_secs"]
    for key in (
        "wins_p0",
        "wins_p1",
        "draws",
        "draws_stalemate",
        "draws_mutual_lane_win",
        "draws_ply_limit",
        "card_rows",
    ):
        into.meta[key] += other.meta[key]
    into._index = into._plies = into._deal = None


def find_chunks(dataset_dir: Path) -> List[Path]:
    return sorted(p.parent for p in dataset_dir.glob("*/*/meta.json"))


def load_dataset(dataset_dir: Path) -> List[Corpus]:
    """Every agent's corpus under one dataset directory, chunks merged, agents in the order
    their first chunk sorts."""
    by_agent: Dict[str, Corpus] = {}
    order: List[str] = []
    for directory in find_chunks(dataset_dir):
        corpus = load_chunk(directory)
        if corpus.agent in by_agent:
            _merge(by_agent[corpus.agent], corpus)
        else:
            by_agent[corpus.agent] = corpus
            order.append(corpus.agent)
    corpora = [by_agent[a] for a in order]
    hashes = {c.meta["rules_hash"] for c in corpora}
    if len(hashes) > 1:
        raise ValueError(
            "these corpora were played under different rulesets "
            f"({', '.join(sorted(hashes))}). A number compared across them would not mean "
            "anything, so it is refused rather than printed."
        )
    return corpora


def datasets(root: Path) -> List[Path]:
    """Every dataset directory under an analysis root — one per variant/ruleset."""
    return sorted({p.parent.parent.parent for p in root.glob("*/*/*/meta.json")})


def iter_rank_columns(table: Table, prefix: str, ranks: Iterable[str]) -> List[list]:
    return [table.col(f"{prefix}_{rank}") for rank in ranks]
