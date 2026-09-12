"""The questions, one function each.

# Adding one

Write a function that takes ``(corpora, ctx)`` and returns a :class:`Section`, and decorate it
with ``@metric``. It runs in registration order and appears in the document in that order.
Everything it needs is a fold over two flat tables — `games.csv` has one row per player-game,
`cards.csv` one row per card that entered play — so a new question costs a function and a
re-render, never a re-run of the games.

# Two conventions every section here follows

**Turns are the player's own turns, 1-based.** The engine counts plies, and P0 owns the even
ones while P1 owns the odd, so a mean taken over raw plies is half a turn later for P1 than
for P0 for no reason anyone cares about. ``own turn k`` is that player's k-th turn — global
ply ``2(k−1) + seat`` — which makes "a card flipped on the turn it was played" exactly 0
turns face-down, as it should be.

**Every interval is clustered on the deal.** See `stats.py`: both games of a colour-paired
deal hold the same cards, and forty card rows come out of one shuffle, so an unclustered
interval on either is too narrow. The point estimates are unaffected.
"""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from . import charts, stats
from .corpus import Corpus
from .stats import EMPTY, Estimate

# ================================================================== documents ==


@dataclass
class Table:
    columns: List[str]
    rows: List[List[str]]
    align: List[str] = field(default_factory=list)
    caption: str = ""
    # Column index -> one key per row, for the HTML's sortable headers. Only needed where the
    # displayed text does not sort into the order the column means: `rank` reads A, 2, … K and
    # would sort 10 < 2 < A < J. A right-aligned column is sorted on the first number in each
    # cell and a left-aligned one on its text, so nothing else here has to say anything.
    sort_keys: Dict[int, List[float]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.align:
            self.align = ["l"] + ["r"] * (len(self.columns) - 1)


@dataclass
class Figure:
    svg: str


@dataclass
class Note:
    text: str


Block = object


@dataclass
class Section:
    key: str
    title: str
    note: str = ""
    blocks: List[Block] = field(default_factory=list)


METRICS: List[Callable[[List[Corpus], "Context"], Optional[Section]]] = []


def metric(fn):
    METRICS.append(fn)
    return fn


@dataclass
class Context:
    dataset: str
    root: Path
    binary: Optional[Path]
    engine_args: List[str]
    card_value_games: int
    prepared: Dict[str, "Prepared"] = field(default_factory=dict)

    def of(self, corpus: Corpus) -> "Prepared":
        if corpus.agent not in self.prepared:
            self.prepared[corpus.agent] = Prepared(corpus)
        return self.prepared[corpus.agent]


# ================================================================== prepared ==


def own_turn(ply: Optional[int]) -> Optional[float]:
    """A ply as its owner's own turn number, 1-based."""
    return None if ply is None else ply // 2 + 1


class Prepared:
    """One corpus's columns, pulled out once and indexed the two ways every metric wants."""

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.ranks = corpus.ranks

        cards = corpus.cards
        self.c_game = cards.col("game")
        self.c_rank = cards.col("rank")
        self.c_base = cards.col("base")
        self.c_enter = cards.col("enter_ply")
        self.c_up = cards.col("faceup_ply")
        self.c_kind = cards.col("faceup_kind")
        self.c_death = cards.col("death_ply")
        self.c_died_up = cards.col("died_face_up")
        self.c_paired = cards.col("paired")

        deal_of = corpus.deal_of_game()
        plies_of = corpus.plies_by_game()
        self.c_deal = [deal_of[g] for g in self.c_game]
        self.c_last_ply = [plies_of[g] - 1 for g in self.c_game]

        self.by_rank: List[List[int]] = [[] for _ in self.ranks]
        for i, r in enumerate(self.c_rank):
            if 0 <= r < len(self.by_rank):
                self.by_rank[r].append(i)

        games = corpus.games
        self.g_game = games.col("game")
        self.g_seat = games.col("seat")
        self.g_deal = games.col("seed")
        self.g_result = games.col("result")
        self.g_score = corpus.scores()
        self.g_unlock = games.col("unlock_ply")
        self.g_hand = games.col("hand_at_unlock")
        self.g_opp_hand = games.col("opp_hand_at_unlock")
        self.g_plies = games.col("plies")
        self.g_pairs = games.col("pairs")
        index = corpus.game_index()
        self.g_opp = [index[(g, 1 - s)] for g, s in zip(self.g_game, self.g_seat)]
        self.first_rows = [i for i, s in enumerate(self.g_seat) if s == 0]

        self.start = [games.col(f"start_{r}") for r in self.ranks]
        self.unlock = [games.col(f"unlock_{r}") for r in self.ranks]
        self.pairs_by_rank = [games.col(f"pairs_{r}") for r in self.ranks]

    # -------------------------------------------------------------- helpers --
    def card_rows(self, rank: Optional[int] = None, *, base: Optional[bool] = None) -> List[int]:
        rows = range(len(self.c_rank)) if rank is None else self.by_rank[rank]
        if base is None:
            return list(rows)
        want = 1 if base else 0
        return [i for i in rows if self.c_base[i] == want]

    def exit_ply(self, i: int) -> int:
        """When a card stopped being face-down, or the last turn of the game if it never
        did. The censored end of the tenure measurement."""
        if self.c_up[i] is not None:
            return self.c_up[i]
        if self.c_death[i] is not None:
            return self.c_death[i]
        return self.c_last_ply[i]


def rank_estimates(
    corpora: Sequence[Corpus],
    ctx: Context,
    compute: Callable[[Prepared, int], Estimate],
) -> Dict[str, List[Estimate]]:
    out: Dict[str, List[Estimate]] = {}
    for corpus in corpora:
        prepared = ctx.of(corpus)
        out[corpus.agent] = [compute(prepared, r) for r in range(len(corpus.ranks))]
    return out


def deck_order(ranks: Sequence[str]) -> Dict[int, List[float]]:
    """Sort keys for a table whose first column is `rank`.

    The labels are the deck's, so the row order already *is* the order the column means and
    the key is just the row's position. Without it the HTML's sortable header would put the
    column in the order the strings happen to fall in — 10, 2, 3, … A, J, K, Q.
    """
    return {0: list(range(len(ranks)))}


def rank_table(
    corpora: Sequence[Corpus],
    estimates: Dict[str, List[Estimate]],
    *,
    places: int = 2,
    plus: bool = False,
    with_n: bool = False,
    caption: str = "",
) -> Table:
    ranks = corpora[0].ranks
    powers = corpora[0].powers
    columns = ["rank", "power"]
    for corpus in corpora:
        columns.append(corpus.label)
        if with_n:
            columns.append("n")
    rows = []
    for r, rank in enumerate(ranks):
        row = [rank, powers[r]]
        for corpus in corpora:
            est = estimates[corpus.agent][r]
            row.append(est.format(places, plus))
            if with_n:
                row.append(f"{est.n:,}" if est.n else "—")
        rows.append(row)
    align = ["l", "l"] + ["r"] * (len(columns) - 2)
    return Table(columns, rows, align=align, caption=caption, sort_keys=deck_order(ranks))


def rank_series(
    corpora: Sequence[Corpus], estimates: Dict[str, List[Estimate]], *, errors: bool = False
) -> List[charts.Series]:
    out = []
    for corpus in corpora:
        values = [e.mean if e else None for e in estimates[corpus.agent]]
        errs = (
            [e.half_width if e and math.isfinite(e.half_width) else None
             for e in estimates[corpus.agent]]
            if errors
            else None
        )
        out.append(charts.Series(corpus.label, values, errs))
    return out


FATE_COLOURS = [f"var(--fate-{i + 1})" for i in range(5)]


# ==================================================================== metrics ==


@metric
def provenance(corpora: List[Corpus], ctx: Context) -> Section:
    rows = []
    for corpus in corpora:
        meta = corpus.meta
        hours = meta["elapsed_secs"] / 3600.0
        # Summed over chunks, so this is throughput for the whole corpus rather than for
        # whichever chunk happened to be written last.
        rate = meta["games"] / meta["elapsed_secs"] if meta["elapsed_secs"] > 0 else float("inf")
        rows.append(
            [
                corpus.label,
                corpus.agent,
                f"{corpus.n_games:,}",
                f"{meta['deals']:,}",
                f"{len(corpus.chunks)}",
                f"{meta['card_rows']:,}",
                f"{rate:.3f}",
                f"{hours:.2f} h",
            ]
        )
    return Section(
        "provenance",
        "Corpora",
        "One agent per column, each playing **itself**. Every corpus in this document was "
        "played under the ruleset named in the header, and the reader refuses to merge two "
        "that were not. A deal is played twice with the seats swapped, so games = 2 × deals.",
        [
            Table(
                ["model", "agent", "games", "deals", "chunks", "card rows", "games/sec",
                 "cpu time"],
                rows,
                align=["l", "l", "r", "r", "r", "r", "r", "r"],
            )
        ],
    )


@metric
def first_player(corpora: List[Corpus], ctx: Context) -> Section:
    rows = []
    values, errors = [], []
    for corpus in corpora:
        p = ctx.of(corpus)
        scores = [p.g_score[i] for i in p.first_rows]
        deals = [p.g_deal[i] for i in p.first_rows]
        est = stats.clustered(scores, deals)
        values.append(est.mean)
        errors.append(est.half_width)
        meta = corpus.meta
        rows.append(
            [
                corpus.label,
                est.format(4),
                f"{meta['wins_p0']:,}",
                f"{meta['wins_p1']:,}",
                f"{meta['draws']:,}",
                f"{100 * meta['draws'] / max(corpus.n_games, 1):.2f}%",
                "yes" if abs(est.mean - 0.5) > est.half_width else "no",
            ]
        )
    figure = charts.intervals(
        [c.label for c in corpora],
        [charts.Series("first-player score", values, errors)],
        places=3,
        reference=0.5,
        reference_label="no advantage",
        caption="First-player score, 1 per win and 0.5 per draw. Whiskers are 95% intervals "
        "clustered on the deal; an interval crossing the line is a result consistent with "
        "no first-player edge.",
    )
    return Section(
        "first-player",
        "First vs second player",
        "The score of whoever moved first, pooled over both halves of every colour-paired "
        "deal. 0.500 is no advantage. The interval is clustered on the deal, which is the "
        "unit that was randomised — both games of a pair hold the same cards.",
        [Table(["model", "first-player score", "P0 wins", "P1 wins", "draws", "draw rate",
                "separated from 0.500"], rows), Figure(figure)],
    )


@metric
def game_shape(corpora: List[Corpus], ctx: Context) -> Section:
    rows = []
    for corpus in corpora:
        p = ctx.of(corpus)
        # `plies` is the whole game's turn count, so it is per game, not per player-game.
        lengths = [p.g_plies[i] for i in p.first_rows]
        unlocks = [own_turn(v) for v in p.g_unlock if v is not None]
        reached = sum(1 for i in p.first_rows if p.g_unlock[i] is not None)
        meta = corpus.meta
        rows.append(
            [
                corpus.label,
                f"{stats.clustered(lengths, [p.g_deal[i] for i in p.first_rows]).mean:.1f}",
                f"{stats.median(lengths):.0f}",
                f"{stats.percentile(lengths, 0.10):.0f} – {stats.percentile(lengths, 0.90):.0f}",
                f"{100 * meta['draws'] / max(corpus.n_games, 1):.2f}%",
                f"{meta['draws_stalemate']:,} / {meta['draws_mutual_lane_win']:,} / "
                f"{meta['draws_ply_limit']:,}",
                f"{100 * reached / max(corpus.n_games, 1):.1f}%",
                f"{stats.plain(unlocks).mean:.1f}" if unlocks else "—",
            ]
        )
    return Section(
        "game-shape",
        "Game shape",
        "Turns here are the game's, not one player's: a game of 42 turns is 21 each. The "
        "unlock is the turn the last draw pile emptied and base cards became attackable "
        "(`game_rules.md` §3) — until then a lane cannot be won, so it divides the game in "
        "two.",
        [
            Table(
                ["model", "mean turns", "median", "p10 – p90", "draw rate",
                 "stalemate / mutual / cap", "reached unlock", "mean unlock turn"],
                rows,
            )
        ],
    )


@metric
def play_turn(corpora: List[Corpus], ctx: Context) -> Section:
    def compute(p: Prepared, rank: int) -> Estimate:
        rows = p.card_rows(rank, base=False)
        return stats.clustered(
            [own_turn(p.c_enter[i]) for i in rows], [p.c_deal[i] for i in rows]
        )

    estimates = rank_estimates(corpora, ctx, compute)
    overall = []
    for corpus in corpora:
        p = ctx.of(corpus)
        rows = p.card_rows(base=False)
        est = stats.clustered(
            [own_turn(p.c_enter[i]) for i in rows], [p.c_deal[i] for i in rows]
        )
        overall.append([corpus.label, est.format(3), f"{est.n:,}"])
    return Section(
        "play-turn",
        "Average card play turn",
        "The owner's own turn on which a card is played from hand, face-down. Base cards are "
        "excluded — they were never played. A low number is a card that goes down early, "
        "which is not the same as a card that goes face-up early.",
        [
            Table(["model", "mean play turn", "cards"], overall),
            rank_table(corpora, estimates, places=2, with_n=True),
            Figure(
                charts.lines(
                    corpora[0].ranks,
                    rank_series(corpora, estimates),
                    places=2,
                    unit=" turns",
                    caption="Mean own-turn a card of each rank is played on.",
                )
            ),
        ],
    )


@metric
def flip_turn(corpora: List[Corpus], ctx: Context) -> Section:
    def voluntary(p: Prepared, rank: int) -> Estimate:
        rows = [i for i in p.card_rows(rank, base=False) if p.c_kind[i] == "chose"]
        return stats.clustered(
            [own_turn(p.c_up[i]) for i in rows], [p.c_deal[i] for i in rows]
        )

    def any_kind(p: Prepared, rank: int) -> Estimate:
        rows = [i for i in p.card_rows(rank, base=False) if p.c_up[i] is not None]
        return stats.clustered(
            [own_turn(p.c_up[i]) for i in rows], [p.c_deal[i] for i in rows]
        )

    def base_cards(p: Prepared, rank: int) -> Estimate:
        rows = [i for i in p.card_rows(rank, base=True) if p.c_up[i] is not None]
        return stats.clustered(
            [own_turn(p.c_up[i]) for i in rows], [p.c_deal[i] for i in rows]
        )

    chosen = rank_estimates(corpora, ctx, voluntary)
    everything = rank_estimates(corpora, ctx, any_kind)
    base = rank_estimates(corpora, ctx, base_cards)
    overall = []
    for corpus in corpora:
        p = ctx.of(corpus)
        rows = [i for i in p.card_rows(base=False) if p.c_kind[i] == "chose"]
        est = stats.clustered(
            [own_turn(p.c_up[i]) for i in rows], [p.c_deal[i] for i in rows]
        )
        allrows = [i for i in p.card_rows(base=False) if p.c_up[i] is not None]
        est_all = stats.clustered(
            [own_turn(p.c_up[i]) for i in allrows], [p.c_deal[i] for i in allrows]
        )
        overall.append([corpus.label, est.format(2), f"{est.n:,}", est_all.format(2)])
    return Section(
        "flip-turn",
        "Average card flip turn",
        "The turn a card goes face-up. The main table counts **only flips its owner chose** "
        "— a card turned up by a 5's cascade or by springing a 3's Trap went face-up without "
        "anyone deciding to, and averaging those in answers a different question. Base cards "
        "are tabled separately: they cannot be flipped before the unlock, so their timing is "
        "a fact about the unlock rather than about the card.",
        [
            Table(["model", "mean flip turn (chosen)", "flips", "any cause"], overall),
            rank_table(corpora, chosen, places=2, with_n=True,
                       caption="Flips the owner chose, played cards only."),
            Figure(
                charts.lines(
                    corpora[0].ranks,
                    rank_series(corpora, chosen),
                    places=2,
                    unit=" turns",
                    caption="Mean own-turn each rank is voluntarily turned face-up.",
                )
            ),
            rank_table(corpora, everything, places=2,
                       caption="Any cause — chosen, cascaded, or sprung."),
            rank_table(corpora, base, places=2, caption="Base cards, any cause."),
        ],
    )


@metric
def face_down_tenure(corpora: List[Corpus], ctx: Context) -> Section:
    def flipped_only(p: Prepared, rank: int) -> Estimate:
        rows = [i for i in p.card_rows(rank, base=False) if p.c_up[i] is not None]
        return stats.clustered(
            [(p.c_up[i] - p.c_enter[i]) / 2.0 for i in rows], [p.c_deal[i] for i in rows]
        )

    def to_exit(p: Prepared, rank: int) -> Estimate:
        rows = p.card_rows(rank, base=False)
        return stats.clustered(
            [(p.exit_ply(i) - p.c_enter[i]) / 2.0 for i in rows], [p.c_deal[i] for i in rows]
        )

    def never(p: Prepared, rank: int) -> Estimate:
        rows = p.card_rows(rank, base=False)
        return stats.clustered(
            [0.0 if p.c_up[i] is not None else 1.0 for i in rows], [p.c_deal[i] for i in rows]
        )

    flipped = rank_estimates(corpora, ctx, flipped_only)
    censored = rank_estimates(corpora, ctx, to_exit)
    unflipped = rank_estimates(corpora, ctx, never)

    overall = []
    for corpus in corpora:
        p = ctx.of(corpus)
        rows = [i for i in p.card_rows(base=False) if p.c_up[i] is not None]
        est = stats.clustered(
            [(p.c_up[i] - p.c_enter[i]) / 2.0 for i in rows], [p.c_deal[i] for i in rows]
        )
        allrows = p.card_rows(base=False)
        exit_est = stats.clustered(
            [(p.exit_ply(i) - p.c_enter[i]) / 2.0 for i in allrows],
            [p.c_deal[i] for i in allrows],
        )
        never_est = stats.clustered(
            [0.0 if p.c_up[i] is not None else 1.0 for i in allrows],
            [p.c_deal[i] for i in allrows],
        )
        overall.append(
            [corpus.label, est.format(2), f"{100 * never_est.mean:.1f}%", exit_est.format(2)]
        )

    return Section(
        "face-down",
        "Turns spent face-down",
        "Measured in the owner's own turns, so a card flipped on the turn it was played is "
        "**0**. Three columns because one number would be a lie by omission: a rank that is "
        "flipped fast *and* killed fast has a short tenure for two different reasons.\n\n"
        "* **among flipped** — cards that were eventually turned face-up. The decision.\n"
        "* **never flipped** — the share that were not, whether killed hidden or still "
        "hidden at the end. This is the censoring, stated rather than dropped.\n"
        "* **to exit** — every played card, counting a hidden death or the end of the game "
        "as the end of its tenure. How long a card actually spends hidden.",
        [
            Table(["model", "mean turns face-down (among flipped)", "never flipped",
                   "mean turns to exit (all cards)"], overall),
            rank_table(corpora, flipped, places=2, with_n=True,
                       caption="Turns face-down before being flipped, by rank."),
            Figure(
                charts.lines(
                    corpora[0].ranks,
                    rank_series(corpora, flipped),
                    places=2,
                    unit=" turns",
                    caption="Turns a card of each rank spends face-down before it is flipped.",
                )
            ),
            rank_table(corpora, unflipped, places=3,
                       caption="Share of played cards of each rank never turned face-up."),
            rank_table(corpora, censored, places=2,
                       caption="Turns face-down counting hidden deaths and the game's end."),
        ],
    )


@metric
def hand_at_unlock(corpora: List[Corpus], ctx: Context) -> Section:
    rows = []
    margin_series = []
    margins = [1, 2, 3, 4]
    margin_labels = ["+1", "+2", "+3", "+4 or more"]
    for corpus in corpora:
        p = ctx.of(corpus)
        held = [v for v in p.g_hand if v is not None]
        deals = [p.g_deal[i] for i, v in enumerate(p.g_hand) if v is not None]
        size = stats.clustered(held, deals)

        # One observation per *game*: the side holding more cards, and what it scored.
        larger, larger_deals, by_margin = [], [], {m: ([], []) for m in margins}
        for i in p.first_rows:
            mine, theirs = p.g_hand[i], p.g_opp_hand[i]
            if mine is None or theirs is None or mine == theirs:
                continue
            bigger = i if mine > theirs else p.g_opp[i]
            gap = abs(mine - theirs)
            larger.append(p.g_score[bigger])
            larger_deals.append(p.g_deal[i])
            bucket = min(gap, margins[-1])
            by_margin[bucket][0].append(p.g_score[bigger])
            by_margin[bucket][1].append(p.g_deal[i])
        est = stats.clustered(larger, larger_deals)
        ties = sum(
            1
            for i in p.first_rows
            if p.g_hand[i] is not None and p.g_hand[i] == p.g_opp_hand[i]
        )
        rows.append(
            [
                corpus.label,
                size.format(2),
                f"{stats.median(held):.0f}",
                est.format(4),
                f"{est.n:,}",
                f"{100 * ties / max(corpus.n_games, 1):.1f}%",
            ]
        )
        margin_series.append(
            charts.Series(
                corpus.label,
                [stats.clustered(*by_margin[m]).mean if by_margin[m][0] else None
                 for m in margins],
                [stats.clustered(*by_margin[m]).half_width if by_margin[m][0] else None
                 for m in margins],
            )
        )

    margin_rows = []
    for mi, label in enumerate(margin_labels):
        row = [label]
        for si, corpus in enumerate(corpora):
            value = margin_series[si].values[mi]
            error = margin_series[si].errors[mi]
            row.append("—" if value is None else f"{value:.4f} ± {error:.4f}")
        margin_rows.append(row)

    return Section(
        "hand-at-unlock",
        "Hand size at the unlock",
        "`FINDINGS.md` H2: every card in hand after the piles empty is a turn the opponent "
        "cannot close a lane. The score column is the **larger-hand side's**, over the games "
        "where the two hands differed — one observation per game, not per player, so a game "
        "cannot vote twice. 0.500 would mean holding more cards is worth nothing.",
        [
            Table(["model", "mean hand at unlock", "median", "score of the larger hand",
                   "games", "tied"], rows),
            Table(["margin"] + [c.label for c in corpora], margin_rows,
                  caption="Score of the side holding this many more cards at the unlock."),
            Figure(
                charts.intervals(
                    margin_labels,
                    margin_series,
                    places=3,
                    reference=0.5,
                    reference_label="no advantage",
                    caption="Score of the larger hand at the unlock, by how much larger.",
                )
            ),
        ],
    )


def _holding_score(
    p: Prepared, counts: List[List[int]], rank: int, exclusive: bool, gate=None
) -> Estimate:
    """Score of the player-games holding at least one of ``rank``.

    ``exclusive`` restricts to the games where the *opponent* held none of it, which is the
    estimator that answers the question. Pooling in the games where both players hold the
    card drags every rank toward 0.500: those observations come in symmetric pairs, one win
    and one loss, and carry no information about the card at all.
    """
    column = counts[rank]
    values, clusters = [], []
    for i in range(len(column)):
        if gate is not None and gate[i] is None:
            continue
        if not column[i]:
            continue
        if exclusive and column[p.g_opp[i]]:
            continue
        values.append(p.g_score[i])
        clusters.append(p.g_deal[i])
    return stats.clustered(values, clusters)


@metric
def starting_hand(corpora: List[Corpus], ctx: Context) -> Section:
    exclusive = rank_estimates(
        corpora, ctx, lambda p, r: _holding_score(p, p.start, r, True)
    )
    inclusive = rank_estimates(
        corpora, ctx, lambda p, r: _holding_score(p, p.start, r, False)
    )
    return Section(
        "starting-hand",
        "Win rate with each card in the opening hand",
        "The opening hand is the one held at the start of that player's **own** first turn — "
        "the deal plus the draw that opens a turn — so both players are measured on the same "
        "number of cards. (`GameState::new` performs P0's opening draw, so 'the hand at "
        "setup' would give P0 six cards and P1 five.)\n\n"
        "**Read the exclusive table.** It is the score of the games where you held the card "
        "and your opponent did not. The inclusive one pools in the games where both held it, "
        "and those contribute a win and a loss in symmetric pairs — they pull every rank "
        "toward 0.500 without saying anything about the card.",
        [
            rank_table(corpora, exclusive, places=4, with_n=True,
                       caption="Score when you hold this rank and the opponent does not."),
            Figure(
                charts.intervals(
                    corpora[0].ranks,
                    rank_series(corpora, exclusive, errors=True),
                    places=3,
                    reference=0.5,
                    reference_label="no advantage",
                    caption="Score of holding this rank at the deal when the opponent does "
                    "not. Whiskers are 95% intervals clustered on the deal — a rank whose "
                    "interval crosses the line has not been shown to be worth anything.",
                )
            ),
            rank_table(corpora, inclusive, places=4,
                       caption="Inclusive: score whenever you hold at least one, whatever the "
                       "opponent holds. Kept for contrast."),
        ],
    )


@metric
def unlock_hand(corpora: List[Corpus], ctx: Context) -> Section:
    exclusive = rank_estimates(
        corpora, ctx, lambda p, r: _holding_score(p, p.unlock, r, True, gate=p.g_hand)
    )

    # Holding any given rank at the unlock is partly just holding *more cards*, which the
    # section above already showed is worth something. Subtracting the mean score at the same
    # hand size takes that out and leaves what the card itself is associated with.
    adjusted: Dict[str, List[Estimate]] = {}
    for corpus in corpora:
        p = ctx.of(corpus)
        baseline: Dict[int, List[float]] = {}
        for i, size in enumerate(p.g_hand):
            if size is None:
                continue
            baseline.setdefault(size, []).append(p.g_score[i])
        means = {size: math.fsum(v) / len(v) for size, v in baseline.items()}
        per_rank = []
        for r in range(len(corpus.ranks)):
            column = p.unlock[r]
            values, clusters = [], []
            for i, size in enumerate(p.g_hand):
                if size is None or not column[i] or column[p.g_opp[i]]:
                    continue
                values.append(p.g_score[i] - means[size])
                clusters.append(p.g_deal[i])
            per_rank.append(stats.clustered(values, clusters))
        adjusted[corpus.agent] = per_rank

    return Section(
        "unlock-hand",
        "Win rate with each card in hand at the unlock",
        "The same two estimators, on the hand held when the last pile emptied. Restricted to "
        "games that reached the unlock.\n\n"
        "The second table is the one to trust. Holding a particular rank at the unlock is "
        "partly just holding *more cards*, and the section above shows that is worth "
        "something on its own; **adjusted** subtracts the mean score at the same hand size, "
        "leaving what is associated with the card rather than with the size of the hand it "
        "sits in. It is a difference from 0, not a win rate: `+0.02` is two points of win "
        "probability above an average hand of that size.",
        [
            rank_table(corpora, exclusive, places=4, with_n=True,
                       caption="Score when you hold this rank at the unlock and the opponent "
                       "does not."),
            rank_table(corpora, adjusted, places=4, plus=True,
                       caption="The same, minus the mean score at that hand size."),
            Figure(
                charts.intervals(
                    corpora[0].ranks,
                    rank_series(corpora, adjusted, errors=True),
                    places=3,
                    reference=0.0,
                    reference_label="an average hand of that size",
                    caption="Hand-size-adjusted value of holding each rank at the unlock, in "
                    "win-probability points.",
                )
            ),
        ],
    )


@metric
def pairs(corpora: List[Corpus], ctx: Context) -> Section:
    def per_game(p: Prepared, rank: int) -> Estimate:
        column = p.pairs_by_rank[rank]
        return stats.clustered([float(v) for v in column], p.g_deal)

    def paired_share(p: Prepared, rank: int) -> Estimate:
        rows = p.card_rows(rank)
        return stats.clustered(
            [float(p.c_paired[i]) for i in rows], [p.c_deal[i] for i in rows]
        )

    declared = rank_estimates(corpora, ctx, per_game)
    share = rank_estimates(corpora, ctx, paired_share)

    rows = []
    for corpus in corpora:
        p = ctx.of(corpus)
        total = stats.clustered([float(v) for v in p.g_pairs], p.g_deal)
        with_any = stats.clustered(
            [1.0 if v else 0.0 for v in p.g_pairs], p.g_deal
        )
        ever = stats.clustered(
            [float(v) for v in p.c_paired], p.c_deal
        )
        rows.append(
            [
                corpus.label,
                total.format(3),
                f"{2 * total.mean:.2f}",
                f"{100 * with_any.mean:.1f}%",
                f"{100 * ever.mean:.1f}%",
            ]
        )

    return Section(
        "pairs",
        "Pairs",
        "A pair is two face-up same-rank cards on one side of one lane, declared with an "
        "action (§5). Rates are **per player-game**, so 'pairs per game' is what one player "
        "declares in one game; the game as a whole sees twice that.",
        [
            Table(["model", "pairs declared per player-game", "per game (both sides)",
                   "player-games with a pair", "cards that were ever paired"], rows),
            rank_table(corpora, share, places=3, with_n=True,
                       caption="Share of the cards of each rank that entered play and were "
                       "ever a member of a declared pair. The rate that is comparable across "
                       "ranks."),
            Figure(
                charts.grouped_bars(
                    corpora[0].ranks,
                    rank_series(corpora, share, errors=True),
                    places=3,
                    caption="Share of each rank's cards that were ever paired.",
                )
            ),
            rank_table(corpora, declared, places=3,
                       caption="Pairs of each rank declared per player-game. Depends on how "
                       "often the rank is drawn as well as on how pairable it is."),
        ],
    )


@metric
def deaths(corpora: List[Corpus], ctx: Context) -> Section:
    def face_up_share(p: Prepared, rank: int) -> Estimate:
        rows = [i for i in p.card_rows(rank) if p.c_death[i] is not None]
        return stats.clustered(
            [float(p.c_died_up[i]) for i in rows], [p.c_deal[i] for i in rows]
        )

    def death_rate(p: Prepared, rank: int) -> Estimate:
        rows = p.card_rows(rank)
        return stats.clustered(
            [1.0 if p.c_death[i] is not None else 0.0 for i in rows],
            [p.c_deal[i] for i in rows],
        )

    up_share = rank_estimates(corpora, ctx, face_up_share)
    killed = rank_estimates(corpora, ctx, death_rate)

    rows = []
    for corpus in corpora:
        p = ctx.of(corpus)
        dead = [i for i in range(len(p.c_rank)) if p.c_death[i] is not None]
        share_up = stats.clustered(
            [float(p.c_died_up[i]) for i in dead], [p.c_deal[i] for i in dead]
        )
        entered = len(p.c_rank)
        rows.append(
            [
                corpus.label,
                f"{len(dead) / max(corpus.n_games, 1):.2f}",
                f"{100 * len(dead) / max(entered, 1):.1f}%",
                share_up.format(4),
                f"{1 - share_up.mean:.4f}",
            ]
        )

    # Which ranks have a death trigger is a property of the *ruleset*, so it is read off the
    # corpus rather than asserted: a rank sprang a trap if some card of it ever went face-up
    # that way. A ruleset that ablates the 3 then says so instead of repeating the canonical
    # claim, which under that ruleset is false.
    sprang = sorted(
        {
            corpora[0].ranks[ctx.of(corpus).c_rank[i]]
            for corpus in corpora
            for i in range(len(ctx.of(corpus).c_rank))
            if ctx.of(corpus).c_kind[i] == "trap"
        },
        key=corpora[0].ranks.index,
    )
    if sprang:
        which = ", ".join(f"**{r}**" for r in sprang)
        trigger_note = (
            f"\n\nOne group of ranks cannot die face-down at all: a face-down card with a "
            f"death trigger springs face-up instead of dying (§6), so it is either killed "
            f"face-up later or not killed at all. In these corpora that is {which}, which is "
            f"why they read 1.000 below."
        )
    else:
        trigger_note = (
            "\n\nNo rank in this ruleset has a death trigger, so nothing springs face-up "
            "when it is killed and every rank can die face-down."
        )
    return Section(
        "deaths",
        "How cards die: face-up or face-down",
        "Every card that entered play and was killed, split by which side it was showing "
        "when it died. A face-down card is a blank 2-HP card whatever its rank (§5), so "
        "dying face-down means its power never did anything — the flip that would have paid "
        "for it never happened." + trigger_note,
        [
            Table(["model", "deaths per game", "share of cards that die",
                   "died face-up", "died face-down"], rows),
            rank_table(corpora, up_share, places=3, with_n=True,
                       caption="Of the cards of this rank that died, the share that were "
                       "face-up at the time."),
            Figure(
                charts.grouped_bars(
                    corpora[0].ranks,
                    rank_series(corpora, up_share, errors=True),
                    places=3,
                    baseline=0.5,
                    baseline_label="even split",
                    caption="Share of each rank's deaths that happened face-up.",
                )
            ),
            rank_table(corpora, killed, places=3,
                       caption="Share of the cards of each rank that entered play and were "
                       "killed at all."),
        ],
    )


FATES = [
    ("flipped by choice", lambda p, i: p.c_kind[i] == "chose"),
    ("flipped by a cascade", lambda p, i: p.c_kind[i] == "cascade"),
    ("sprang its trap", lambda p, i: p.c_kind[i] == "trap"),
    ("killed face-down", lambda p, i: p.c_up[i] is None and p.c_death[i] is not None),
    ("face-down at the end", lambda p, i: p.c_up[i] is None and p.c_death[i] is None),
]


@metric
def fates(corpora: List[Corpus], ctx: Context) -> Section:
    overall_rows = []
    figures = []
    for corpus in corpora:
        p = ctx.of(corpus)
        rows = p.card_rows(base=False)
        n = max(len(rows), 1)
        shares = [
            sum(1 for i in rows if test(p, i)) / n for _, test in FATES
        ]
        overall_rows.append(
            [corpus.label] + [f"{s:.3f}" for s in shares]
            + [f"{shares[0] + shares[1] + shares[2]:.3f}"]
        )
        layers = []
        for (name, test), _ in zip(FATES, range(len(FATES))):
            values = []
            for r in range(len(corpus.ranks)):
                rank_rows = p.card_rows(r, base=False)
                values.append(
                    sum(1 for i in rank_rows if test(p, i)) / len(rank_rows)
                    if rank_rows
                    else None
                )
            layers.append(charts.Series(name, values))
        figures.append(
            Figure(
                charts.stacked_bars(
                    corpus.ranks,
                    layers,
                    palette=FATE_COLOURS,
                    caption=f"{corpus.label}: what became of every card of each rank played "
                    "from hand.",
                )
            )
        )

    def flipped_ever(p: Prepared, rank: int) -> Estimate:
        rows = p.card_rows(rank, base=False)
        return stats.clustered(
            [1.0 if p.c_up[i] is not None else 0.0 for i in rows],
            [p.c_deal[i] for i in rows],
        )

    def killed_hidden(p: Prepared, rank: int) -> Estimate:
        rows = p.card_rows(rank, base=False)
        return stats.clustered(
            [1.0 if (p.c_up[i] is None and p.c_death[i] is not None) else 0.0 for i in rows],
            [p.c_deal[i] for i in rows],
        )

    return Section(
        "fates",
        "What becomes of a face-down card",
        "Every card is played face-down, so this is the whole population: of the cards you "
        "put on the board, how many ever come up, how many are killed before they do, and "
        "how many are still hidden when the game ends. The five outcomes partition the "
        "cards played from hand — base cards are excluded, since nobody chose to play them.",
        [
            Table(
                ["model"] + [name for name, _ in FATES] + ["ever face-up"],
                overall_rows,
            ),
            rank_table(corpora, rank_estimates(corpora, ctx, flipped_ever), places=3,
                       with_n=True,
                       caption="Share of each rank played from hand that was ever face-up, "
                       "by any cause."),
            rank_table(corpora, rank_estimates(corpora, ctx, killed_hidden), places=3,
                       caption="Share killed while still face-down — the power never fired."),
        ]
        + figures,
    )


# ============================================================== card value ==


def _card_value_table(corpus: Corpus, ctx: Context) -> Optional[List[List[str]]]:
    """Run `duel52 card-value` for one checkpoint and parse its Markdown table.

    Returns ``None`` when the agent has no checkpoint, the binary is missing, the tool
    refuses the ruleset, or the control fails — the last of which is the whole reason the
    tool prints a control at all.
    """
    checkpoint = None
    name = corpus.agent
    if ":" in name:
        rest = name.split(":", 1)[1]
        checkpoint = rest.rsplit("@", 1)[0] if "@" in rest else rest
    if not checkpoint or ctx.binary is None or ctx.card_value_games <= 0:
        return None
    command = [
        str(ctx.binary),
        "card-value",
        "--checkpoint",
        checkpoint,
        "--games",
        str(ctx.card_value_games),
        "--markdown",
        "--seed",
        "1",
        *ctx.engine_args,
    ]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    text = done.stdout
    if "CONTROL FAILED" in text or "POSITION(S) SURVIVED" in text:
        return None
    rows = []
    for line in text.splitlines():
        if not line.startswith("| ") or line.startswith("|---") or "| rank |" in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # rank | power | kind | in hand | ± | on board | gap
        if len(cells) == 7:
            rows.append(cells)
    return rows or None


@metric
def card_value(corpora: List[Corpus], ctx: Context) -> Optional[Section]:
    blocks: List[Block] = []
    ranks = corpora[0].ranks

    # ---- the counterfactual table, which needs a value head ----
    counterfactual: Dict[str, Dict[str, str]] = {}
    for corpus in corpora:
        rows = _card_value_table(corpus, ctx)
        if rows:
            # `in hand` with its paired standard error, which is the error on the comparison
            # between cards rather than on any card's absolute win rate.
            counterfactual[corpus.agent] = {r[0]: f"{r[3]} ± {r[4]}" for r in rows}
    if counterfactual:
        columns = ["rank", "power"] + [
            c.label for c in corpora if c.agent in counterfactual
        ]
        table_rows = []
        for i, rank in enumerate(ranks):
            row = [rank, corpora[0].powers[i]]
            for corpus in corpora:
                if corpus.agent in counterfactual:
                    row.append(counterfactual[corpus.agent].get(rank, "—"))
            table_rows.append(row)
        # Same unit as the fitted tables below: points of win probability.
        blocks.append(
            Table(
                columns,
                table_rows,
                align=["l", "l"] + ["r"] * (len(columns) - 2),
                sort_keys=deck_order(ranks),
                caption="`duel52 card-value`: holding the position fixed, what is this card "
                "worth in hand rather than an average card, in win-probability points? Each "
                "column is that checkpoint's own value head.",
            )
        )
    else:
        blocks.append(
            Note(
                "The counterfactual table is not available for these corpora — no agent "
                "carries a checkpoint, the engine binary was not found, or `card-value` "
                "refused the ruleset (it needs a rank to be plausibly hidden, which "
                "`mirrored` almost never allows). It is the only measurement here that "
                "cannot come from played games."
            )
        )

    # ---- the corpus-derived tables, which need nothing but the games ----
    for counts_of, gate, key, caption in (
        (
            lambda p: p.start,
            None,
            "dealt",
            "**Dealt.** A logistic fit of the result on how many more of each rank you were "
            "dealt than your opponent, relative to an average card, in win-probability "
            "points. The opening hand is dealt at random, so this is a randomised "
            "comparison rather than a correlation — it is the closest thing here to an "
            "experiment.",
        ),
        (
            lambda p: p.unlock,
            "hand",
            "unlock",
            "**Held at the unlock.** The same fit on the hand held when the piles emptied. "
            "Descriptive rather than randomised: you chose what to still be holding, so a "
            "card that gets kept in positions that are already won looks good for that "
            "reason.",
        ),
    ):
        fits = {}
        for corpus in corpora:
            p = ctx.of(corpus)
            counts = counts_of(p)
            rows, targets, clusters = [], [], []
            for i in range(len(p.g_score)):
                if gate == "hand" and p.g_hand[i] is None:
                    continue
                opponent = p.g_opp[i]
                rows.append(
                    [float(counts[r][i] - counts[r][opponent]) for r in range(len(ranks))]
                )
                targets.append(p.g_score[i])
                clusters.append(p.g_deal[i])
            fit = stats.logistic_fit(rows, targets, clusters) if rows else None
            if fit:
                fits[corpus.agent] = fit
        if not fits:
            continue

        # The contrast that makes this comparable to the counterfactual table: rank r
        # against the *average* rank, which is what "worth more than an average card" means.
        # Summing the raw coefficients would instead measure holding one more card of any
        # kind, which the hand-size section already answers.
        k = len(ranks)
        contrasts: Dict[str, List[tuple]] = {}
        for agent, fit in fits.items():
            per_rank = []
            for r in range(k):
                weights = [0.0] * (k + 1)
                for j in range(k):
                    weights[j + 1] = -1.0 / k
                weights[r + 1] += 1.0
                value, half = fit.contrast(weights)
                # A logit coefficient moves the win probability by β/4 at p = 0.5, and the
                # ×100 puts it in **points** — the unit `duel52 card-value` prints, so the
                # two tables can be read against each other without rescaling one by eye.
                per_rank.append((value / 4.0 * 100.0, half / 4.0 * 100.0))
            contrasts[agent] = per_rank

        columns = ["rank", "power"] + [c.label for c in corpora if c.agent in fits]
        table_rows = []
        for r, rank in enumerate(ranks):
            row = [rank, corpora[0].powers[r]]
            for corpus in corpora:
                if corpus.agent in fits:
                    value, half = contrasts[corpus.agent][r]
                    row.append(f"{value:+.2f} ± {half:.2f}")
            table_rows.append(row)
        blocks.append(
            Table(
                columns,
                table_rows,
                align=["l", "l"] + ["r"] * (len(columns) - 2),
                sort_keys=deck_order(ranks),
                caption=caption,
            )
        )
        blocks.append(
            Figure(
                charts.intervals(
                    ranks,
                    [
                        charts.Series(
                            corpus.label,
                            [contrasts[corpus.agent][r][0] for r in range(k)],
                            [contrasts[corpus.agent][r][1] for r in range(k)],
                        )
                        for corpus in corpora
                        if corpus.agent in fits
                    ],
                    places=2,
                    reference=0.0,
                    reference_label="an average card",
                    unit=" points",
                    caption=f"Win-probability points for holding this rank rather than an "
                    f"average card, {key}. Intervals are 95% and clustered on the deal.",
                )
            )
        )
    if not any(isinstance(b, Table) for b in blocks[1:]):
        blocks.append(
            Note(
                "The corpus-derived tables need numpy (`.venv/bin/pip install numpy`)."
            )
        )

    return Section(
        "card-value",
        "What a card is worth",
        "Two answers to the same question, by different routes, and they are worth reading "
        "against each other.\n\n"
        "**Counterfactual** holds a real position fixed and swaps the card in hand, asking "
        "the value head what changed. It is exact about the position and only as good as "
        "that head. It is the one measurement in this document that cannot come from played "
        "games.\n\n"
        "**Corpus-derived** fits the result on how many more of each rank you held than your "
        "opponent, so each rank is measured with the rest of the hand held fixed. It needs "
        "no network, so it works for any agent — including `random`. Both tables are "
        "**relative to an average card**, which is the counterfactual's convention, so the "
        "two are on the same scale.\n\n"
        "The *dealt* table is the one with an identification argument behind it: the opening "
        "hand is dealt at random, so how many of a rank you were dealt is randomly assigned "
        "and its coefficient is a causal effect rather than a correlation. The *unlock* "
        "table is what you were still holding, which you chose, and it is descriptive.",
        blocks,
    )
