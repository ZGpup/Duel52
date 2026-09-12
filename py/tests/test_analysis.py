"""The analysis reader, the estimators, and the document.

These are hermetic: the corpora here are written by hand, so the tests say what the reader
and the statistics do rather than what the engine happened to produce. One test at the end
runs the real binary if it has been built, which is the only place the two sides meet.

The estimator tests matter most. A wrong mean is usually visible; a **wrong interval** is
not, and every table in the document carries one.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duel52.analysis import charts, corpus as corpus_mod, report, stats  # noqa: E402
from duel52.analysis.metrics import Context, Prepared, own_turn  # noqa: E402

RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
POWERS = [
    "Action", "View", "Trap", "Foresight", "Flip", "Freeze", "Heal All",
    "Retaliate", "Nimble", "Twinstrike", "Taunt", "Move", "Empower",
]

GAMES_HEADER = (
    "game,seed,seat,result,draw_reason,plies,decisions,unlock_ply,hand_at_unlock,"
    "opp_hand_at_unlock,hand_at_end,draws_taken,stuck_turns,plays,attacks,pairs,"
    "lane_conc,attack_conc"
    + "".join(f",start_{r}" for r in RANKS)
    + "".join(f",unlock_{r}" for r in RANKS)
    + "".join(f",pairs_{r}" for r in RANKS)
)
CARDS_HEADER = (
    "game,owner,rank,base,enter_ply,faceup_ply,faceup_kind,death_ply,died_face_up,paired"
)


def write_corpus(directory: Path, *, agent="random", games=4, first_seed=1, rules="abc123"):
    """A tiny corpus with a known shape: `games` player-games over `games / 2` deals."""
    directory.mkdir(parents=True, exist_ok=True)
    game_rows, card_rows = [], []
    for g in range(games):
        seed = first_seed + g // 2
        for seat in (0, 1):
            won = (g + seat) % 2 == 0
            start = [0] * len(RANKS)
            start[(g + seat) % len(RANKS)] = 1
            start[0] += 1
            unlock = [0] * len(RANKS)
            unlock[(g + seat) % len(RANKS)] = 1
            pairs = [0] * len(RANKS)
            game_rows.append(
                ",".join(
                    str(v)
                    for v in [
                        g, seed, seat, "win" if won else "loss", "", 20, 60, 8,
                        1 + seat, 2 - seat, 0, 6, 0, 4, 3, 0, 0.5, 0.5,
                        *start, *unlock, *pairs,
                    ]
                )
            )
        # Four cards a game: a base card, one flipped, one killed hidden, one still hidden.
        card_rows += [
            f"{g},0,12,1,0,,never,,,0",
            f"{g},0,5,0,2,6,chose,10,1,1",
            f"{g},1,7,0,3,,never,9,0,0",
            f"{g},1,3,0,5,,never,,,0",
        ]
    (directory / "games.csv").write_text(GAMES_HEADER + "\n" + "\n".join(game_rows) + "\n")
    (directory / "cards.csv").write_text(CARDS_HEADER + "\n" + "\n".join(card_rows) + "\n")
    (directory / "meta.json").write_text(
        json.dumps(
            {
                "schema": 1, "agent": agent, "games": games, "first_seed": first_seed,
                "deals": games // 2, "threads": 1, "eval_batch": 1, "elapsed_secs": 1.0,
                "games_per_sec": games, "variant": "split", "two_power": "bottom",
                "rules_name": "test", "rules_hash": rules,
                "rules_label": f"test/{rules}", "config_summary": "test",
                "lanes": 3, "lanes_to_win": 2, "hand_size": 5, "copies_per_rank": 4,
                "stalemate_quiet_plies": 20, "wins_p0": games // 2, "wins_p1": games // 2,
                "draws": 0, "draws_stalemate": 0, "draws_mutual_lane_win": 0,
                "draws_ply_limit": 0, "card_rows": len(card_rows),
                "ranks": RANKS, "powers": POWERS,
            }
        )
    )


# ================================================================ estimators ==


def test_clustered_matches_the_plain_error_when_every_cluster_is_one_row():
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    solo = stats.clustered(values, range(len(values)))
    assert solo.mean == pytest.approx(0.625)
    n = len(values)
    variance = sum((v - solo.mean) ** 2 for v in values) / (n * n) * (n / (n - 1))
    assert solo.half_width == pytest.approx(stats.Z95 * math.sqrt(variance))


def test_clustering_widens_the_interval_when_rows_repeat_within_a_cluster():
    """The reason every interval in the document is clustered.

    The same eight observations, once as eight independent draws and once as four deals each
    played twice. The mean is identical; the interval must not be.
    """
    values = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    independent = stats.clustered(values, range(8))
    paired = stats.clustered(values, [0, 0, 1, 1, 2, 2, 3, 3])
    assert paired.mean == pytest.approx(independent.mean)
    assert paired.half_width > independent.half_width
    # Perfectly duplicated pairs carry half the information, so the interval widens by about
    # √2 — exactly √2 apart from the two different small-sample corrections (8/7 against 4/3).
    ratio = paired.half_width / independent.half_width
    assert ratio == pytest.approx(math.sqrt(2 * (4 / 3) / (8 / 7)))
    assert ratio == pytest.approx(math.sqrt(2), rel=0.12)


def test_an_empty_sample_is_not_a_zero():
    assert not stats.clustered([], [])
    assert stats.clustered([], []).format() == "—"


def test_own_turn_counts_from_the_players_own_first_turn():
    # P0 owns the even plies and P1 the odd, and both start on their turn 1.
    assert own_turn(0) == 1 and own_turn(2) == 2 and own_turn(4) == 3
    assert own_turn(1) == 1 and own_turn(3) == 2 and own_turn(5) == 3
    assert own_turn(None) is None


# ==================================================================== reader ==


def test_chunks_of_one_agent_merge(tmp_path):
    write_corpus(tmp_path / "split" / "random" / "s1-g4", games=4, first_seed=1)
    write_corpus(tmp_path / "split" / "random" / "s3-g4", games=4, first_seed=3)
    corpora = corpus_mod.load_dataset(tmp_path / "split")
    assert len(corpora) == 1
    merged = corpora[0]
    assert merged.n_games == 8, "four games from each chunk"
    assert len(merged.games) == 16, "two player-game rows per game"
    assert merged.meta["deals"] == 4
    # Game indices are shifted so the two chunks do not collide on the join key.
    assert len(set(merged.games.col("game"))) == 8


def test_overlapping_chunks_are_refused(tmp_path):
    write_corpus(tmp_path / "split" / "random" / "s1-g4", games=4, first_seed=1)
    write_corpus(tmp_path / "split" / "random" / "s2-g4", games=4, first_seed=2)
    with pytest.raises(ValueError, match="re-plays"):
        corpus_mod.load_dataset(tmp_path / "split")


def test_corpora_from_different_rulesets_are_refused(tmp_path):
    write_corpus(tmp_path / "split" / "a" / "s1-g4", agent="a", rules="1111")
    write_corpus(tmp_path / "split" / "b" / "s1-g4", agent="b", rules="2222")
    with pytest.raises(ValueError, match="different rulesets"):
        corpus_mod.load_dataset(tmp_path / "split")


def test_a_future_schema_is_refused(tmp_path):
    directory = tmp_path / "split" / "random" / "s1-g4"
    write_corpus(directory)
    meta = json.loads((directory / "meta.json").read_text())
    meta["schema"] = corpus_mod.SCHEMA + 1
    (directory / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="schema"):
        corpus_mod.load_dataset(tmp_path / "split")


def test_an_empty_field_reads_as_missing_not_as_zero(tmp_path):
    write_corpus(tmp_path / "split" / "random" / "s1-g4")
    corpus = corpus_mod.load_dataset(tmp_path / "split")[0]
    faceup = corpus.cards.col("faceup_ply")
    assert None in faceup, "a card that was never flipped has no flip ply"
    assert 0 not in [v for v in faceup if v is not None]


def test_prepared_indexes_cards_by_rank_and_finds_the_opponent(tmp_path):
    write_corpus(tmp_path / "split" / "random" / "s1-g4")
    corpus = corpus_mod.load_dataset(tmp_path / "split")[0]
    p = Prepared(corpus)
    # `rank` in the CSV is a rank *index*, so column value 5 is the 6.
    assert len(p.by_rank[5]) == 4, "one card of rank index 5 per game, four games"
    for i, opponent in enumerate(p.g_opp):
        assert p.g_game[i] == p.g_game[opponent]
        assert p.g_seat[i] != p.g_seat[opponent]
    # The censored end of a tenure: a card that is never flipped and never dies runs to the
    # last turn of its game.
    survivors = [i for i in p.by_rank[3] if p.c_base[i] == 0]
    assert survivors and p.exit_ply(survivors[0]) == p.c_last_ply[survivors[0]]


# ==================================================================== report ==


def render(tmp_path) -> tuple:
    write_corpus(tmp_path / "split" / "alpha" / "s1-g8", agent="alpha", games=8)
    write_corpus(tmp_path / "split" / "beta" / "s1-g8", agent="beta", games=8)
    corpora = corpus_mod.load_dataset(tmp_path / "split")
    ctx = Context(
        dataset="split", root=tmp_path, binary=None, engine_args=[], card_value_games=0
    )
    return corpora, ctx, report.build(corpora, ctx)


def test_every_metric_produces_a_section(tmp_path):
    corpora, ctx, sections = render(tmp_path)
    from duel52.analysis.metrics import METRICS

    assert len(sections) == len(METRICS)
    assert len({s.key for s in sections}) == len(sections), "section keys must be unique"


def test_the_document_renders_in_both_forms(tmp_path):
    corpora, ctx, sections = render(tmp_path)
    md = report.markdown(corpora, sections, ctx)
    assert md.startswith("# Duel 52")
    assert "alpha" in md and "beta" in md
    for section in sections:
        assert f"## {section.title}" in md
    html = report.html_report(corpora, sections, ctx)
    assert html.startswith("<!doctype html>")
    assert html.count("<table") >= len(sections)


def test_every_html_table_sorts_and_every_rank_column_sorts_by_the_deck(tmp_path):
    """The HTML's sortable headers, and the one column whose text does not sort correctly.

    `rank` reads A, 2, … 10, J, Q, K, which as strings falls 10 < 2 < A < J — so it carries an
    explicit key. Everything else is read off the alignment, and a column that gained a `rank`
    heading without a key would sort into nonsense with nothing to say so.
    """
    corpora, ctx, sections = render(tmp_path)
    html = report.html_report(corpora, sections, ctx)
    # The attribute pair, not `aria-sort` alone, which the stylesheet also names.
    assert html.count('<th class') == html.count('tabindex="0" aria-sort="none"') == sum(
        len(b.columns) for s in sections for b in s.blocks if isinstance(b, report.Table)
    ), "every header of every table is sortable"
    assert "<script>" in html

    ranks = corpora[0].ranks
    keyed = 0
    for section in sections:
        for block in section.blocks:
            if isinstance(block, report.Table) and block.columns[0] == "rank":
                assert block.sort_keys.get(0) == [float(i) for i in range(len(ranks))], (
                    f"{section.key}: a rank column without the deck's order"
                )
                keyed += 1
    assert keyed, "the fixture produced no per-rank table, so this test proved nothing"


def test_the_stylesheet_and_script_carry_no_control_characters(tmp_path):
    """`"\\21C5"` in a Python string is an octal escape, not a CSS one.

    It renders as a control character the browser silently drops, so the arrow in a sortable
    header just stops appearing and nothing anywhere says why. Caught once, pinned here.
    """
    corpora, ctx, sections = render(tmp_path)
    for name, text in (("stylesheet", report._STYLE), ("script", report._SCRIPT)):
        bad = {c for c in text if ord(c) < 32 and c not in "\n\t"}
        assert not bad, f"{name} holds {[hex(ord(c)) for c in bad]}"
    assert not {c for c in report.html_report(corpora, sections, ctx)
                if ord(c) < 32 and c not in "\n\t"}


def test_every_markdown_table_is_rectangular(tmp_path):
    """A ragged Markdown table renders as a wrong table rather than as an error."""
    corpora, ctx, sections = render(tmp_path)
    for section in sections:
        for block in section.blocks:
            if isinstance(block, report.Table):
                for row in block.rows:
                    assert len(row) == len(block.columns), (
                        f"{section.key}: {row} against {block.columns}"
                    )


def test_every_figure_is_well_formed_svg(tmp_path):
    import xml.etree.ElementTree as ET

    corpora, ctx, sections = render(tmp_path)
    figures = 0
    for section in sections:
        for block in section.blocks:
            if isinstance(block, report.Figure) and block.svg:
                start = block.svg.index("<svg")
                end = block.svg.index("</svg>") + len("</svg>")
                ET.fromstring(block.svg[start:end])
                figures += 1
    assert figures > 0


def test_a_chart_of_nothing_is_no_chart_rather_than_a_broken_one():
    assert charts.grouped_bars(["a"], [charts.Series("s", [None])]) == ""
    assert charts.lines(["a"], [charts.Series("s", [None])]) == ""
    assert charts.intervals(["a"], [charts.Series("s", [None])]) == ""


# ================================================================ end to end ==

BINARY = Path(__file__).resolve().parents[2] / "target" / "release" / "duel52"


@pytest.mark.skipif(not BINARY.exists(), reason="cargo build --release has not been run")
def test_the_engine_writes_a_corpus_this_reader_can_read(tmp_path):
    """The one test that spans both languages: the CSV the engine writes is the CSV this
    reads, including the columns that are per-rank and therefore per-ruleset."""
    subprocess.run(
        [str(BINARY), "analyze", "--agents", "random", "--games", "20", "--seed", "1",
         "--out", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    corpora = corpus_mod.load_dataset(tmp_path / "split")
    assert len(corpora) == 1
    corpus = corpora[0]
    assert corpus.n_games == 20, "20 games, which is 10 colour-paired deals"
    assert len(corpus.games) == 40, "two player-game rows per game"
    p = Prepared(corpus)
    # Both players open on the same number of cards — the asymmetry the corpus exists to
    # avoid is P0's opening draw landing in `start_*` and P1's not.
    sizes = {
        sum(p.start[r][i] for r in range(len(corpus.ranks)))
        for i in range(len(corpus.games))
    }
    assert len(sizes) == 1, f"opening hands differ in size: {sizes}"
    ctx = Context(dataset="split", root=tmp_path, binary=None, engine_args=[],
                  card_value_games=0)
    written = report.write(corpora, ctx, tmp_path)
    assert all(path.exists() and path.stat().st_size > 0 for path in written)
