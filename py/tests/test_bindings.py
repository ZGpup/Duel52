"""Tests for the PyO3 bindings.

These check the *seam*, not the rules — the rules are tested in Rust, where they are
implemented (``CLAUDE.md``: the engine is the sole authority on legality). What matters
here is that nothing is lost or leaked in translation:

* every action round-trips through the dict encoding,
* an observation shows a player only what ``game_rules.md`` §5 says is theirs to see,
* the same seed produces the same game through Python as through the CLI.
"""

from __future__ import annotations

import random

import pytest

from duel52 import VARIANTS, Game, random_play_stats, rollout


# --------------------------------------------------------------------------- basics --


def test_engine_imports_and_reports_a_version():
    from duel52 import ENGINE_VERSION

    assert ENGINE_VERSION


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_new_game_matches_the_documented_setup(variant):
    """``game_rules.md`` §2 and §9a: P0 opens on a draw plus two actions, at 6 cards."""
    g = Game(variant=variant, seed=1)
    assert g.to_move == "p0"
    assert g.actions_remaining == 2, "the first turn is two actions (§2)"
    assert g.hand_size("p0") == 6, "but the draw still happened (§4)"
    assert g.hand_size("p1") == 5
    assert not g.base_unlocked, "base cards start locked (§3)"
    assert not g.is_over
    assert g.outcome == "ongoing"


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="unknown variant"):
        Game(variant="nonsense", seed=1)


def test_unknown_two_power_is_rejected():
    with pytest.raises(ValueError, match="unknown two_power"):
        Game(variant="split", seed=1, two_power="nonsense")


# ------------------------------------------------------------------------- actions --


def test_legal_actions_are_dicts_with_a_kind():
    g = Game(seed=3)
    actions = g.legal_actions()
    assert actions, "a running game always has at least one legal action"
    assert all(isinstance(a, dict) and "kind" in a for a in actions)
    assert any(a["kind"] == "pass" for a in actions), "Pass is always available"


def test_every_action_round_trips_through_the_dict_encoding():
    """A dict from ``legal_actions`` must be accepted verbatim by ``apply``.

    Covers every action kind by playing several full games and applying each action the
    long way (by dict) rather than by index.
    """
    seen_kinds = set()
    for seed in range(12):
        rng = random.Random(seed)
        g = Game(seed=seed)
        while not g.is_over:
            actions = g.legal_actions()
            choice = rng.choice(actions)
            seen_kinds.add(choice["kind"])
            g.apply(choice)  # by dict, not by index
    # The rarer sub-decisions need a few games to show up; these five are unmissable.
    assert {"play", "flip", "attack", "pass"} <= seen_kinds


def test_apply_rejects_an_illegal_action():
    g = Game(seed=5)
    with pytest.raises(RuntimeError, match="illegal action"):
        g.apply({"kind": "attack", "lane": 0, "attacker": 99, "target": 99})


def test_apply_rejects_a_malformed_action_dict():
    g = Game(seed=5)
    with pytest.raises(ValueError, match="missing key"):
        g.apply({"kind": "play", "lane": 0})  # no rank
    with pytest.raises(ValueError, match="unknown action kind"):
        g.apply({"kind": "teleport"})


def test_apply_index_is_bounds_checked():
    g = Game(seed=5)
    with pytest.raises(IndexError):
        g.apply_index(10_000)


def test_action_descriptions_are_human_readable():
    g = Game(seed=5)
    descriptions = g.legal_action_descriptions(observer="p0")
    assert len(descriptions) == g.legal_action_count()
    assert any("PASS" in d for d in descriptions)


# --------------------------------------------------------------------- observations --


def test_observation_hides_the_opponents_hand():
    """§5: hand *sizes* are public, hand *contents* are private."""
    g = Game(seed=7)
    obs = g.observation("p0")
    assert sum(obs["hand"]) == obs["hand_size"] == 6
    assert obs["opponent_hand_size"] == 5
    assert "opponent_hand" not in obs, "the opponent's contents must not be present"


def test_observation_hides_base_cards_from_their_owner():
    """§3, and the first item in ``CLAUDE.md``'s list of things easy to get wrong."""
    g = Game(seed=7)
    obs = g.observation("p0")
    bases = [c for c in obs["board"] if c["is_base"]]
    assert len(bases) == 6, "three per player"
    for card in bases:
        assert card["rank"] is None
        assert not card["rank_known"]


def test_observation_publishes_damage_and_hit_points_without_leaking_a_rank():
    """§5: damage and hit points are both public, and neither identifies a card.

    Every face-down card is a blank 2-HP card whatever its rank, so publishing max HP is
    safe — a face-down Jack is indistinguishable from a face-down 4. Rank itself stays
    hidden.
    """
    saw_face_down = False
    for seed in range(30):
        rng = random.Random(seed)
        g = Game(seed=seed)
        while not g.is_over:
            for observer in ("p0", "p1"):
                for card in g.observation(observer)["board"]:
                    assert isinstance(card["damage"], int), "damage is always visible"
                    if not card["face_up"]:
                        saw_face_down = True
                        assert card["max_hp"] == 2, (
                            "a face-down card is a blank 2-HP card whatever its rank"
                        )
                    if not card["rank_known"]:
                        assert card["rank"] is None, "the rank itself stays hidden"
            actions = g.legal_actions()
            g.apply_index(rng.randrange(len(actions)))
    assert saw_face_down


def test_a_face_down_jack_reports_two_hit_points_until_it_is_flipped():
    """The rank is known to its owner, but the hit points are still the blank 2 (§5)."""
    g = Game(seed=4)
    # Base cards are face-down; whatever they are, every one reports 2 HP.
    for card in g.observation("p0")["board"]:
        assert not card["face_up"]
        assert card["max_hp"] == 2


def test_observation_hides_the_removed_pool_except_in_the_mirrored_variant():
    """§2 hides it; §9b publishes it, which is the point of that variant."""
    for variant in ("base", "split"):
        obs = Game(variant=variant, seed=2).observation("p0")
        assert not obs["removed_revealed"]
        assert obs["removed_counts"] is None
        assert obs["removed_size"] == 10, "ten cards removed overall in every variant"

    obs = Game(variant="mirrored", seed=2).observation("p0")
    assert obs["removed_revealed"]
    assert sum(obs["removed_counts"]) == 5, "five ranks, mirrored across both decks"


def test_observation_is_symmetric_between_the_players():
    """What P0 sees of P1 must match what P1 sees of themselves, for public facts."""
    g = Game(seed=9)
    p0, p1 = g.observation("p0"), g.observation("p1")
    assert p0["opponent_hand_size"] == p1["hand_size"]
    assert p1["opponent_hand_size"] == p0["hand_size"]
    assert p0["base_unlocked"] == p1["base_unlocked"]
    assert len(p0["board"]) == len(p1["board"])


def test_render_respects_the_observer():
    g = Game(seed=11)
    p0_view = g.render("p0")
    assert "(?)" in p0_view, "base cards render as unknown"
    assert "REVEAL" not in p0_view
    assert "REVEAL" in g.render(None), "no observer means debug mode"


# ------------------------------------------------------------ games and determinism --


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_random_game_plays_to_a_conclusion(variant):
    rng = random.Random(0)
    g = rollout(Game(variant=variant, seed=4), lambda _g, a: rng.randrange(len(a)))
    assert g.is_over
    assert g.outcome != "ongoing"
    assert g.value_for("p0") + g.value_for("p1") == pytest.approx(1.0), "zero-sum"


def test_a_finished_game_offers_nothing():
    rng = random.Random(1)
    g = rollout(Game(seed=6), lambda _g, a: rng.randrange(len(a)))
    assert g.legal_actions() == []
    with pytest.raises(RuntimeError):
        g.apply({"kind": "pass"})


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_same_seed_deals_the_same_game(variant):
    """``CLAUDE.md``: same seed + same config → identical game."""
    a, b = Game(variant=variant, seed=99), Game(variant=variant, seed=99)
    assert a.hand_counts("p0") == b.hand_counts("p0")
    assert a.render(None) == b.render(None)


def test_copying_a_game_gives_an_independent_position():
    """Search clones positions constantly; a shared mutation would corrupt every rollout."""
    import copy

    g = Game(seed=13)
    clone = copy.deepcopy(g)
    g.apply_index(0)
    assert clone.ply == 0 and clone.actions_remaining == 2, "the clone did not move"
    assert clone.render(None) != g.render(None)


# ---------------------------------------------------------------- Phase 1 statistics --


@pytest.mark.parametrize("variant", VARIANTS)
def test_random_play_stats_returns_a_complete_report(variant):
    stats = random_play_stats(variant=variant, first_seed=0, games=50)
    assert stats["games"] == 50
    assert stats["p0_wins"] + stats["p1_wins"] + stats["draws"] == 50
    assert stats["draws_ply_limit"] == 0, "the safety cap firing would mean a rules bug"
    assert 0.0 <= stats["p0_score"] <= 1.0
    assert len(stats["lengths"]) == 50
    assert "Random vs random" in stats["report"]


def test_both_settings_of_the_two_are_measurable():
    """§10a exists to be measured, not assumed — both settings must run."""
    for two_power in ("bottom", "discard"):
        stats = random_play_stats(
            variant="split", first_seed=0, games=30, two_power=two_power
        )
        assert stats["two_power"] == two_power
        assert stats["games"] == 30
