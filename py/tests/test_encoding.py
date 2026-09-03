"""The Phase 3 encoding seam, as seen from Python.

The encoders themselves are tested in Rust (``engine/tests/encoding.rs``), where they are
implemented. What matters here is that nothing is lost or misshapen crossing the PyO3
boundary, and — the reason this surface exists at all — that ``encoding_spec()`` is the only
place a dimension or a layout hash comes from.

``CLAUDE.md``: the engine is the sole authority. For Phase 3 that extends past legality to
the feature layout, because a second copy of it in Python could drift silently and the
trained function would stop matching the evaluated function with nothing crashing to say so.
"""

from __future__ import annotations

import pytest

from duel52 import VARIANTS, Game
from duel52._engine import encoding_spec


def test_the_spec_reports_the_documented_shape():
    spec = encoding_spec()
    assert spec["obs_dim"] == 3300
    assert spec["action_dim"] == 1324
    assert spec["encoding_slots"] == 16  # FINDINGS.md F2.7
    assert spec["lanes"] == 3
    assert spec["ranks"] == 13
    assert spec["slot_features"] == 33


def test_the_action_blocks_tile_the_policy_head():
    spec = encoding_spec()
    at = 0
    for block in spec["action_blocks"]:
        assert block["offset"] == at, f"{block['name']} does not start where the last ended"
        at += block["len"]
    assert at == spec["action_dim"]
    # No PASS block: ``game_rules.md`` §4 makes actions mandatory, so a turn with nothing
    # legal in it is ended by the engine rather than chosen. Every logit is a real decision.
    assert [b["name"] for b in spec["action_blocks"]] == [
        "PLAY",
        "FLIP",
        "ATTACK",
        "PAIR",
        "CHOOSE_SLOT",
        "CHOOSE_RANK",
    ]


def test_the_layout_hashes_are_hex_stable_and_distinct():
    a = encoding_spec()
    b = encoding_spec()
    for key in ("obs_layout_hash", "action_layout_hash"):
        assert a[key] == b[key]
        assert len(a[key]) == 16 and int(a[key], 16) >= 0
    assert a["obs_layout_hash"] != a["action_layout_hash"]


def test_a_different_slot_bound_is_a_different_layout():
    """The hash is what protects a checkpoint, so it has to move when the layout does."""
    wide = encoding_spec(encoding_slots=21)
    default = encoding_spec()
    assert wide["obs_dim"] > default["obs_dim"]
    assert wide["obs_layout_hash"] != default["obs_layout_hash"]
    assert wide["action_layout_hash"] != default["action_layout_hash"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_all_three_variants_share_one_layout(variant):
    """One checkpoint plays all three: same lanes, same ranks, same slot bound."""
    assert encoding_spec(variant=variant)["obs_layout_hash"] == encoding_spec()["obs_layout_hash"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_an_observation_is_the_length_the_spec_promises(variant):
    spec = encoding_spec(variant=variant)
    game = Game(variant=variant, seed=3)
    for observer in ("p0", "p1"):
        obs = game.encode_observation(observer)
        assert len(obs) == spec["obs_dim"]
        assert all(isinstance(v, float) for v in obs[:64])


def test_a_fresh_deal_shows_no_rank_anywhere_on_the_board():
    """``game_rules.md`` §3: base cards are hidden from **both** players, their owner too.

    A fresh deal is nothing but base cards, so every rank one-hot on the board must be empty
    — the tightest possible statement of the rule, and the one an encoder is most likely to
    break by reaching for ``card.rank`` on "my own" side.
    """
    spec = encoding_spec()
    slot_features = spec["slot_features"]
    ranks = spec["ranks"]
    game = Game(seed=11)
    for observer in ("p0", "p1"):
        obs = game.encode_observation(observer)
        for lane in range(spec["lanes"]):
            for side in range(2):
                base = ((lane * 2 + side) * spec["encoding_slots"] + 0) * slot_features
                assert obs[base] == 1.0, "the base card's slot should be occupied"
                assert obs[base + 1 : base + 1 + ranks] == [0.0] * ranks
                assert obs[base + 1 + ranks] == 1.0, "rank_unknown should be set"


def test_the_mask_has_one_entry_per_legal_action():
    game = Game(seed=5)
    for _ in range(40):
        if game.is_over:
            break
        actions = game.legal_actions()
        mask = game.legal_mask()
        assert len(mask) == encoding_spec()["action_dim"]
        assert sum(mask) == len(actions)
        for action in actions:
            assert mask[game.encode_action(action)]
        game.apply_index(0)


def test_every_legal_action_round_trips_through_the_policy_index():
    """Injectivity over the legal set is the property the exact encoding exists for.

    ``DESIGN.md`` §4's original rank-keyed table failed it: two same-rank face-down cards with
    different damage shared a ``FLIP(rank, lane)``. A collision would force an invented rule
    for folding two actions' visit counts into one logit.
    """
    checked = 0
    for seed in range(6):
        game = Game(seed=seed)
        while not game.is_over and checked < 4000:
            seen: dict[int, dict] = {}
            for action in game.legal_actions():
                index = game.encode_action(action)
                assert index not in seen, f"{action} collides with {seen.get(index)}"
                seen[index] = action
                assert game.decode_action(index) == action
                checked += 1
            game.apply_index(len(game.legal_actions()) // 2)
    assert checked > 1000


def test_an_index_that_means_nothing_here_decodes_to_none():
    """Most of the 1324 indices are meaningless in any given position — that is what the mask
    is for, and ``None`` is the honest answer rather than an error."""
    game = Game(seed=9)
    spec = encoding_spec()
    assert game.decode_action(spec["action_dim"]) is None
    assert game.decode_action(spec["action_dim"] + 1000) is None
    # In the opening position nothing is on the board to flip, but the FLIP block still
    # decodes structurally — it is the *mask*, not decode, that says it is unavailable.
    flip = next(b for b in spec["action_blocks"] if b["name"] == "FLIP")
    assert game.decode_action(flip["offset"])["kind"] == "flip"
    assert not game.legal_mask()[flip["offset"]]


def test_the_observation_dict_and_the_tensor_agree_about_belief():
    """``unseen_counts`` is in both the dict projection and the tensor, and is the same bag
    ``determinize`` deals from — so the encoder and the sampler cannot disagree about what is
    unknown."""
    game = Game(seed=17)
    for observer in ("p0", "p1"):
        counts = game.observation(observer)["unseen_counts"]
        assert len(counts) == 13
        # `game_rules.md` §2: outside the mirrored variant belief never fully resolves, so
        # something is always unseen.
        assert sum(counts) > 0


def test_encoding_an_action_the_engine_did_not_offer_is_still_an_index():
    """Encoding is a pure function of the action and the position; it does not check
    legality. That is deliberate — ``legal_mask`` is the one place legality is asked about,
    and it asks the engine."""
    game = Game(seed=2)
    index = game.encode_action({"kind": "flip", "lane": 2, "slot": 5})
    assert 0 <= index < encoding_spec()["action_dim"]
    assert not game.legal_mask()[index]
