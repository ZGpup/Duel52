"""The encoder reserve, from the Python side — ``MODULAR_RULES.md`` §7.

Three things are worth asserting here rather than in Rust:

1. **Nothing moved for canonical rules.** The spec Python builds a checkpoint against is
   byte-for-byte what it was before the reserve, so every file in ``models/`` still loads.
2. **A base-layout checkpoint is refused loudly** by a ruleset that claims the reserve —
   the failure the whole layout-hash mechanism exists to produce.
3. **``python -m duel52.nn widen`` preserves the function.** This is the one that matters:
   the bridge rewrites trained weights, and a mistake in it would not crash — it would make
   a strong agent quietly bad, with the training run as the natural suspect.

The Rust suite (``engine/tests/reserve.rs``) proves the embedding reproduces what the
extended encoder actually writes. This file composes that with the weight transform: embed a
real observation, run both networks, and require the same numbers out.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from duel52._engine import Game, encoding_spec, reserve_embedding  # noqa: E402
from duel52.nn.checkpoint import read_checkpoint  # noqa: E402
from duel52.nn.model import NetConfig, build_net, lane_spec_for, spec_for  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RESERVE_RULES = REPO / "configs" / "rules" / "seven-shield.toml"
SLOTS = 21

#: The layout every shipped checkpoint was written against. Corroborated by the headers of
#: all six files in ``models/``, which predate the reserve.
SHIPPED_OBS_HASH = "b1355a841a1fdc4a"
SHIPPED_ACTION_HASH = "5169f9461d627b39"


def _rules(path: Path) -> str:
    return str(path)


# ------------------------------------------------------------- the base layout holds --


def test_the_canonical_spec_is_unchanged_by_the_reserve():
    """What `models/*.d52nn` were trained against is still what this build produces."""
    spec = spec_for("split", SLOTS)
    assert spec["extended_encoder"] is False
    assert spec["obs_dim"] == 4290
    assert spec["action_dim"] == 2194
    assert spec["obs_layout_hash"] == SHIPPED_OBS_HASH
    assert spec["action_layout_hash"] == SHIPPED_ACTION_HASH
    assert spec["phase_count"] == 7
    assert spec["slot_features"] == 33


def test_every_shipped_checkpoint_still_matches_this_build():
    """The promise stated against the actual files, not against a remembered hash."""
    spec = spec_for("split", SLOTS)
    models = sorted((REPO / "models").glob("*.d52nn"))
    assert models, "no shipped checkpoints to check"
    for path in models:
        ckpt = read_checkpoint(path)
        # Raises if the layout moved. That is the whole point of the header.
        ckpt.check_against(spec)


# ------------------------------------------------------------ the reserve, and loudly --


def test_a_reserve_ruleset_moves_the_layout_and_only_the_layout_it_must():
    base = spec_for("split", SLOTS)
    ext = spec_for("split", SLOTS, _rules(RESERVE_RULES))

    assert ext["extended_encoder"] is True
    assert ext["phase_count"] == 12
    assert ext["slot_features"] == base["slot_features"] + 8
    # 8 flags on every slot of every side of every lane, plus the five spare phase slots.
    assert ext["obs_dim"] == base["obs_dim"] + 3 * 2 * SLOTS * 8 + 5
    # CHOOSE_LANE (a lane on either side) and CHOOSE_OPTION.
    assert ext["action_dim"] == base["action_dim"] + 2 * 3 + 4
    assert {b["name"] for b in ext["action_blocks"]} - {
        b["name"] for b in base["action_blocks"]
    } == {"CHOOSE_LANE", "CHOOSE_OPTION"}
    # The rules hash moves too, but independently: it would move for a Tier 1 change that
    # left the layout alone, and the two are checked in different places for that reason.
    assert ext["rules_hash"] != base["rules_hash"]


def test_every_reserve_ruleset_shares_one_layout():
    """§7's batching argument: one break buys the whole reserve, for every ruleset.

    If each reserve feature had its own switch, the tenth flag-using ruleset would be the
    tenth from-scratch run. They all land on one layout instead, so a checkpoint trained
    under any of them warm-starts any other.
    """
    rules = sorted((REPO / "configs" / "rules").glob("*.toml"))
    layouts = {
        (s["obs_layout_hash"], s["action_layout_hash"])
        for s in (spec_for("split", SLOTS, _rules(p)) for p in rules)
        if s["extended_encoder"]
    }
    assert len(layouts) == 1, f"reserve rulesets disagree on the layout: {layouts}"


def test_a_base_checkpoint_is_refused_by_a_reserve_ruleset():
    """The loud failure. A shipped net cannot silently play a game it has no inputs for."""
    ext = spec_for("split", SLOTS, _rules(RESERVE_RULES))
    ckpt = read_checkpoint(REPO / "models" / "duel52-split-lane-gen032.d52nn")
    with pytest.raises(ValueError) as e:
        ckpt.check_against(ext)
    # The message has to name what moved, or the reader's first guess is the training run.
    assert "obs" in str(e.value).lower()


def test_reserve_embedding_refuses_a_base_ruleset():
    """There is nothing to widen into if the ruleset does not claim the reserve."""
    canonical = REPO / "configs" / "rules" / "canonical.toml"
    with pytest.raises(ValueError, match="base encoder layout"):
        reserve_embedding(str(canonical), SLOTS)


# ------------------------------------------------------------------------ the bridge --


def _model(path: Path, rules_file: str | None):
    ckpt = read_checkpoint(path)
    config = NetConfig(
        obs_dim=ckpt.obs_dim,
        action_dim=ckpt.action_dim,
        width=ckpt.width,
        blocks=ckpt.blocks,
        value_hidden=ckpt.value_hidden,
        arch=ckpt.arch,
    )
    lanes = lane_spec_for("split", SLOTS, rules_file) if ckpt.arch == "lane" else None
    model = build_net(config, lanes)
    model.load_tensors(ckpt.tensors)
    model.eval()
    return model


def test_widen_preserves_the_function(tmp_path):
    """A widened checkpoint computes what it computed before.

    The composition that makes this conclusive: ``engine/tests/reserve.rs``'s
    ``reserve_embedding_preserves_the_encoding`` shows that scattering a base observation
    through the embedding gives exactly what the extended encoder writes for that position.
    So running the widened network on the scattered vector *is* running it on the position,
    and if the policy and value come back unchanged, the widened weights are the same
    function.

    Run on the real champion rather than a random init, because the failure being excluded is
    a trained weight landing in the wrong row — which a random net would show just as well,
    but a real one makes the assertion mean something about a file people actually use.
    """
    from duel52.nn.__main__ import main

    source = REPO / "models" / "duel52-split-lane-gen032.d52nn"
    wide = tmp_path / "wide.d52nn"
    assert (
        main(
            [
                "widen",
                "--in",
                str(source),
                "--out",
                str(wide),
                "--rules-file",
                _rules(RESERVE_RULES),
                "--encoding-slots",
                str(SLOTS),
            ]
        )
        == 0
    )

    base_model = _model(source, None)
    wide_model = _model(wide, _rules(RESERVE_RULES))

    embed = reserve_embedding(_rules(RESERVE_RULES), SLOTS)
    obs_map = np.frombuffer(embed["obs"], dtype="<u4").astype(np.int64)
    action_map = np.frombuffer(embed["action"], dtype="<u4").astype(np.int64)
    ext_dim = embed["extended_obs_dim"]

    checked = 0
    for seed in range(12):
        game = Game(variant="split", seed=seed, encoding_slots=SLOTS)
        # Walk a few plies so the board is not just the opening deal.
        rng = np.random.default_rng(seed)
        for _ in range(int(rng.integers(0, 25))):
            if game.is_over:
                break
            game.apply_index(int(rng.integers(0, game.legal_action_count())))
        if game.is_over:
            continue
        for observer in ("p0", "p1"):
            base_obs = np.asarray(game.encode_observation(observer), dtype=np.float32)
            ext_obs = np.zeros(ext_dim, dtype=np.float32)
            ext_obs[obs_map] = base_obs

            with torch.no_grad():
                p0, v0 = base_model(torch.from_numpy(base_obs).unsqueeze(0))
                p1, v1 = wide_model(torch.from_numpy(ext_obs).unsqueeze(0))

            # ``atol`` rather than exact equality, and the reason is worth stating: the
            # widened input projection is the same weights over a **wider, zero-padded**
            # matrix, and torch's GEMM tiles the reduction differently at the two widths. So
            # the sums reassociate and float32 lands a few ULPs apart — measured at 8e-6 on
            # logits of magnitude ~5, i.e. a relative error of ~1e-6. It is not a weight in
            # the wrong row: that would move a logit by whole units, not by a ULP.
            #
            # The engine's own forward pass does not have this looseness — its input layer
            # walks the observation's non-zeros in index order and the embedding is monotone,
            # so the added zeros are never visited at all.
            assert torch.allclose(v0, v1, atol=1e-5), f"seed {seed}: the value moved"
            assert torch.allclose(
                p0[0], p1[0, torch.from_numpy(action_map)], atol=1e-4
            ), f"seed {seed}: a policy logit moved"
            checked += 1
    assert checked >= 12, "the position sample is too thin to prove anything"


def test_widen_refuses_a_checkpoint_that_is_already_wide(tmp_path):
    """Widening twice would scatter an extended vector through a base-sized map."""
    from duel52.nn.__main__ import main

    source = REPO / "models" / "duel52-split-lane-gen032.d52nn"
    wide = tmp_path / "wide.d52nn"
    main(
        [
            "widen",
            "--in", str(source), "--out", str(wide),
            "--rules-file", _rules(RESERVE_RULES), "--encoding-slots", str(SLOTS),
        ]
    )
    with pytest.raises(ValueError, match="already matches|base layout"):
        main(
            [
                "widen",
                "--in", str(wide), "--out", str(tmp_path / "twice.d52nn"),
                "--rules-file", _rules(RESERVE_RULES), "--encoding-slots", str(SLOTS),
            ]
        )


def test_widen_refuses_a_slot_count_mismatch(tmp_path):
    """A 21-slot checkpoint and a 16-slot layout is a different kind of wrong, and says so."""
    from duel52.nn.__main__ import main

    with pytest.raises(ValueError, match="base layout"):
        main(
            [
                "widen",
                "--in", str(REPO / "models" / "duel52-split-lane-gen032.d52nn"),
                "--out", str(tmp_path / "x.d52nn"),
                "--rules-file", _rules(RESERVE_RULES),
            ]
        )
