"""The Phase 3 step 3 training loop, from the Python side.

What is tested here, and what deliberately is not
-------------------------------------------------

Self-play, the shard format and the replay are all Rust and are tested in
``engine/tests/selfplay.rs``, where they are implemented. ``CLAUDE.md``: the engine is the
sole authority, and a second set of assertions about the shard format in Python would be a
second opinion about it.

What is left, and what is here, is the half that only exists in Python: the buffer's batch
assembly, the config loader, and the claim that a batch really carries the sample it says it
does. The ragged gather in :func:`duel52.train.buffer._ragged_gather` is the one piece of
non-obvious index arithmetic in the package — it is vectorised because a Python loop over
512 samples per batch and several thousand batches per generation is minutes of pure
overhead — and getting it subtly wrong would pair observations with the wrong policy targets
and present as a network that will not learn.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from duel52.train.buffer import ReplayBuffer, _ragged_gather, load_generation
from duel52.train.config import load_config
from duel52.train.trainer import resolve_device

REPO = Path(__file__).resolve().parents[2]
ENGINE = REPO / "target" / "release" / "duel52"

needs_engine = pytest.mark.skipif(
    not ENGINE.exists(), reason="build the engine first: cargo build --release"
)


# ============================================================== the ragged gather ==


def test_the_ragged_gather_picks_out_whole_rows():
    # Three samples of lengths 2, 0 and 3 in one CSR payload.
    offset = np.array([0, 2, 2, 5], dtype=np.uint32)
    payload = np.array([10, 11, 20, 21, 22])

    pos, rows = _ragged_gather(offset, np.array([0, 2]))
    assert list(payload[pos]) == [10, 11, 20, 21, 22]
    assert list(rows) == [0, 0, 1, 1, 1]

    # An empty row contributes nothing and does not shift the ones around it.
    pos, rows = _ragged_gather(offset, np.array([1]))
    assert len(pos) == 0 and len(rows) == 0

    # Order follows the request, not the payload.
    pos, rows = _ragged_gather(offset, np.array([2, 0]))
    assert list(payload[pos]) == [20, 21, 22, 10, 11]
    assert list(rows) == [0, 0, 0, 1, 1]


def test_the_ragged_gather_handles_a_repeated_row():
    """Sampling with replacement is the normal case — a batch can draw one sample twice."""
    offset = np.array([0, 2, 4], dtype=np.uint32)
    payload = np.array([1, 2, 3, 4])
    pos, rows = _ragged_gather(offset, np.array([1, 1]))
    assert list(payload[pos]) == [3, 4, 3, 4]
    assert list(rows) == [0, 0, 1, 1]


# ===================================================================== the config ==


def test_a_typo_in_a_config_key_is_an_error():
    """A silent typo in a training config is a two-hour run that did not do what you asked."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write("[selfplay]\ngames = 10\nsimms = 64\n")
        path = f.name
    with pytest.raises(ValueError, match="simms"):
        load_config(path)


def test_an_unknown_section_is_an_error():
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write("[selfpay]\ngames = 10\n")
        path = f.name
    with pytest.raises(ValueError, match="selfpay"):
        load_config(path)


def test_the_shipped_training_config_loads_and_uses_the_safe_slot_bound():
    config = load_config(REPO / "configs" / "train-fast.toml")
    # FINDINGS.md F3.1: 16 does not survive an evaluation ladder against `random`.
    assert config.game.encoding_slots == 21
    # FINDINGS.md F3.6: at the engine default of 0.5 the loop learns to stall.
    assert config.game.stalemate_value == 0.0
    assert "--stalemate-value" in config.game.cli_flags()
    assert "--encoding-slots" in config.game.cli_flags()
    # And the gate has both of its tests.
    assert config.gate.threshold == 0.55
    assert config.gate.reference


def test_the_gate_reads_decisive_games_not_half_points():
    """``FINDINGS.md`` F3.6, as an assertion.

    The exact match that broke the old gate: 199 draws, one win, zero losses. On the
    ordinary scale that is 0.502 — indistinguishable from a dead-even fight, and enough to
    clear a 0.5 bar. On decisive games it is one game's worth of evidence, which is what it
    is.
    """
    from duel52.train.loop import MatchResult

    stalled = MatchResult(score=0.502, ci95=0.005, wins=1, losses=0, draws=199)
    assert stalled.decisive == 1
    assert stalled.decisive_score == 1.0
    assert "none decisive" not in str(stalled)

    dead_stall = MatchResult(score=0.5, ci95=0.0, wins=0, losses=0, draws=200)
    assert dead_stall.decisive == 0
    assert "none decisive" in str(dead_stall)

    real = MatchResult(score=0.7, ci95=0.04, wins=120, losses=40, draws=40)
    assert real.decisive == 160
    assert real.decisive_score == 0.75


def test_the_gate_refuses_a_regression_against_the_reference_panel():
    """The check that would have caught F3.6 at generation 2: the mirror match said 0.502
    and `random` had fallen from 0.929 to 0.600."""
    from dataclasses import replace

    from duel52.train.config import TrainConfig
    from duel52.train.loop import MatchResult, TrainingLoop

    config = TrainConfig()
    loop = TrainingLoop.__new__(TrainingLoop)  # no run directory, no engine, no model
    loop.config = config
    loop.reference_best = {"random": 0.929, "greedy": 0.558}

    stalled_mirror = MatchResult(score=0.502, ci95=0.005, wins=1, losses=0, draws=199)
    promoted, why = loop.judge(stalled_mirror, {"random": 0.600, "greedy": 0.496})
    assert not promoted
    assert "random" in why

    # A candidate that holds its ground on the panel and wins its decisive games passes.
    good = MatchResult(score=0.62, ci95=0.04, wins=90, losses=40, draws=70)
    promoted, why = loop.judge(good, {"random": 0.940, "greedy": 0.600})
    assert promoted, why

    # Holding the panel but losing the decisive games is still a refusal.
    bad = MatchResult(score=0.42, ci95=0.04, wins=40, losses=90, draws=70)
    promoted, why = loop.judge(bad, {"random": 0.940, "greedy": 0.600})
    assert not promoted
    assert "decisive" in why

    # The veto is a high-water mark, so a slow drift cannot ratchet the baseline down one
    # tolerance at a time. 0.89 is inside 0.05 of 0.929, but the panel remembers the best.
    loop.reference_best = {"random": 0.929, "greedy": 0.558}
    promoted, _ = loop.judge(good, {"random": 0.890, "greedy": 0.560})
    assert promoted, "one small give-back is inside tolerance"
    promoted, why = loop.judge(good, {"random": 0.850, "greedy": 0.560})
    assert not promoted, "a second give-back is measured against the same high-water mark"
    assert "best-ever" in why

    # An all-draw mirror with no panel regression abstains rather than blocking, so a run
    # is never deadlocked by the mirror alone.
    promoted, why = loop.judge(
        MatchResult(score=0.5, ci95=0.0, wins=0, losses=0, draws=200),
        {"random": 0.940, "greedy": 0.600},
    )
    assert promoted
    assert "abstains" in why


def test_the_device_string_is_the_whole_handoff():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"mps", "cuda", "cpu"}


# ============================================================= end to end, small ==


@pytest.fixture(scope="module")
def shard(tmp_path_factory) -> Path:
    """A handful of real self-play games, produced by the engine binary."""
    if not ENGINE.exists():
        pytest.skip("engine not built")
    out = tmp_path_factory.mktemp("shards") / "tiny.d52sp"
    subprocess.run(
        [
            str(ENGINE), "selfplay",
            "--checkpoint", str(_tiny_checkpoint(tmp_path_factory)),
            "--out", str(out),
            "--games", "6",
            "--sims", "12",
            "--seed", "1",
            "--quiet",
            "--variant", "split",
            "--encoding-slots", "21",
        ],
        check=True,
        capture_output=True,
    )
    return out


def _tiny_checkpoint(tmp_path_factory) -> Path:
    from duel52.nn.checkpoint import write_checkpoint
    from duel52.nn.model import Duel52Net, NetConfig, spec_for

    spec = spec_for("split", 21)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.d52nn"
    model = Duel52Net(NetConfig(spec["obs_dim"], spec["action_dim"], width=32, blocks=1, value_hidden=16))
    write_checkpoint(path, model=model, spec=spec)
    return path


@needs_engine
def test_a_replayed_generation_is_shaped_the_way_the_buffer_expects(shard):
    gen = load_generation(shard, generation=1)
    assert gen.samples > 20
    assert gen.obs_offset.shape == (gen.samples + 1,)
    assert gen.policy_offset.shape == (gen.samples + 1,)
    assert gen.value.shape == (gen.samples,)
    # Offsets are non-decreasing and end at the payload length: the CSR invariant.
    assert gen.obs_offset[0] == 0 and gen.obs_offset[-1] == len(gen.obs_index)
    assert np.all(np.diff(gen.obs_offset.astype(np.int64)) >= 0)
    assert gen.obs_index.max() < gen.obs_dim
    assert gen.policy_index.max() < gen.action_dim
    assert set(np.unique(gen.value)).issubset({-1.0, 0.0, 1.0})


@needs_engine
def test_a_batch_carries_the_sample_it_says_it_does(shard):
    """The claim the whole trainer rests on: row `i` of the batch is one position's
    observation *and* that same position's policy target, not two different ones."""
    buffer = ReplayBuffer(max_generations=2, max_samples=10**6)
    gen = buffer.add(shard, 1)
    rng = np.random.default_rng(0)
    batch = buffer.sample_batch(rng, batch_size=32)

    # Reassemble row 0 and find it in the generation by its exact non-zero set.
    row0_cols = batch["obs_cols"][batch["obs_rows"] == 0]
    row0_vals = batch["obs_vals"][batch["obs_rows"] == 0]
    matches = [
        i
        for i in range(gen.samples)
        if np.array_equal(gen.obs_index[gen.obs_offset[i] : gen.obs_offset[i + 1]], row0_cols)
    ]
    assert matches, "the batch's first row is not any sample in the generation"
    assert np.array_equal(gen.obs_value[gen.obs_offset[matches[0]] : gen.obs_offset[matches[0] + 1]], row0_vals)

    # Its policy target must be a distribution.
    row0_probs = batch["policy_vals"][batch["policy_rows"] == 0]
    assert abs(float(row0_probs.sum()) - 1.0) < 1e-4
    assert batch["value"][0] in (-1.0, 0.0, 1.0)


@needs_engine
def test_striding_thins_the_generation(shard):
    full = ReplayBuffer(max_generations=1, max_samples=10**6, stride=1)
    thin = ReplayBuffer(max_generations=1, max_samples=10**6, stride=3)
    assert thin.add(shard, 1).samples < full.add(shard, 1).samples


@needs_engine
def test_the_window_drops_the_oldest_generation(shard):
    buffer = ReplayBuffer(max_generations=2, max_samples=10**6)
    buffer.add(shard, 1)
    buffer.add(shard, 2)
    buffer.add(shard, 3)
    assert [g.generation for g in buffer.generations] == [2, 3]


@needs_engine
def test_a_few_steps_of_training_run_and_move_the_loss(shard):
    """Not a claim that it learns anything — a claim that the gradient path is connected.

    A batch that never reaches the loss, or a policy target scattered into the wrong
    columns, both present as a loss that sits exactly at ``ln(action_dim)`` forever.
    """
    from duel52.nn.model import spec_for
    from duel52.train.config import TrainConfig
    from duel52.train.trainer import Trainer

    from dataclasses import replace

    spec = spec_for("split", 21)
    config = TrainConfig()
    config = replace(
        config,
        net=replace(config.net, width=32, blocks=1, value_hidden=16),
        train=replace(config.train, device="cpu", batch_size=32),
    )
    buffer = ReplayBuffer(max_generations=1, max_samples=10**6)
    buffer.add(shard, 1)

    trainer = Trainer(config, spec)
    stats = trainer.fit(buffer, np.random.default_rng(1), steps=12)
    assert stats.steps == 12
    # A uniform policy over `action_dim` logits scores ln(action_dim); anything connected
    # gets below that within a few steps.
    assert stats.policy_first < np.log(spec["action_dim"]) + 0.5
    assert stats.policy_last < stats.policy_first
