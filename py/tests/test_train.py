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


@pytest.mark.parametrize("name", ["train-2h", "train-3h", "train-big"])
def test_the_phase4_configs_carry_the_scale_up(name):
    """``PLAN.md`` §4.2, as an assertion. These are the changes the run is for, and a
    config that quietly lost one of them would produce a result that means something else.
    """
    config = load_config(REPO / "configs" / f"{name}.toml")

    # Change 3: the teacher was capped at 64 and this is the cap coming off.
    assert config.selfplay.sims == 256
    # Change 1: the threshold is the bug, and the sample size is what makes a lower one
    # safe. Three refusals ended the first run; five is the new stopping rule.
    assert config.gate.threshold == 0.52
    assert config.gate.max_consecutive_refusals == 5
    assert config.gate.min_decisive == config.gate.games // 10
    # The gate measures the weights at the budget they are trained for.
    assert config.gate.sims == config.selfplay.sims
    # Change 2: a panel of `random` and `greedy` alone was saturated by generation 19, so a
    # shipped net is the third rung. *Which* one is the run's own business — `train-3h`
    # measures against gen022, the incumbent it warm-starts from, because gen022 already
    # beats gen016 and a gen016 column would re-measure a fight that is over.
    assert any(opponent.startswith("netmcts:models/") for opponent in config.gate.reference)
    # Changes 5 and 7.
    assert config.train.lr_schedule
    assert config.train.holdout_samples > 0
    # A `buffer_samples` cap below the window silently truncates it, which is invisible in
    # the readout. ~136 decisions a game, thinned by the stride.
    window = config.train.buffer_generations * config.selfplay.games * 136 // config.train.sample_stride
    assert config.train.buffer_samples >= window
    # §4.2a is Stage 0b's one experimental change, and it is the only config that carries it.
    assert config.train.lane_augment == (name == "train-3h")


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


def test_the_gate_reports_the_interval_around_its_decisive_score():
    """``FINDINGS.md`` F3.7: the three refusals that ended the first run were every one of
    them inside their own interval of even, and nothing in the readout said so while it was
    happening. A gate whose interval straddles the threshold has not decided anything."""
    from duel52.train.loop import MatchResult

    # The first run's generation 17: 0.503 over 200 games, which is a coin.
    undecided = MatchResult(score=0.503, ci95=0.069, wins=101, losses=99, draws=0)
    assert 0.06 < undecided.decisive_ci95 < 0.08
    assert undecided.decisive_score - undecided.decisive_ci95 < 0.5

    # Four times the games is half the interval: the sample size is what buys a decision.
    bigger = MatchResult(score=0.503, ci95=0.035, wins=404, losses=396, draws=0)
    assert bigger.decisive_ci95 == pytest.approx(undecided.decisive_ci95 / 2, rel=0.02)

    # Nothing decided is not a tight interval around 0.5; it is no interval at all.
    assert MatchResult(score=0.5, ci95=0.0, wins=0, losses=0, draws=200).decisive_ci95 == 0.0
    assert "±" in str(undecided)


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


# ============================================================ the learning rate ==


def test_the_lr_schedule_is_keyed_to_the_generation_index():
    """``PLAN.md`` §4.2 change 5. Keyed to the generation index and not to an optimiser
    step count, because ``--resume`` rebuilds the Trainer and the step count does not
    survive it — a step-keyed schedule silently restarts at full rate after an
    interruption, which on a preemptible box is every few hours."""
    from dataclasses import replace

    from duel52.train.config import TrainConfig
    from duel52.train.trainer import Trainer

    config = TrainConfig()
    config = replace(
        config, train=replace(config.train, lr=2e-3, lr_schedule=[[0, 1.0], [20, 0.25], [38, 0.0625]])
    )
    trainer = Trainer.__new__(Trainer)  # lr_for reads the config and nothing else
    trainer.config = config

    assert trainer.lr_for(1) == pytest.approx(2e-3)
    assert trainer.lr_for(19) == pytest.approx(2e-3)
    assert trainer.lr_for(20) == pytest.approx(5e-4)      # the boundary is inclusive
    assert trainer.lr_for(37) == pytest.approx(5e-4)
    assert trainer.lr_for(38) == pytest.approx(1.25e-4)
    assert trainer.lr_for(999) == pytest.approx(1.25e-4)

    # An empty schedule is a constant rate, so every Phase 3 run reproduces unchanged.
    trainer.config = replace(config, train=replace(config.train, lr_schedule=[]))
    assert trainer.lr_for(0) == trainer.lr_for(50) == pytest.approx(2e-3)


def test_the_fitting_scales_with_the_buffer_rather_than_being_a_fixed_step_count():
    """Generation 1 of `runs/fourth`, as an assertion.

    A fixed 600 steps of batch 512 is 0.9 epochs of a full buffer and **4.1 epochs of the
    74k one a warm-started run holds at generation 1**. Four passes over a quarter of a
    generation, on weights that were already good, memorised the shard: the training value
    MSE fell to 0.225 while the held-out set rose to 0.870, from the 0.774 the starting
    checkpoint came in at. The third run never met this because it started from a random
    net, where an overfitted first generation still beats random.
    """
    from dataclasses import replace

    from duel52.train.config import TrainConfig

    train = TrainConfig().train
    fixed = replace(train, steps_per_generation=600, batch_size=512, epochs_per_generation=0.0)
    # The bug: the same step count is four times the passes when the buffer is a quarter full.
    assert fixed.steps_for(74_250) == fixed.steps_for(340_000) == 600
    assert 600 * 512 / 74_250 == pytest.approx(4.1, abs=0.05)

    scaled = replace(fixed, epochs_per_generation=0.9)
    assert scaled.steps_for(74_250) * 512 / 74_250 == pytest.approx(0.9, abs=0.01)
    assert scaled.steps_for(490_000) * 512 / 490_000 == pytest.approx(0.9, abs=0.01)
    # Never zero steps, however empty the buffer.
    assert scaled.steps_for(1) >= 1

    with pytest.raises(ValueError, match="negative"):
        load_config(_toml("[train]\nepochs_per_generation = -1.0\n"))


def _toml(text: str) -> str:
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write(text)
        return f.name


def test_an_lr_schedule_that_does_not_ascend_is_rejected():
    """Out of order, the last matching entry wins and the decay silently runs backwards."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write("[train]\nlr_schedule = [[20, 0.25], [4, 1.0]]\n")
        path = f.name
    with pytest.raises(ValueError, match="ascend"):
        load_config(path)

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write("[train]\nlr_schedule = [[0, 1.0], [20, 0.0]]\n")
        path = f.name
    with pytest.raises(ValueError, match="positive"):
        load_config(path)


# ================================================================== warm start ==


def test_a_warm_start_refuses_a_checkpoint_the_config_disagrees_with(tmp_path_factory, tmp_path):
    """``[net]`` is ignored when a checkpoint is loaded — the shape comes from the file,
    which is the only thing that can be right about it. That is correct on ``--resume`` and
    a trap on ``--init-from``: someone asking for 6 blocks while inheriting a 3-block
    checkpoint gets 3 blocks and no error."""
    from dataclasses import replace

    from duel52.nn.model import spec_for
    from duel52.train.config import TrainConfig
    from duel52.train.loop import TrainingLoop

    checkpoint = _tiny_checkpoint(tmp_path_factory)  # width 32, blocks 1, value_hidden 16
    loop = TrainingLoop.__new__(TrainingLoop)
    loop.spec = spec_for("split", 21)
    loop.best = tmp_path / "best.d52nn"
    loop.config = TrainConfig()  # the default 128 × 3

    with pytest.raises(ValueError, match="silently mean nothing"):
        loop._warm_start(checkpoint)
    assert not loop.best.exists(), "a refused warm start must not leave an incumbent behind"

    with pytest.raises(FileNotFoundError):
        loop._warm_start(tmp_path / "nope.d52nn")

    config = TrainConfig()
    loop.config = replace(config, net=replace(config.net, width=32, blocks=1, value_hidden=16))
    loop._warm_start(checkpoint)
    assert loop.best.read_bytes() == checkpoint.read_bytes()


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


# ==================================================================== the holdout ==


@needs_engine
def test_a_sliced_generation_is_the_same_rows_with_the_csr_rebased(shard):
    """The slice is what lets a fixed holdout cost 8,000 samples instead of a whole
    generation of self-play. Getting the re-basing wrong would pair observations with the
    wrong policy targets — the same failure the ragged gather is guarded against."""
    gen = load_generation(shard, 1)
    piece = gen.slice(5, 15)

    assert piece.samples == 10
    assert piece.obs_offset[0] == 0 and piece.obs_offset[-1] == len(piece.obs_index)
    assert piece.policy_offset[0] == 0 and piece.policy_offset[-1] == len(piece.policy_index)
    assert np.array_equal(piece.value, gen.value[5:15])

    # Row 0 of the slice is row 5 of the original — observation *and* policy target.
    assert np.array_equal(
        piece.obs_index[piece.obs_offset[0] : piece.obs_offset[1]],
        gen.obs_index[gen.obs_offset[5] : gen.obs_offset[6]],
    )
    assert np.array_equal(
        piece.policy_prob[piece.policy_offset[0] : piece.policy_offset[1]],
        gen.policy_prob[gen.policy_offset[5] : gen.policy_offset[6]],
    )

    # Scoring a holdout means covering it exactly once, not sampling it with replacement.
    assert sum(len(b["value"]) for b in piece.batches(4)) == piece.samples

    with pytest.raises(ValueError):
        gen.slice(0, gen.samples + 1)


@needs_engine
def test_the_holdout_is_carved_off_the_front_and_never_reaches_the_buffer(shard):
    """The whole value of the number is that it was never trained on, and "held out except
    after a restart" is not a distinction anyone would notice in a log file. So the carve
    happens on the way into the buffer, on every path in."""
    from dataclasses import replace

    from duel52.train.config import TrainConfig
    from duel52.train.loop import HOLDOUT_GENERATION, TrainingLoop

    keep = 8
    config = TrainConfig()
    config = replace(config, train=replace(config.train, holdout_samples=keep, sample_stride=1))
    full = load_generation(shard, HOLDOUT_GENERATION, stride=1)

    loop = TrainingLoop.__new__(TrainingLoop)
    loop.config = config
    loop.holdout = None
    loop.buffer = ReplayBuffer(max_generations=4, max_samples=10**6, stride=1)

    trained_on = loop._replay_into_buffer(shard, HOLDOUT_GENERATION)
    assert loop.holdout.samples == keep
    assert trained_on.samples == full.samples - keep
    # The two partition the shard, and the holdout is its front.
    assert np.array_equal(loop.holdout.value, full.value[:keep])
    assert np.array_equal(trained_on.value, full.value[keep:])
    assert loop.buffer.samples == full.samples - keep

    # Only the first generation is carved; every later one goes in whole.
    loop._replay_into_buffer(shard, HOLDOUT_GENERATION + 1)
    assert loop.buffer.generations[-1].samples == full.samples
    assert loop.holdout.samples == keep


@needs_engine
def test_the_optimiser_moments_survive_a_round_trip(shard, tmp_path):
    """``--resume`` rebuilds the Trainer from ``best.d52nn``, which carries weights and
    nothing else. Without this, every interruption throws away AdamW's momentum."""
    from dataclasses import replace

    import torch

    from duel52.nn.model import spec_for
    from duel52.train.config import TrainConfig
    from duel52.train.trainer import Trainer

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
    assert not trainer.load_optimizer(tmp_path / "absent.pt"), "nothing to restore is not an error"
    trainer.fit(buffer, np.random.default_rng(1), steps=3)
    path = trainer.save_optimizer(tmp_path / "optimizer.pt")

    fresh = Trainer(config, spec)
    assert not fresh.optimizer.state_dict()["state"], "a cold AdamW has no moments at all"
    assert fresh.load_optimizer(path)
    warm, cold = trainer.optimizer.state_dict()["state"], fresh.optimizer.state_dict()["state"]
    assert warm.keys() == cold.keys()
    assert torch.allclose(warm[0]["exp_avg"], cold[0]["exp_avg"])
    assert torch.allclose(warm[0]["exp_avg_sq"], cold[0]["exp_avg_sq"])


@needs_engine
def test_the_held_out_score_is_a_number_the_training_batches_cannot_produce(shard):
    """``PLAN.md`` §4.2 change 7. Not a claim that the value head improves — a claim that
    the holdout is scored end to end and reports one sample's worth of loss per sample."""
    from dataclasses import replace

    from duel52.nn.model import spec_for
    from duel52.train.config import TrainConfig
    from duel52.train.trainer import Trainer

    spec = spec_for("split", 21)
    config = TrainConfig()
    config = replace(
        config,
        net=replace(config.net, width=32, blocks=1, value_hidden=16),
        train=replace(config.train, device="cpu", batch_size=16),
    )
    holdout = load_generation(shard, 1).slice(0, 40)
    trainer = Trainer(config, spec)

    stats = trainer.evaluate(holdout)
    assert stats.samples == 40
    # A tanh head against a ±1 target cannot do worse than 4, and an untrained one sits
    # near 1. The policy loss is a cross-entropy over `action_dim` logits.
    assert 0.0 <= stats.value_mse <= 4.0
    assert 0.0 < stats.policy_loss < np.log(spec["action_dim"]) + 0.5
    # A short last batch must not get a full batch's weight: 40 samples in batches of 16 is
    # 16 + 16 + 8, so an unweighted mean would be a different number.
    assert trainer.evaluate(holdout.slice(0, 32)).samples == 32


# ========================================================= lane augmentation ==
#
# ``PLAN.md`` §4.2a. The tables themselves are checked against the encoder in Rust
# (``engine/tests/encoding.rs::phase4_lane_permutation_commutes_with_the_encoder``), which
# is the only place that check is worth making. What is Python's to get wrong is the
# gather: whether a drawn row's observation and its policy target are relabelled by the
# *same* permutation, and whether the holdout is left alone.


def test_the_lane_permutation_tables_arrive_intact():
    from duel52.train.buffer import LaneAugmenter

    from duel52.nn.model import spec_for

    spec = spec_for("split", 21)
    aug = LaneAugmenter.from_engine("split", 21)
    assert aug.count == 6, "|S₃| = 6"
    aug.check(spec["obs_dim"], spec["action_dim"])
    for table in (aug.obs, aug.action):
        assert np.array_equal(table[0], np.arange(table.shape[1])), "the identity comes first"
        for row in table:
            assert len(np.unique(row)) == table.shape[1], "a table that collides is not a relabelling"
    # And a table built for a different board is refused rather than gathering garbage.
    with pytest.raises(ValueError, match="different encoding_slots"):
        LaneAugmenter.from_engine("split", 16).check(spec["obs_dim"], spec["action_dim"])


@needs_engine
def test_augmentation_relabels_a_row_and_its_target_by_the_same_permutation(shard):
    """The invariant the whole transform rests on.

    A relabelling that hit the observation and the policy target differently would train
    the network on one position's board against another's move — and nothing would crash,
    which is why this is a test rather than a comment. The two buffers draw from the same
    seed, so their picks are identical row for row and the comparison is exact.
    """
    from duel52.train.buffer import LaneAugmenter

    aug = LaneAugmenter.from_engine("split", 21)
    plain = ReplayBuffer(max_generations=1, max_samples=10**6)
    moved = ReplayBuffer(max_generations=1, max_samples=10**6, augment=aug)
    plain.add(shard, 1)
    moved.add(shard, 1)

    a = plain.sample_batch(np.random.default_rng(7), batch_size=64)
    b = moved.sample_batch(np.random.default_rng(7), batch_size=64)

    # A relabelling moves indices and nothing else: same rows, same values, same outcome.
    assert np.array_equal(a["obs_rows"], b["obs_rows"])
    assert np.array_equal(a["obs_vals"], b["obs_vals"])
    assert np.array_equal(a["policy_rows"], b["policy_rows"])
    assert np.array_equal(a["policy_vals"], b["policy_vals"])
    assert np.array_equal(a["value"], b["value"])

    used = set()
    for row in range(64):
        obs_before = a["obs_cols"][a["obs_rows"] == row]
        obs_after = b["obs_cols"][b["obs_rows"] == row]
        pol_before = a["policy_cols"][a["policy_rows"] == row]
        pol_after = b["policy_cols"][b["policy_rows"] == row]
        sigma = [
            k
            for k in range(aug.count)
            if np.array_equal(aug.obs[k][obs_before], obs_after)
            and np.array_equal(aug.action[k][pol_before], pol_after)
        ]
        assert sigma, f"row {row} is not any single lane relabelling of the sample it came from"
        used.update(sigma)
    # One σ per sample per draw, so 64 rows must not all share one. (Six permutations of
    # 64 rows: the chance of this failing by luck is 6 · (5/6)^64, about 1 in 10⁴.)
    assert len(used) > 1, "every row got the same permutation — the draw is not per sample"


@needs_engine
def test_the_holdout_is_never_augmented(shard):
    """A yardstick that moves is not a yardstick. ``Generation.batches`` is the holdout's
    only path to the trainer, and it must hand over the samples as they were recorded."""
    from duel52.train.buffer import LaneAugmenter

    gen = load_generation(shard, 1)
    aug = LaneAugmenter.from_engine("split", 21)
    ReplayBuffer(max_generations=1, max_samples=10**6, augment=aug).append(gen)

    first = next(iter(gen.slice(0, 8).batches(8)))
    for row in range(8):
        recorded = gen.obs_index[gen.obs_offset[row] : gen.obs_offset[row + 1]]
        assert np.array_equal(first["obs_cols"][first["obs_rows"] == row], recorded)
