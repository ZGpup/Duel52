"""R-NaD, beside AlphaZero — ``PLAN.md`` item 8.

Three groups:

* **Parity with the reference.** ``fixtures/rnad_reference.npz`` holds inputs and outputs
  produced by OpenSpiel's ``rnad.py`` at ``d1dcdf5d`` (``fixtures/make_rnad_reference.py``
  regenerates it). The port in ``duel52.rnad.core`` must reproduce them.
* **The learner finds a known equilibrium.** Kuhn poker, exploitability computed exactly.
* **Nothing AlphaZero relies on moved,** and the Duel 52 plumbing — ``GameBatch``, the
  linear value head, the run loop — does what it says.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from duel52.rnad import core

REPO = Path(__file__).resolve().parents[2]
BINARY = REPO / "target" / "release" / "duel52"
FIXTURE = Path(__file__).parent / "fixtures" / "rnad_reference.npz"


@pytest.fixture(scope="module")
def reference():
    return np.load(FIXTURE)


# ================================================================ reference parity ==


def _vtrace_case(reference, n):
    prefix = f"vtrace{n}_"
    return {k[len(prefix):]: reference[k] for k in reference.files if k.startswith(prefix)}


def _our_vtrace(case, player):
    """Run the scalar-scan port on a reference case and rebuild the reference's outputs."""
    actions = case["actions_oh"]
    a = actions.argmax(-1)
    mu_a = np.take_along_axis(case["mu"], a[..., None], -1)[..., 0]
    pi_a = np.take_along_axis(case["pi"], a[..., None], -1)[..., 0]
    valid = case["valid"].astype(bool)
    ratio = np.where(valid, pi_a / mu_a, 1.0)
    inv_mu = np.where(valid, 1.0 / mu_a, 1.0)
    kl = (case["pi"] * case["log_ratio"]).sum(-1)
    v = case["v"][..., 0]
    eta = float(case["eta"])
    v_target, has_played, coef = core.v_trace(
        v=v, valid=case["valid"], player_id=case["player_id"], policy_ratio=ratio,
        inv_mu=inv_mu, kl=kl, reward=case["rewards"][:, :, player], player=player, eta=eta,
    )
    learning_output = core.learning_output_from(
        v=v, coef=coef, has_played=has_played, merged_log_policy=case["log_ratio"],
        actions_oh=actions, eta=eta,
    )
    return v_target, has_played, learning_output, coef


def test_rnad_vtrace_matches_reference(reference):
    for n in range(int(reference["vtrace_cases"])):
        case = _vtrace_case(reference, n)
        for player in (0, 1):
            v_target, has_played, learning_output, _ = _our_vtrace(case, player)
            np.testing.assert_allclose(v_target, case[f"v_target_{player}"][..., 0], atol=1e-9)
            np.testing.assert_allclose(has_played, case[f"has_played_{player}"], atol=0)
            np.testing.assert_allclose(
                learning_output, case[f"learning_output_{player}"], atol=1e-9
            )


def _flat_nerd_and_value(case):
    """The flat per-step losses the learner computes, fed from a reference case."""
    valid = case["valid"].astype(bool)
    t_idx, b_idx = np.nonzero(valid)
    players = case["player_id"][t_idx, b_idx].astype(int)
    per_player = [_our_vtrace(case, p) for p in (0, 1)]
    coef = np.array([per_player[p][3][t, b] for p, t, b in zip(players, t_idx, b_idx)])
    v_target = np.array([per_player[p][0][t, b] for p, t, b in zip(players, t_idx, b_idx)])
    counts = np.array([np.sum(players == p) for p in (0, 1)], dtype=np.float64)
    weight = 1.0 / counts[players]

    f64 = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)  # noqa: E731
    action = torch.as_tensor(case["actions_oh"].argmax(-1)[t_idx, b_idx])
    advantage = core.nerd_advantage(
        pi=f64(case["pi"][t_idx, b_idx]), log_ratio=f64(case["log_ratio"][t_idx, b_idx]),
        action=action, coef=f64(coef), eta=float(case["eta"]), clip=10_000.0,
    )
    nerd = core.nerd_loss(
        logits=f64(case["logits"][t_idx, b_idx]), advantage=advantage,
        legal=torch.as_tensor(case["legal"][t_idx, b_idx] > 0), weight=f64(weight), beta=2.0,
    )
    online_v = case["online_v"][t_idx, b_idx, 0]
    value = float(np.sum(weight * (online_v - v_target) ** 2))
    return float(nerd), value


def test_rnad_nerd_matches_reference(reference):
    for n in range(int(reference["vtrace_cases"])):
        case = _vtrace_case(reference, n)
        nerd, _ = _flat_nerd_and_value(case)
        assert nerd == pytest.approx(float(case["nerd_loss"]), rel=1e-9, abs=1e-9)


def test_rnad_value_loss_matches_reference(reference):
    for n in range(int(reference["vtrace_cases"])):
        case = _vtrace_case(reference, n)
        _, value = _flat_nerd_and_value(case)
        assert value == pytest.approx(float(case["value_loss"]), rel=1e-9, abs=1e-9)


def test_rnad_legal_policies_match_reference(reference):
    logits = torch.as_tensor(reference["policy_logits"])
    legal = torch.as_tensor(reference["policy_legal"] > 0)
    np.testing.assert_allclose(core.legal_policy(logits, legal).numpy(), reference["policy_pi"], atol=1e-12)
    np.testing.assert_allclose(
        core.legal_log_policy(logits, legal).numpy(), reference["policy_log_pi"], atol=1e-12
    )


def test_rnad_post_processing_matches_reference(reference):
    ours = core.post_process_policy(reference["post_policy"], reference["post_mask"])
    np.testing.assert_allclose(ours, reference["post_out"], atol=1e-12)


def test_rnad_entropy_schedule_matches_reference(reference):
    for n in range(int(reference["schedule_cases"])):
        schedule = core.EntropySchedule(
            sizes=reference[f"schedule{n}_sizes"].tolist(),
            repeats=reference[f"schedule{n}_repeats"].tolist(),
        )
        for step, (alpha, update) in enumerate(
            zip(reference[f"schedule{n}_alpha"], reference[f"schedule{n}_update"])
        ):
            ours = schedule(step)
            assert ours[0] == pytest.approx(float(alpha), abs=1e-12), (n, step)
            assert ours[1] == bool(update), (n, step)


def test_rnad_skipping_a_forced_decision_is_exact_on_policy():
    """Duel 52's actor never offers a forced move, as ``.d52sp`` never records one.

    On policy (``π = μ``) a forced step has ratio 1, no regulariser and no reward, and the
    V-trace target telescopes straight through it: removing the step changes no other step's
    value target or learning-output coefficient.
    """
    rng = np.random.default_rng(3)
    T, B = 10, 4
    players = rng.integers(0, 2, size=(T, B)).astype(float)
    v = rng.normal(size=(T, B))
    kl = np.abs(rng.normal(size=(T, B))) * 0.1
    inv_mu = 1.0 / rng.uniform(0.1, 1.0, size=(T, B))
    reward = np.zeros((T, B))
    reward[-1] = rng.choice([-1.0, 1.0], size=B)
    ones = np.ones((T, B))

    forced = 4
    keep = [t for t in range(T) if t != forced]
    kl_f, inv_f = kl.copy(), inv_mu.copy()
    kl_f[forced], inv_f[forced] = 0.0, 1.0  # a forced step: π = μ = 1, nothing to regularise

    for p in (0, 1):
        full = core.v_trace(v=v, valid=ones, player_id=players, policy_ratio=ones, inv_mu=inv_f,
                            kl=kl_f, reward=reward, player=p, eta=0.2)
        skipped = core.v_trace(v=v[keep], valid=ones[keep], player_id=players[keep],
                               policy_ratio=ones[keep], inv_mu=inv_mu[keep], kl=kl[keep],
                               reward=reward[keep], player=p, eta=0.2)
        np.testing.assert_allclose(full[0][keep], skipped[0], atol=1e-12)
        np.testing.assert_allclose(full[2][keep], skipped[2], atol=1e-12)


# ================================================================ a known equilibrium ==


def test_rnad_kuhn_poker_converges():
    """The whole learner — actor output shape, V-trace, NeuRD, the target average and the
    schedule — drives Kuhn poker's exploitability from 0.458 (uniform) to near zero.

    CPU and fixed seeds, so the run is deterministic: it measured 0.0070 at 3,000 steps. The
    bar sits at 0.02 because sampled trajectories keep exploitability oscillating around 0.01.
    """
    from duel52.rnad.config import RNaDSettings
    from duel52.rnad.kuhn import KuhnNet, exploitability, play_kuhn, policy_table
    from duel52.rnad.learner import RNaDLearner

    torch.manual_seed(0)
    settings = RNaDSettings(
        batch_games=256, learning_rate=1e-3, entropy_schedule_size=[200],
        entropy_schedule_repeats=[1], target_network_avg=0.01,
    )
    learner = RNaDLearner(KuhnNet(), settings, torch.device("cpu"))
    start = exploitability(policy_table(learner.target))
    for step in range(3000):
        learner.step(play_kuhn(learner.online, games=256, rng=np.random.default_rng([1, step])))
    final = exploitability(policy_table(learner.target))
    assert start > 0.2, f"a random init should be far from equilibrium, got {start}"
    assert final < 0.02, f"R-NaD reached exploitability {final:.4f} on Kuhn poker"
    assert learner.reg_updates == 3000 // 200


def test_kuhn_exploitability_is_exact():
    """The judge of the test above: zero at a known equilibrium, 11/24 for uniform play."""
    from duel52.rnad.kuhn import HISTORIES, exploitability

    a = 0.2
    nash = {
        (0, ""): a, (1, ""): 0.0, (2, ""): 3 * a,
        (0, "pb"): 0.0, (1, "pb"): a + 1 / 3, (2, "pb"): 1.0,
        (0, "p"): 1 / 3, (1, "p"): 0.0, (2, "p"): 1.0,
        (0, "b"): 0.0, (1, "b"): 1 / 3, (2, "b"): 1.0,
    }
    assert exploitability(nash) == pytest.approx(0.0, abs=1e-12)
    uniform = {(c, h): 0.5 for c in range(3) for h in HISTORIES}
    assert exploitability(uniform) == pytest.approx(11 / 24, abs=1e-12)


# ============================================================= Duel 52 plumbing ==


def _buffers(batch):
    n, od, ad = len(batch), batch.obs_dim, batch.action_dim
    bufs = bytearray(n * od * 4), bytearray(n * ad), bytearray(n * 4), bytearray(n)
    views = (
        np.frombuffer(bufs[0], dtype=np.float32).reshape(n, od),
        np.frombuffer(bufs[1], dtype=np.uint8).reshape(n, ad),
        np.frombuffer(bufs[2], dtype=np.uint32),
        np.frombuffer(bufs[3], dtype=np.uint8),
    )
    return bufs, views


def test_game_batch_plays_the_same_games_as_game():
    """``PLAN.md`` item 8, Stage 1's exit: a game driven through ``GameBatch`` is the game
    ``Game`` plays — the same observation and legal mask at every decision, the same moves
    forced, the same outcome and length."""
    from duel52 import Game
    from duel52._engine import GameBatch

    games, seed = 4, 40
    batch = GameBatch(games=games, seed=seed, encoding_slots=21, stalemate_value=0.0, threads=3)
    (obs, mask, ids, players), (O, M, I, P) = _buffers(batch)
    rng = np.random.default_rng(0)
    seen: dict[int, list] = {g: [] for g in range(games)}
    while not batch.done():
        k = batch.observe(obs, mask, ids, players)
        picks = []
        for row in range(k):
            index = int(rng.choice(np.flatnonzero(M[row])))
            seen[int(I[row])].append((O[row].copy(), M[row].copy(), int(P[row]), index))
            picks.append(index)
        batch.apply(I[:k].tobytes(), np.asarray(picks, dtype=np.uint32).tobytes())

    for g in range(games):
        game = Game(variant="split", seed=seed + g, encoding_slots=21)
        for o, m, p, index in seen[g]:
            while not game.is_over and game.legal_action_count() == 1:
                game.apply_index(0)
            who = "p0" if p == 0 else "p1"
            assert game.to_move == who
            np.testing.assert_array_equal(np.asarray(game.encode_observation(who), dtype=np.float32), o)
            np.testing.assert_array_equal(np.asarray(game.legal_mask(), dtype=np.uint8), m)
            game.apply(game.decode_action(index))
        while not game.is_over and game.legal_action_count() == 1:
            game.apply_index(0)
        assert game.is_over
        assert game.outcome == batch.outcomes()[g]
        assert game.ply == batch.plies()[g]
        assert len(seen[g]) == batch.decisions()[g]


def test_game_batch_refuses_an_illegal_action():
    from duel52._engine import GameBatch

    batch = GameBatch(games=2, seed=1, encoding_slots=21)
    (obs, mask, ids, players), (_, M, I, _) = _buffers(batch)
    k = batch.observe(obs, mask, ids, players)
    illegal = int(np.flatnonzero(M[0] == 0)[0])
    with pytest.raises(ValueError, match="game 0"):
        batch.apply(I[:1].tobytes(), np.asarray([illegal], dtype=np.uint32).tobytes())
    with pytest.raises(ValueError, match="twice"):
        batch.apply(np.asarray([0, 0], dtype=np.uint32).tobytes(), np.asarray([0, 0], dtype=np.uint32).tobytes())
    with pytest.raises(RuntimeError, match="still running"):
        batch.returns()
    assert k == 2


def test_the_actor_records_the_behaviour_probability_of_every_move():
    """``μ(a)`` is what V-trace divides by, so it must be the probability the sampler used."""
    from duel52.rnad.actor import play_batch
    from duel52.rnad.config import RNaDConfig
    from duel52.rnad.core import legal_policy
    from duel52.rnad.loop import build_model
    from duel52.nn.model import spec_for
    from duel52.train.config import GameSettings, NetSettings

    config = RNaDConfig(net=NetSettings(arch="lane", width=16, blocks=1, value_hidden=8))
    torch.manual_seed(1)
    model = build_model(config, spec_for("split", 21))
    traj = play_batch(model, games=3, seed=5, rng=np.random.default_rng(2), game=GameSettings(),
                      device=torch.device("cpu"), threads=1)
    assert traj.decisions == int(traj.lengths.sum())
    with torch.no_grad():
        logits, _ = model(traj.obs)
        probs = legal_policy(logits, traj.legal)
    torch.testing.assert_close(probs.gather(1, traj.action[:, None]).squeeze(1), traj.mu)
    assert bool(traj.legal.gather(1, traj.action[:, None]).all())
    assert set(np.unique(traj.returns)) <= {-1.0, 1.0, 0.0}


# ======================================================= AlphaZero is left alone ==


def test_the_alphazero_package_does_not_import_rnad():
    """``PLAN.md`` item 8, test 3: the dependency runs one way."""
    code = "import sys, duel52.train, duel52.train.loop; print(any(m.startswith('duel52.rnad') for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


@pytest.mark.parametrize("config", sorted((REPO / "configs").glob("train-*.toml")), ids=lambda p: p.name)
def test_every_alphazero_config_still_loads(config):
    """``PLAN.md`` item 8, test 3, for every shipped training config."""
    from duel52.train.config import load_config

    loaded = load_config(config)
    assert loaded.net.arch in ("mlp", "lane")


def test_a_linear_head_round_trips_and_the_alphazero_trainer_refuses_it(tmp_path):
    from duel52.nn.checkpoint import read_checkpoint, write_checkpoint
    from duel52.nn.model import NetConfig, build_net, lane_spec_for, spec_for
    from duel52.train.config import TrainConfig
    from duel52.train.trainer import Trainer

    spec = spec_for("split", 21)
    lanes = lane_spec_for("split", 21)
    tanh = build_net(NetConfig.from_spec(spec, width=16, blocks=1, value_hidden=8, arch="lane"), lanes)
    linear = build_net(
        NetConfig.from_spec(spec, width=16, blocks=1, value_hidden=8, arch="lane", value_head="linear"),
        lanes,
    )
    write_checkpoint(tmp_path / "tanh.d52nn", model=tanh, spec=spec)
    write_checkpoint(tmp_path / "linear.d52nn", model=linear, spec=spec, learner="rnad")

    plain = (tmp_path / "tanh.d52nn").read_bytes()
    assert b"value_head" not in plain and b"learner" not in plain
    back = read_checkpoint(tmp_path / "linear.d52nn")
    assert (back.value_head, back.learner) == ("linear", "rnad")
    assert read_checkpoint(tmp_path / "tanh.d52nn").value_head == "tanh"

    with pytest.raises(ValueError, match="linear value head"):
        Trainer(TrainConfig(), spec, tmp_path / "linear.d52nn")


# ================================================================== the run loop ==


def _toy_config(tmp_path, **run):
    from duel52.rnad.config import RNaDConfig, RNaDRunSettings, RNaDSettings
    from duel52.train.config import NetSettings

    return RNaDConfig(
        net=NetSettings(arch="lane", width=16, blocks=1, value_hidden=8),
        rnad=RNaDSettings(device="cpu", batch_games=4, entropy_schedule_size=[2], learner_chunk=97),
        run=RNaDRunSettings(seed=11, threads=2, log_every=1, eval_every=0, engine=str(BINARY), **run),
    )


def test_the_rnad_run_resumes_to_the_same_networks_as_an_uninterrupted_one(tmp_path):
    """``PLAN.md`` item 8, Stage 3's exit and test 4: the toy run completes, and on the CPU a
    run stopped after two steps and resumed for two more ends bit-identical to four straight."""
    from duel52.rnad.loop import RNaDRun

    straight = RNaDRun(_toy_config(tmp_path, max_steps=4, eval_opponents=[]), tmp_path / "straight")
    straight.run()

    first = RNaDRun(_toy_config(tmp_path, max_steps=2, eval_opponents=[]), tmp_path / "resumed")
    first.run()
    second = RNaDRun(_toy_config(tmp_path, max_steps=4, eval_opponents=[]), tmp_path / "resumed", resume=True)
    second.run()

    assert straight.learner.steps == second.learner.steps == 4
    assert straight.learner.reg_updates == second.learner.reg_updates == 2
    for name in ("online", "target", "reg", "reg_prev"):
        a = getattr(straight.learner, name).state_dict()
        b = getattr(second.learner, name).state_dict()
        for key in a:
            assert torch.equal(a[key], b[key]), f"{name}.{key} differs after a resume"
    assert (tmp_path / "resumed" / "checkpoints" / "latest.d52nn").exists()
    with pytest.raises(FileExistsError):
        RNaDRun(_toy_config(tmp_path, max_steps=4), tmp_path / "resumed")


def test_the_rnad_checkpoint_plays_through_netsample(tmp_path):
    """The shipped artefact is a ``.d52nn`` the engine plays: ``duel52 match`` with
    ``netsample`` reads it, and the loop parses the score."""
    if not BINARY.exists():
        pytest.skip(f"{BINARY} is not built — run `cargo build --release`")
    from duel52.rnad.loop import RNaDRun

    run = RNaDRun(_toy_config(tmp_path, max_steps=1, eval_games=4, eval_opponents=["random"]), tmp_path / "eval")
    run.run()
    lines = [l for l in (tmp_path / "eval" / "log.jsonl").read_text().splitlines() if '"eval"' in l]
    assert len(lines) == 1 and '"random"' in lines[0]


def test_the_rnad_configs_load_and_check(capsys):
    from duel52.rnad.__main__ import main

    for name in ("rnad-fast.toml", "rnad-3h.toml"):
        assert main(["check", "--config", str(REPO / "configs" / name)]) == 0
    out = capsys.readouterr().out
    assert "value_head=linear" in out
    assert out.count("eval        ") == 2, "check stopped before printing the whole plan"
    # Neither shipped config may have the regularisation policy pinned to its random init.
    assert "⚠️" not in out


def test_check_warns_when_the_target_average_cannot_follow_the_schedule(capsys):
    from duel52.rnad.__main__ import _warn_coupling
    from duel52.rnad.config import RNaDSettings

    _warn_coupling(RNaDSettings(entropy_schedule_size=[100], target_network_avg=0.001))
    assert "⚠️" in capsys.readouterr().out
    _warn_coupling(RNaDSettings())  # the reference: 20,000 × 0.001
    assert "⚠️" not in capsys.readouterr().out
