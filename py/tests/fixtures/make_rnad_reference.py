"""Generate ``rnad_reference.npz`` from the R-NaD reference implementation — ``PLAN.md`` item 8.

Run once, in a throwaway environment, and commit the output. The test suite reads the
``.npz`` and never needs JAX.

    python3 -m venv /tmp/jaxenv && /tmp/jaxenv/bin/pip install jax chex dm-haiku optax numpy
    gh api "repos/google-deepmind/open_spiel/contents/open_spiel/python/algorithms/rnad/rnad.py?ref=d1dcdf5dc9c0a98a0ff0a6b476236d0cc91bf2d5" \
        --jq .content | base64 --decode > /tmp/rnad.py
    /tmp/jaxenv/bin/python py/tests/fixtures/make_rnad_reference.py /tmp/rnad.py \
        py/tests/fixtures/rnad_reference.npz

The reference module imports ``pyspiel`` and ``open_spiel.python.policy`` at the top, for the
solver class; the functions exercised here use neither, so both are stubbed.
"""

from __future__ import annotations

import importlib.util
import sys
import types

import numpy as np


def load_reference(path: str):
    pyspiel = types.ModuleType("pyspiel")
    pyspiel.State = object  # only named in type annotations
    open_spiel = types.ModuleType("open_spiel")
    python = types.ModuleType("open_spiel.python")
    policy = types.ModuleType("open_spiel.python.policy")
    policy.Policy = object
    python.policy = policy
    open_spiel.python = python
    sys.modules.update(
        {
            "pyspiel": pyspiel,
            "open_spiel": open_spiel,
            "open_spiel.python": python,
            "open_spiel.python.policy": policy,
        }
    )
    spec = importlib.util.spec_from_file_location("rnad_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def softmax_legal(rng, T, B, A, legal):
    logits = rng.normal(size=(T, B, A)) * 1.5
    logits = np.where(legal, logits, -np.inf)
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.where(legal, np.exp(logits), 0.0)
    return p / p.sum(axis=-1, keepdims=True)


def vtrace_case(rnad, rng, T, B, A, eta):
    import jax.numpy as jnp

    legal = rng.random((T, B, A)) < 0.6
    legal[..., 0] |= ~legal.any(axis=-1)  # at least one legal action everywhere
    lengths = rng.integers(1, T + 1, size=B)
    valid = (np.arange(T)[:, None] < lengths[None, :]).astype(np.float64)
    # Runs of the same player, as Duel 52's three-action turns produce.
    player_id = np.zeros((T, B))
    for b in range(B):
        p, t = int(rng.integers(0, 2)), 0
        while t < T:
            run = int(rng.integers(1, 4))
            player_id[t : t + run, b] = p
            p, t = 1 - p, t + run

    mu = softmax_legal(rng, T, B, A, legal)
    pi = softmax_legal(rng, T, B, A, legal)
    pi_reg = softmax_legal(rng, T, B, A, legal)
    log_ratio = np.where(legal, np.log(np.where(legal, pi, 1.0)) - np.log(np.where(legal, pi_reg, 1.0)), 0.0)

    actions = np.zeros((T, B, A))
    for t in range(T):
        for b in range(B):
            actions[t, b, rng.choice(A, p=mu[t, b])] = 1.0

    v = rng.normal(size=(T, B, 1)) * 0.5
    outcome = rng.choice([-1.0, 1.0], size=B)
    rewards = np.zeros((T, B, 2))
    for b in range(B):
        rewards[lengths[b] - 1, b, 0] = outcome[b]
        rewards[lengths[b] - 1, b, 1] = -outcome[b] if rng.random() < 0.8 else -1.0

    case = {
        "v": v, "valid": valid, "player_id": player_id, "mu": mu, "pi": pi,
        "log_ratio": log_ratio, "actions_oh": actions, "rewards": rewards, "legal": legal.astype(np.float64),
        "eta": np.float64(eta),
    }
    for player in (0, 1):
        v_target, has_played, learning_output = rnad.v_trace(
            jnp.asarray(v), jnp.asarray(valid), jnp.asarray(player_id), jnp.asarray(mu),
            jnp.asarray(pi), jnp.asarray(log_ratio),
            rnad._player_others(jnp.asarray(player_id), jnp.asarray(valid), player),
            jnp.asarray(actions), jnp.asarray(rewards[:, :, player]), player,
            lambda_=1.0, c=1.0, rho=np.inf, eta=eta,
        )
        case[f"v_target_{player}"] = np.asarray(v_target)
        case[f"has_played_{player}"] = np.asarray(has_played)
        case[f"learning_output_{player}"] = np.asarray(learning_output)

    logits = rng.normal(size=(T, B, A)) * 3.0
    case["logits"] = logits
    q_list = [jnp.asarray(case[f"learning_output_{p}"]) for p in (0, 1)]
    ones = jnp.expand_dims(jnp.ones_like(jnp.asarray(valid)), axis=-1)
    case["nerd_loss"] = np.float64(
        rnad.get_loss_nerd(
            [jnp.asarray(logits)] * 2, [jnp.asarray(pi)] * 2, q_list, jnp.asarray(valid),
            jnp.asarray(player_id), jnp.asarray(legal.astype(np.float64)), [ones] * 2,
            clip=10_000, threshold=2.0,
        )
    )
    online_v = rng.normal(size=(T, B, 1))
    case["online_v"] = online_v
    case["value_loss"] = np.float64(
        rnad.get_loss_v(
            [jnp.asarray(online_v)] * 2,
            [jnp.asarray(case[f"v_target_{p}"]) for p in (0, 1)],
            [jnp.asarray(case[f"has_played_{p}"]) for p in (0, 1)],
        )
    )
    return case


def main(reference_path: str, out_path: str) -> None:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    rnad = load_reference(reference_path)
    rng = np.random.default_rng(20260913)
    arrays: dict[str, np.ndarray] = {}

    for n, (T, B, A, eta) in enumerate([(12, 5, 4, 0.2), (30, 8, 7, 0.2), (9, 3, 2, 0.5)]):
        for key, value in vtrace_case(rnad, rng, T, B, A, eta).items():
            arrays[f"vtrace{n}_{key}"] = np.asarray(value)
    arrays["vtrace_cases"] = np.int64(3)

    # Policies from logits, including rows with a single legal action.
    logits = rng.normal(size=(40, 9)) * 4.0
    legal = rng.random((40, 9)) < 0.5
    legal[:, 3] |= ~legal.any(axis=-1)
    legal[0] = False
    legal[0, 5] = True
    arrays["policy_logits"] = logits
    arrays["policy_legal"] = legal.astype(np.float64)
    arrays["policy_pi"] = np.asarray(rnad._legal_policy(jnp.asarray(logits), jnp.asarray(legal)))
    arrays["policy_log_pi"] = np.asarray(
        rnad.legal_log_policy(jnp.asarray(logits), jnp.asarray(legal.astype(np.float64)))
    )

    # Post-processing: ordinary rows, ties, and a row under the threshold everywhere.
    policies = rng.dirichlet(np.ones(40) * 0.3, size=30)
    policies[0] = np.full(40, 1.0 / 40)
    policies[1, :4] = [0.25, 0.25, 0.25, 0.25]
    policies[1, 4:] = 0.0
    masks = np.ones_like(policies)
    tune = rnad.FineTuning(from_learner_steps=0, policy_threshold=0.03, policy_discretization=32)
    arrays["post_policy"] = policies
    arrays["post_mask"] = masks
    arrays["post_out"] = np.asarray(
        tune.post_process_policy(jnp.asarray(policies), jnp.asarray(masks))
    )

    # The schedule, for a few shapes, at every step through two full passes and beyond.
    for n, (sizes, repeats) in enumerate([([10], [1]), ([3, 5, 10], [2, 4, 1]), ([7, 2], [3, 1])]):
        schedule = rnad.EntropySchedule(sizes=sizes, repeats=repeats)
        steps = np.arange(0, 80)
        alphas, updates = zip(*(schedule(int(s)) for s in steps))
        arrays[f"schedule{n}_sizes"] = np.asarray(sizes)
        arrays[f"schedule{n}_repeats"] = np.asarray(repeats)
        arrays[f"schedule{n}_alpha"] = np.asarray([float(a) for a in alphas])
        arrays[f"schedule{n}_update"] = np.asarray([bool(u) for u in updates])
    arrays["schedule_cases"] = np.int64(3)

    np.savez_compressed(out_path, **arrays)
    print(f"wrote {out_path}: {len(arrays)} arrays")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
