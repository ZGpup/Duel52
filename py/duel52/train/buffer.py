"""The replay buffer: self-play shards, replayed into tensors, in a sliding window.

One generation is one ``.d52sp`` shard. Adding it replays the trajectories through the Rust
engine — :func:`duel52._engine.replay_shard` — which is the *only* place training data is
encoded. ``CLAUDE.md``: there is exactly one encoder and it is in Rust.

Why the observations stay sparse
--------------------------------

An observation is 4290 floats of which ~205 are non-zero (``FINDINGS.md`` F3.3). A
generation of 3000 games is roughly half a million decisions, so dense storage is 8 GB and
sparse storage is 800 MB. Batches are scattered into a dense tensor on the way to the
device, which is a few milliseconds and never more than ``batch_size`` rows at a time.

The window
----------

Fitting only the newest generation is the classic way to get an agent that chases its own
tail. The buffer keeps the last ``buffer_generations`` shards, subject to a total
``buffer_samples`` cap, and drops from the oldest end — so the network always sees some of
what it used to believe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["Generation", "ReplayBuffer"]


@dataclass
class Generation:
    """One replayed shard, in CSR-ish form.

    ``obs_offset`` has ``samples + 1`` entries; sample ``i`` owns
    ``obs_index[obs_offset[i]:obs_offset[i + 1]]``.
    """

    generation: int
    path: Path
    games: int
    samples: int
    obs_dim: int
    action_dim: int
    obs_offset: np.ndarray
    obs_index: np.ndarray
    obs_value: np.ndarray
    policy_offset: np.ndarray
    policy_index: np.ndarray
    policy_prob: np.ndarray
    value: np.ndarray
    root_value: np.ndarray
    header: dict[str, str]

    @property
    def nbytes(self) -> int:
        return int(
            self.obs_index.nbytes
            + self.obs_value.nbytes
            + self.policy_index.nbytes
            + self.policy_prob.nbytes
            + self.value.nbytes
        )

    def slice(self, start: int, stop: int) -> Generation:
        """Samples ``[start, stop)`` as a generation in their own right.

        Used to carve a fixed holdout off the front of a shard (``PLAN.md`` §4.2 change 7)
        without spending a whole generation of self-play on one. Views into the same
        buffers — the CSR payload is re-based, not copied.

        ``games`` becomes 0: a slice is a set of positions, and the games they came from are
        cut across. Nothing reads it, and a plausible-looking wrong number would be worse.
        """
        if not 0 <= start <= stop <= self.samples:
            raise ValueError(f"slice [{start}, {stop}) is outside 0..{self.samples}")
        obs_from, obs_to = int(self.obs_offset[start]), int(self.obs_offset[stop])
        pol_from, pol_to = int(self.policy_offset[start]), int(self.policy_offset[stop])
        return Generation(
            generation=self.generation,
            path=self.path,
            games=0,
            samples=stop - start,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            obs_offset=self.obs_offset[start : stop + 1] - self.obs_offset[start],
            obs_index=self.obs_index[obs_from:obs_to],
            obs_value=self.obs_value[obs_from:obs_to],
            policy_offset=self.policy_offset[start : stop + 1] - self.policy_offset[start],
            policy_index=self.policy_index[pol_from:pol_to],
            policy_prob=self.policy_prob[pol_from:pol_to],
            value=self.value[start:stop],
            root_value=self.root_value[start:stop],
            header=self.header,
        )

    def batches(self, batch_size: int):
        """Every sample once, in order, in batches of at most ``batch_size``.

        The evaluation counterpart to :meth:`ReplayBuffer.sample_batch`: scoring a holdout
        means covering it exactly once, not sampling it with replacement.
        """
        for start in range(0, self.samples, batch_size):
            rows = np.arange(start, min(start + batch_size, self.samples), dtype=np.int64)
            yield _batch_from([(self, rows)])


def load_generation(path: str | Path, generation: int, *, stride: int = 1, threads: int = 0) -> Generation:
    """Replay a shard into arrays. Zero-copy over the buffers the engine hands back."""
    from .._engine import replay_shard

    d: dict[str, Any] = replay_shard(str(path), threads=threads, stride=stride)
    u32 = lambda key: np.frombuffer(d[key], dtype="<u4")  # noqa: E731
    f32 = lambda key: np.frombuffer(d[key], dtype="<f4")  # noqa: E731
    return Generation(
        generation=generation,
        path=Path(path),
        games=int(d["games"]),
        samples=int(d["samples"]),
        obs_dim=int(d["obs_dim"]),
        action_dim=int(d["action_dim"]),
        obs_offset=u32("obs_offset"),
        obs_index=u32("obs_index"),
        obs_value=f32("obs_value"),
        policy_offset=u32("policy_offset"),
        policy_index=u32("policy_index"),
        policy_prob=f32("policy_prob"),
        value=f32("value"),
        root_value=f32("root_value"),
        header=dict(d["header"]),
    )


def _ragged_gather(offset: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions in a CSR payload for the samples in ``rows``, and the row each belongs to.

    Vectorised on purpose: a Python loop over 512 samples per batch, several thousand
    batches a generation, is minutes of pure overhead.
    """
    starts = offset[rows].astype(np.int64)
    lengths = (offset[rows + 1] - offset[rows]).astype(np.int64)
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    # `arange(total)` counted from the start of each run, i.e. 0,1,2, 0,1, 0,1,2,3, ...
    ends = np.cumsum(lengths)
    within = np.arange(total, dtype=np.int64) - np.repeat(ends - lengths, lengths)
    return np.repeat(starts, lengths) + within, np.repeat(
        np.arange(len(rows), dtype=np.int64), lengths
    )


def _batch_from(pieces: list[tuple[Generation, np.ndarray]]) -> dict[str, np.ndarray]:
    """Assemble one batch out of ``(generation, row indices)`` pairs.

    One code path for both callers, because the invariant they share is the one the whole
    trainer rests on: row *i* of the batch is one position's observation *and that same
    position's* policy target. Rows are laid out piece by piece, so a batch drawn across
    several generations is grouped by generation rather than in draw order — which no
    caller may depend on, and none does.
    """
    obs_rows, obs_cols, obs_vals = [], [], []
    pol_rows, pol_cols, pol_vals = [], [], []
    values = []
    base = 0
    for gen, local in pieces:
        if local.size == 0:
            continue
        pos, row = _ragged_gather(gen.obs_offset, local)
        obs_rows.append(row + base)
        obs_cols.append(gen.obs_index[pos].astype(np.int64))
        obs_vals.append(gen.obs_value[pos])

        pos, row = _ragged_gather(gen.policy_offset, local)
        pol_rows.append(row + base)
        pol_cols.append(gen.policy_index[pos].astype(np.int64))
        pol_vals.append(gen.policy_prob[pos])

        values.append(gen.value[local])
        base += local.size

    empty_i, empty_f = np.empty(0, np.int64), np.empty(0, np.float32)
    join = lambda parts, empty: np.concatenate(parts) if parts else empty  # noqa: E731
    return {
        "obs_rows": join(obs_rows, empty_i),
        "obs_cols": join(obs_cols, empty_i),
        "obs_vals": join(obs_vals, empty_f),
        "policy_rows": join(pol_rows, empty_i),
        "policy_cols": join(pol_cols, empty_i),
        "policy_vals": join(pol_vals, empty_f),
        "value": join(values, empty_f),
    }


class ReplayBuffer:
    """A sliding window over replayed generations."""

    def __init__(self, *, max_generations: int, max_samples: int, stride: int = 1, threads: int = 0):
        self.max_generations = max_generations
        self.max_samples = max_samples
        self.stride = stride
        self.threads = threads
        self.generations: list[Generation] = []

    # ------------------------------------------------------------------ contents --

    @property
    def samples(self) -> int:
        return sum(g.samples for g in self.generations)

    @property
    def nbytes(self) -> int:
        return sum(g.nbytes for g in self.generations)

    def add(self, path: str | Path, generation: int) -> Generation:
        return self.append(load_generation(path, generation, stride=self.stride, threads=self.threads))

    def append(self, gen: Generation) -> Generation:
        """Put an already-replayed generation into the window.

        Split out from :meth:`add` so a caller that has to look at a shard before it is
        trained on — carving the fixed holdout off the front of it — does not have to
        replay it twice.
        """
        if self.generations:
            first = self.generations[0]
            if (gen.obs_dim, gen.action_dim) != (first.obs_dim, first.action_dim):
                # Only reachable if the encoder changed mid-run, which would silently mix
                # two layouts into one gradient step.
                raise ValueError(
                    f"{path} replays to obs_dim={gen.obs_dim} action_dim={gen.action_dim}, but "
                    f"the buffer holds {first.obs_dim}/{first.action_dim}"
                )
        self.generations.append(gen)
        while len(self.generations) > self.max_generations or (
            len(self.generations) > 1 and self.samples > self.max_samples
        ):
            self.generations.pop(0)
        return gen

    # ------------------------------------------------------------------ batching --

    def sample_batch(self, rng: np.random.Generator, batch_size: int) -> dict[str, np.ndarray]:
        """Draw a batch uniformly over every sample currently held.

        Sampling across the whole window rather than per-generation is deliberate: a
        generation with more decisions in it *should* contribute more, because a longer game
        genuinely contains more positions.
        """
        if not self.generations:
            raise RuntimeError("the replay buffer is empty")
        sizes = np.array([g.samples for g in self.generations], dtype=np.int64)
        bounds = np.concatenate([[0], np.cumsum(sizes)])
        picks = rng.integers(0, bounds[-1], size=batch_size)
        which = np.searchsorted(bounds, picks, side="right") - 1

        return _batch_from(
            [
                (gen, (picks[which == g_index] - bounds[g_index]).astype(np.int64))
                for g_index, gen in enumerate(self.generations)
            ]
        )

    # ------------------------------------------------------------- diagnostics --

    def value_target_mix(self) -> tuple[float, float, float]:
        """Share of value targets that are a win, a draw and a loss.

        Worth watching: an early network draws most of its games (the stalemate rule at
        `game_rules.md` §7 fires when neither side wants to attack), and a value head trained
        almost entirely on zeros learns nothing. If this stays near all-draws for several
        generations, the run is stuck rather than slow.
        """
        if not self.generations:
            return (0.0, 0.0, 0.0)
        v = np.concatenate([g.value for g in self.generations])
        n = max(len(v), 1)
        return (
            float((v > 0.5).sum() / n),
            float((np.abs(v) <= 0.5).sum() / n),
            float((v < -0.5).sum() / n),
        )
