"""Estimators, and the one thing that makes them honest here.

# Every observation belongs to a deal, and observations from one deal are not independent

`ladder.rs` plays every deal **twice**, with the seats swapped, and that is deliberate: it
cancels deal luck out of a score. It also means two rows of `games.csv` were dealt the
identical cards, and that a card's whole life is one draw of one shuffle. Treating those as
independent understates every interval — by about √2 for the paired games alone, and by more
for anything counted per card, where forty rows come out of one deal.

So every mean here is reported with a **cluster-robust interval, clustered on the deal**.
:func:`clustered` is the whole of it, and it is the same eleven lines whatever is being
averaged, which is why no metric in `metrics.py` has to think about it.

The point estimate is unaffected — clustering changes the interval, never the mean.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Z95 = 1.959963984540054


class Estimate:
    """A mean, its 95% half-width, and how many observations and clusters produced it."""

    __slots__ = ("mean", "half_width", "n", "clusters")

    def __init__(self, mean: float, half_width: float, n: int, clusters: int):
        self.mean = mean
        self.half_width = half_width
        self.n = n
        self.clusters = clusters

    def __bool__(self) -> bool:
        return self.n > 0

    def format(self, places: int = 3, plus: bool = False) -> str:
        if not self.n:
            return "—"
        sign = "+" if plus else ""
        if math.isnan(self.half_width):
            return f"{self.mean:{sign}.{places}f}"
        return f"{self.mean:{sign}.{places}f} ± {self.half_width:.{places}f}"

    def compact(self, places: int = 2) -> str:
        return "—" if not self.n else f"{self.mean:.{places}f}"


EMPTY = Estimate(float("nan"), float("nan"), 0, 0)


def clustered(values: Sequence[float], clusters: Sequence) -> Estimate:
    """The mean of ``values`` with a 95% interval robust to clustering by ``clusters``.

    ``Var(mean) = Σ_c (Σ_{i∈c} (x_i − mean))² / n²``, with the usual ``C/(C−1)`` small-sample
    correction. With one observation per cluster this collapses to the ordinary standard
    error, so nothing is lost by using it everywhere.
    """
    n = len(values)
    if n == 0:
        return EMPTY
    mean = math.fsum(values) / n
    sums: Dict[object, float] = {}
    for value, key in zip(values, clusters):
        sums[key] = sums.get(key, 0.0) + (value - mean)
    c = len(sums)
    if c < 2:
        return Estimate(mean, float("nan"), n, c)
    variance = math.fsum(s * s for s in sums.values()) / (n * n) * (c / (c - 1))
    return Estimate(mean, Z95 * math.sqrt(max(variance, 0.0)), n, c)


def plain(values: Sequence[float]) -> Estimate:
    """A mean with an unclustered interval — for quantities with one observation per deal
    already, where clustering would be a no-op."""
    return clustered(values, range(len(values)))


def difference(a: Estimate, b: Estimate) -> str:
    """``a − b`` as text, when the two are independent enough to add variances. Used only for
    contrasts between disjoint sets of games."""
    if not a or not b:
        return "—"
    gap = a.mean - b.mean
    if math.isnan(a.half_width) or math.isnan(b.half_width):
        return f"{gap:+.3f}"
    return f"{gap:+.3f} ± {math.hypot(a.half_width, b.half_width):.3f}"


def median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank, matching `stats.rs` so the two agree on the same data."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(int(round(q * (len(ordered) - 1))), len(ordered) - 1)
    return float(ordered[index])


def share(count: int, total: int) -> float:
    return float("nan") if total == 0 else count / total


# ------------------------------------------------------------------ regression --


class Fit:
    """A logistic fit: coefficients (intercept first) and their covariance.

    The covariance rather than the standard errors, because the table wants a **contrast** —
    what a rank is worth *relative to an average card* — and the variance of a contrast is
    ``cᵀVc``, which the diagonal alone cannot give.
    """

    __slots__ = ("beta", "cov", "n", "clusters")

    def __init__(self, beta, cov, n: int, clusters: int):
        self.beta = beta
        self.cov = cov
        self.n = n
        self.clusters = clusters

    def contrast(self, weights) -> Tuple[float, float]:
        """``(value, 95% half-width)`` for ``weightsᵀβ``."""
        import numpy as np

        w = np.asarray(weights, dtype=np.float64)
        value = float(w @ self.beta)
        variance = float(w @ self.cov @ w)
        return value, Z95 * math.sqrt(max(variance, 0.0))


def logistic_fit(
    rows: Sequence[Sequence[float]],
    targets: Sequence[float],
    clusters: Optional[Sequence] = None,
    ridge: float = 1e-3,
    iterations: int = 40,
) -> Optional["Fit"]:
    """Fit ``P(win) = σ(b0 + Σ b_k x_k)`` by IRLS. ``None`` if numpy is unavailable.

    ``clusters`` gives a cluster-robust sandwich covariance, for the same reason every mean
    here is clustered: both games of a colour-paired deal were dealt the same cards, so the
    model-based covariance would be about √2 too narrow.

    The ridge term keeps the fit defined when one rank is nearly collinear with the rest. It
    is small, and it is named here rather than hidden.
    """
    try:
        import numpy as np
    except ImportError:
        return None

    x = np.asarray(rows, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] == 0:
        return None
    x = np.hstack([np.ones((x.shape[0], 1)), x])
    y = np.asarray(targets, dtype=np.float64)
    beta = np.zeros(x.shape[1])
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0  # never penalise the intercept
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
        w = np.clip(p * (1 - p), 1e-9, None)
        hessian = x.T @ (x * w[:, None]) + penalty
        gradient = x.T @ (y - p) - penalty @ beta
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if np.max(np.abs(step)) < 1e-10:
            break

    p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
    w = np.clip(p * (1 - p), 1e-9, None)
    try:
        bread = np.linalg.inv(x.T @ (x * w[:, None]) + penalty)
    except np.linalg.LinAlgError:
        return None

    groups = 0
    if clusters is None:
        cov = bread
    else:
        scores = x * (y - p)[:, None]
        keys = {}
        for key in clusters:
            if key not in keys:
                keys[key] = len(keys)
        index = np.fromiter((keys[k] for k in clusters), dtype=np.int64, count=len(x))
        groups = len(keys)
        summed = np.zeros((groups, x.shape[1]))
        np.add.at(summed, index, scores)
        meat = summed.T @ summed
        correction = groups / max(groups - 1, 1)
        cov = bread @ meat @ bread * correction
    return Fit(beta, cov, len(x), groups)


def counted(values: Iterable[Optional[float]]) -> List[float]:
    """Drop the missing values. Written once so no metric does it by hand and forgets."""
    return [float(v) for v in values if v is not None]
