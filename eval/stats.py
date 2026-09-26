"""Paired significance statistics for run-vs-base comparisons on the same eval items."""
from __future__ import annotations

from math import comb

import numpy as np

_BOOT_CHUNK = 1000  # bootstrap replicates per vectorised chunk (bounds memory for large item sets)


def paired_bootstrap_ci(
    base_points: list[float],
    run_points: list[float],
    n_boot: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the TOTAL point delta sum(run - base) over paired items.

    Items (not predictions) are resampled with replacement, so the interval reflects which items happen to be
    in the eval set. Lists must be aligned by item. Empty input -> (0.0, 0.0).
    """
    if len(base_points) != len(run_points):
        raise ValueError(f"paired lists differ in length: {len(base_points)} vs {len(run_points)}")
    diffs = np.asarray(run_points, dtype=float) - np.asarray(base_points, dtype=float)
    n = diffs.size
    if n == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    totals = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, _BOOT_CHUNK):
        size = min(_BOOT_CHUNK, n_boot - start)
        idx = rng.integers(0, n, size=(size, n))
        totals[start : start + size] = diffs[idx].sum(axis=1)
    lo, hi = np.quantile(totals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def sign_test(wins: int, losses: int) -> float:
    """Two-sided exact binomial sign test (p=0.5) on non-tied pairs. No non-tied pairs -> 1.0."""
    if wins < 0 or losses < 0:
        raise ValueError(f"wins/losses must be non-negative, got {wins}/{losses}")
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / 2**n)
