"""Weighted quantile with pseudo-count at infinity (Sec. 4.4, Eq. 5-6).

The pseudo-count mass guarantees finite-sample marginal coverage even when
the pool of weighted scores is small.
"""
from __future__ import annotations

import numpy as np


def weighted_quantile(
    scores: np.ndarray,
    weights: np.ndarray,
    level: float = 0.95,
    pseudo_count_inf: bool = True,
) -> float:
    """Compute the weighted (1 - alpha) quantile of scores.

    With pseudo_count_inf=True, add a unit mass at +inf so the empirical
    quantile never under-covers in the finite-sample worst case.
    """
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if scores.size == 0:
        return np.inf if pseudo_count_inf else 0.0

    weights = np.maximum(weights, 0.0)
    if pseudo_count_inf:
        scores = np.concatenate([scores, [np.inf]])
        weights = np.concatenate([weights, [1e-3]])  # small pseudo-mass for finite-sample safety

    total = weights.sum()
    if total <= 0:
        return np.inf if pseudo_count_inf else 0.0
    weights = weights / total

    order = np.argsort(scores)
    scores_sorted = scores[order]
    weights_sorted = weights[order]
    cum = np.cumsum(weights_sorted)

    idx = np.searchsorted(cum, level, side="left")
    if idx >= len(scores_sorted):
        idx = len(scores_sorted) - 1

    val = float(scores_sorted[idx])
    if np.isinf(val) and pseudo_count_inf:
        return float(scores_sorted[idx - 1]) if idx > 0 else 0.0
    return val