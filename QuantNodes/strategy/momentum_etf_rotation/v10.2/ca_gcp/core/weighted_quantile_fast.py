"""Fast weighted quantile for batch queries.

When the same pool of weighted scores is queried many times (one per
test day), pre-sorting once and reusing the sorted order with
np.searchsorted is much faster than re-sorting per query.
"""
from __future__ import annotations

import numpy as np


class PrecomputedWeightedQuantile:
    """Cache the sorted scores and cumulative weights for fast queries."""

    def __init__(
        self,
        scores: np.ndarray,
        weights: np.ndarray,
        pseudo_count_inf: bool = True,
    ):
        scores = np.asarray(scores, dtype=float)
        weights = np.asarray(weights, dtype=float)
        weights = np.maximum(weights, 0.0)
        if scores.size == 0:
            self._sorted = np.array([np.inf])
            self._cum = np.array([1.0])
            self._pseudo = pseudo_count_inf
            return
        if pseudo_count_inf:
            scores = np.concatenate([scores, [np.inf]])
            weights = np.concatenate([weights, [1e-3]])
        total = weights.sum()
        if total <= 0:
            self._sorted = np.array([np.inf])
            self._cum = np.array([1.0])
            self._pseudo = pseudo_count_inf
            return
        norm_w = weights / total
        order = np.argsort(scores, kind="mergesort")
        self._sorted = scores[order]
        self._cum = np.cumsum(norm_w[order])
        self._pseudo = pseudo_count_inf

    def query(self, level: float) -> float:
        if len(self._cum) == 0:
            return np.inf if self._pseudo else 0.0
        idx = int(np.searchsorted(self._cum, level, side="left"))
        if idx >= len(self._sorted):
            idx = len(self._sorted) - 1
        val = float(self._sorted[idx])
        if np.isinf(val) and self._pseudo and idx > 0:
            return float(self._sorted[idx - 1])
        return val