"""Cross-asset correlation KNN graph (Sec. 4.1).

Builds a degree-normalized adjacency from pairwise return correlations
on the training window. The graph is fixed during calibration/test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_knn_graph(
    returns_train: pd.DataFrame,
    k: int = 8,
    method: str = "correlation",
    sectors: dict[str, str] | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[int, list[int]], list[str]]:
    """Build correlation KNN graph on training window.

    Returns:
        A_norm: (N, N) degree-normalized adjacency (GCN propagation rule).
        neighbors: dict mapping target index -> list of source indices (incl. self).
        codes: list of asset codes aligned with A_norm rows/cols.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    codes = list(returns_train.columns)
    n = len(codes)
    corr = returns_train.corr().fillna(0.0).values

    if method == "correlation":
        scores = corr.copy()
    elif method == "sector":
        if sectors is None:
            raise ValueError("sectors dict required for method='sector'")
        scores = np.zeros((n, n))
        for i, ci in enumerate(codes):
            for j, cj in enumerate(codes):
                if sectors.get(ci) == sectors.get(cj):
                    scores[i, j] = 1.0
        np.fill_diagonal(scores, 1.0)
    elif method == "random":
        scores = rng.uniform(0, 1, size=(n, n))
        scores = (scores + scores.T) / 2
        np.fill_diagonal(scores, 1.0)
    else:
        raise ValueError(f"unknown method={method}")

    np.fill_diagonal(scores, -np.inf)

    neighbors: dict[int, list[int]] = {}
    for i in range(n):
        top_idx = np.argsort(-scores[i])[:k]
        top_idx = top_idx[np.isfinite(scores[i, top_idx])]
        neighbors[i] = sorted([i, *top_idx.tolist()])

    adjacency = np.zeros((n, n), dtype=float)
    for i, nbrs in neighbors.items():
        for j in nbrs:
            adjacency[i, j] = 1.0
    adjacency = (adjacency + adjacency.T) / 2
    adjacency = np.maximum(adjacency, 0.0)
    np.fill_diagonal(adjacency, 1.0)

    deg = adjacency.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(deg_inv_sqrt)
    A_norm = D_inv_sqrt @ adjacency @ D_inv_sqrt

    return A_norm, neighbors, codes