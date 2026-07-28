"""Tests for graph construction."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ca_gcp.core.graph import build_knn_graph


def test_build_knn_graph_correlation():
    rng = np.random.default_rng(0)
    n, t = 20, 200
    base = rng.normal(0, 1, (t, 3))
    R = rng.normal(0, 1, (t, n))
    R[:, :5] = base @ rng.uniform(-1, 1, (3, 5))
    R[:, 5:10] = base @ rng.uniform(-1, 1, (3, 5))
    df = pd.DataFrame(R, columns=[f"A{i}" for i in range(n)])

    A, nbrs, codes = build_knn_graph(df, k=4, method="correlation")
    assert A.shape == (n, n)
    assert np.allclose(A, A.T)
    assert len(codes) == n
    for i in range(n):
        assert i in nbrs[i]
        assert len(nbrs[i]) >= 1


def test_build_knn_graph_random_method():
    df = pd.DataFrame(np.random.default_rng(0).normal(0, 1, (100, 10)), columns=[f"X{i}" for i in range(10)])
    A, nbrs, _ = build_knn_graph(df, k=3, method="random", rng=np.random.default_rng(42))
    assert A.shape == (10, 10)
    assert np.allclose(A, A.T)