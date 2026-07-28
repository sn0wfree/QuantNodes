"""Tests for the fast precomputed weighted quantile."""
from __future__ import annotations

import numpy as np

from ca_gcp.core.weighted_quantile_fast import PrecomputedWeightedQuantile
from ca_gcp.core.weighted_quantile import weighted_quantile


def test_matches_slow_uniform_weights():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.ones(5)
    cache = PrecomputedWeightedQuantile(scores, weights, pseudo_count_inf=False)
    assert abs(cache.query(0.5) - weighted_quantile(scores, weights, level=0.5, pseudo_count_inf=False)) < 1e-9


def test_matches_slow_pseudo_count_inf():
    scores = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 1.0, 1.0])
    cache = PrecomputedWeightedQuantile(scores, weights, pseudo_count_inf=True)
    slow = weighted_quantile(scores, weights, level=0.95, pseudo_count_inf=True)
    fast = cache.query(0.95)
    assert abs(fast - slow) < 1e-9 or (fast == slow)


def test_skewed_weights_match():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([100.0, 1.0, 1.0, 1.0, 1.0])
    cache = PrecomputedWeightedQuantile(scores, weights, pseudo_count_inf=False)
    assert cache.query(0.5) == 1.0


def test_empty():
    cache = PrecomputedWeightedQuantile(np.array([]), np.array([]), pseudo_count_inf=True)
    assert np.isinf(cache.query(0.95))


def test_repeated_queries_consistent():
    scores = np.random.default_rng(0).uniform(0, 5, 1000)
    weights = np.random.default_rng(1).uniform(0.5, 1.5, 1000)
    cache = PrecomputedWeightedQuantile(scores, weights, pseudo_count_inf=False)
    v1 = cache.query(0.95)
    v2 = cache.query(0.95)
    assert v1 == v2


def test_speedup_smoke():
    """Compare construction cost vs repeated queries."""
    import time
    scores = np.random.default_rng(0).uniform(0, 5, 5000)
    weights = np.random.default_rng(1).uniform(0.1, 1.0, 5000)

    t0 = time.time()
    for _ in range(100):
        weighted_quantile(scores, weights, level=0.95, pseudo_count_inf=False)
    t_slow = time.time() - t0

    cache = PrecomputedWeightedQuantile(scores, weights, pseudo_count_inf=False)
    t0 = time.time()
    for _ in range(100):
        cache.query(0.95)
    t_fast = time.time() - t0

    assert t_fast < t_slow, f"Expected cache faster than slow ({t_fast:.3f}s vs {t_slow:.3f}s)"