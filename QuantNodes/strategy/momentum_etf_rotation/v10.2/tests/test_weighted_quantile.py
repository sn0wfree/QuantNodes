"""Tests for weighted quantile."""
from __future__ import annotations

import numpy as np

from ca_gcp.core.weighted_quantile import weighted_quantile


def test_weighted_quantile_uniform_weights():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.ones(5)
    assert weighted_quantile(scores, weights, level=0.5, pseudo_count_inf=False) == 3.0


def test_weighted_quantile_skewed_weights():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = np.array([100.0, 1.0, 1.0, 1.0, 1.0])
    q = weighted_quantile(scores, weights, level=0.5, pseudo_count_inf=False)
    assert q == 1.0


def test_weighted_quantile_pseudo_count_prevents_inf_when_no_data():
    scores = np.array([1.0, 2.0])
    weights = np.array([1.0, 1.0])
    q = weighted_quantile(scores, weights, level=0.99, pseudo_count_inf=True)
    assert q == 2.0


def test_weighted_quantile_empty():
    q = weighted_quantile(np.array([]), np.array([]), level=0.95, pseudo_count_inf=True)
    assert np.isinf(q)