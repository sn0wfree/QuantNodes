"""Tests for theoretical coverage gap bound via TV distance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import CAGCPConfig, CAGCPipeline
from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import (
    compare_bound_to_empirical,
    theoretical_coverage_bound,
    total_variation_distance_ecdf,
)


def test_tv_distance_identical_distributions_zero():
    rng = np.random.default_rng(0)
    p = rng.normal(0, 1, 1000)
    q = p.copy()
    assert total_variation_distance_ecdf(p, q) == 0.0


def test_tv_distance_disjoint_distributions_one():
    p = np.zeros(100)
    q = np.ones(100)
    tv = total_variation_distance_ecdf(p, q)
    assert tv == pytest.approx(1.0, abs=1e-6)


def test_tv_distance_overlapping_distributions_in_unit_interval():
    rng = np.random.default_rng(1)
    p = rng.normal(0, 1, 1000)
    q = rng.normal(0.5, 1, 1000)
    tv = total_variation_distance_ecdf(p, q)
    assert 0.0 <= tv <= 1.0


def test_tv_distance_shift_increases():
    rng = np.random.default_rng(2)
    p = rng.normal(0, 1, 500)
    q_close = rng.normal(0.3, 1, 500)
    q_far = rng.normal(1.0, 1, 500)
    tv_close = total_variation_distance_ecdf(p, q_close)
    tv_far = total_variation_distance_ecdf(p, q_far)
    assert tv_far > tv_close


def test_tv_distance_handles_empty():
    assert total_variation_distance_ecdf(np.array([]), np.array([1, 2, 3])) == 1.0
    assert total_variation_distance_ecdf(np.array([1, 2, 3]), np.array([])) == 1.0


@pytest.fixture
def fitted_pipeline():
    rng = np.random.default_rng(3)
    n, t = 10, 250
    base = rng.normal(0, 1, (t, 3))
    R = rng.normal(0, 1, (t, n))
    R[:, :4] = base @ rng.uniform(-1, 1, (3, 4))
    R[:, 4:8] = base @ rng.uniform(-1, 1, (3, 4))
    df = pd.DataFrame(R, columns=[f"E{i}" for i in range(n)])
    df.index = pd.date_range("2020-01-01", periods=t, freq="B")
    pipe = CAGCPipeline(CAGCPConfig(k=3))
    pipe.fit(df)
    return pipe


def test_theoretical_coverage_bound_returns_dataframe(fitted_pipeline):
    rng = np.random.default_rng(4)
    scores_calib = pd.DataFrame(
        rng.normal(0, 1, (250, len(fitted_pipeline.codes))),
        columns=fitted_pipeline.codes,
    )
    bounds = theoretical_coverage_bound(fitted_pipeline, scores_calib)
    assert len(bounds) == len(fitted_pipeline.codes)
    assert "bound" in bounds.columns
    assert "max_tv" in bounds.columns
    assert "weighted_tv_sum" in bounds.columns
    assert (bounds["bound"] >= 0).all()
    assert (bounds["bound"] <= 1).all()


def test_theoretical_coverage_bound_high_correlation_gives_low_bound(fitted_pipeline):
    rng = np.random.default_rng(5)
    n_codes = len(fitted_pipeline.codes)

    target = rng.normal(0, 1, 250)
    high_scores = pd.DataFrame(
        {c: target + rng.normal(0, 0.01, 250) for c in fitted_pipeline.codes},
        columns=fitted_pipeline.codes,
    )
    low_scores = pd.DataFrame(
        {c: rng.normal(0, 1, 250) for c in fitted_pipeline.codes},
        columns=fitted_pipeline.codes,
    )

    high_bounds = theoretical_coverage_bound(fitted_pipeline, high_scores)
    low_bounds = theoretical_coverage_bound(fitted_pipeline, low_scores)

    assert high_bounds["weighted_tv_sum"].mean() < low_bounds["weighted_tv_sum"].mean()


def test_compare_bound_to_empirical_returns_ratio():
    bounds = pd.DataFrame(
        {"bound": [0.10, 0.10, 0.10]},
        index=pd.Index(["A", "B", "C"], name="code"),
    )
    gaps = pd.Series([0.02, 0.05, 0.20], index=pd.Index(["A", "B", "C"], name="code"))
    df = compare_bound_to_empirical(bounds, gaps)
    assert len(df) == 3
    assert "ratio" in df.columns
    assert "bound_satisfied" in df.columns
    assert df.loc["A", "bound_satisfied"]
    assert df.loc["B", "bound_satisfied"]
    assert not df.loc["C", "bound_satisfied"]