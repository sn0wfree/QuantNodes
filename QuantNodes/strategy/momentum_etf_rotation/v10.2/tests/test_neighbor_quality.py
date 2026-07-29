"""Tests for neighbor-quality scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import CAGCPConfig, CAGCPipeline
from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import (
    NeighborQuality,
    compute_neighbor_quality,
    quality_dataframe,
    recommend_borrow_strategy,
)


@pytest.fixture
def fitted_pipeline() -> CAGCPipeline:
    rng = np.random.default_rng(0)
    n, t = 12, 200
    base = rng.normal(0, 1, (t, 3))
    R = rng.normal(0, 1, (t, n))
    R[:, :4] = base @ rng.uniform(-1, 1, (3, 4))
    R[:, 4:8] = base @ rng.uniform(-1, 1, (3, 4))
    R[:, 8:] = rng.normal(0, 1, (t, n - 8))
    df = pd.DataFrame(R, columns=[f"E{i}" for i in range(n)])
    df.index = pd.date_range("2020-01-01", periods=t, freq="B")
    pipe = CAGCPipeline(CAGCPConfig(k=4))
    pipe.fit(df)
    return pipe


def test_compute_neighbor_quality_returns_dataclass(fitted_pipeline):
    nq = compute_neighbor_quality(fitted_pipeline, 0, calib_days=252)
    assert isinstance(nq, NeighborQuality)
    assert nq.target_idx == 0
    assert nq.n_neighbors > 0
    assert 0.0 <= nq.mean_corr <= 1.0
    assert nq.borrow_recommendation in {"strong", "moderate", "weak"}


def test_weighted_corr_sum_increases_with_sharpness(fitted_pipeline):
    base = compute_neighbor_quality(fitted_pipeline, 0)
    fitted_pipeline.config.sharpness_p = 2.0
    sharpened = compute_neighbor_quality(fitted_pipeline, 0)
    assert sharpened.weighted_corr_sum <= base.weighted_corr_sum + 1e-9


def test_recommend_borrow_strategy_all_assets(fitted_pipeline):
    recs = recommend_borrow_strategy(fitted_pipeline)
    assert set(recs.keys()) == set(fitted_pipeline.codes)
    assert all(r in {"strong", "moderate", "weak"} for r in recs.values())


def test_quality_dataframe_shape(fitted_pipeline):
    df = quality_dataframe(fitted_pipeline)
    assert df.shape[0] == len(fitted_pipeline.codes)
    expected_cols = {
        "n_neighbors",
        "weighted_corr_sum",
        "mean_corr",
        "effective_sample_size",
        "borrow_recommendation",
    }
    assert expected_cols.issubset(set(df.columns))


def test_strong_threshold_filters_correctly(fitted_pipeline):
    df = quality_dataframe(fitted_pipeline)
    strong = set(df[df["borrow_recommendation"] == "strong"].index)
    moderate = set(df[df["borrow_recommendation"] == "moderate"].index)
    weak = set(df[df["borrow_recommendation"] == "weak"].index)
    assert strong | moderate | weak == set(fitted_pipeline.codes)
    assert strong.isdisjoint(moderate)
    assert strong.isdisjoint(weak)
    assert moderate.isdisjoint(weak)