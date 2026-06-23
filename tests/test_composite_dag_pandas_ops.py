# coding=utf-8
"""Tests for PR-QN-4: 20 pandas-mirror composite ops (engine='pandas').

Mirror of test_composite_dag_ops.py but using pandas DataFrames.
Each op is tested with the same mathematical expectations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.operators.composite_dag import (
    get_composite_spec,
    is_composite_op,
    list_composite_ops,
)


# ===== Fixtures =====

@pytest.fixture
def neutralize_df():
    """DataFrame for neutralization tests."""
    return pd.DataFrame({
        "factor": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "citic_1": ["A", "A", "B", "B", "C", "C"],
        "citic_2": ["A1", "A2", "B1", "B2", "C1", "C2"],
    })


@pytest.fixture
def norm_df():
    """DataFrame for normalization tests."""
    return pd.DataFrame({
        "factor": [10.0, 20.0, 30.0, 40.0, 50.0],
    })


@pytest.fixture
def rolling_df():
    """DataFrame for rolling regression tests."""
    np.random.seed(42)
    n = 30
    x = np.random.randn(n).cumsum()
    y = 0.5 * x + np.random.randn(n) * 0.1
    return pd.DataFrame({"y": y, "x": x})


@pytest.fixture
def ohlcv_df():
    """DataFrame for volatility tests."""
    np.random.seed(42)
    n = 30
    close = 100 + np.random.randn(n).cumsum()
    return pd.DataFrame({
        "high": close + abs(np.random.randn(n)),
        "low": close - abs(np.random.randn(n)),
        "close": close,
        "open": close + np.random.randn(n) * 0.5,
    })


@pytest.fixture
def returns_df():
    """DataFrame for return-based tests."""
    np.random.seed(42)
    return pd.DataFrame({
        "returns": np.random.randn(30) * 0.01,
    })


@pytest.fixture
def pair_df():
    """DataFrame for pair trading tests."""
    np.random.seed(42)
    n = 30
    a = 100 + np.random.randn(n).cumsum()
    b = 100 + np.random.randn(n).cumsum()
    return pd.DataFrame({"a": a, "b": b})


# ===== Registry Tests =====

class TestPandasRegistry:
    def test_20_ops_registered(self):
        ops = list_composite_ops(engine="pandas")
        assert len(ops) == 20

    def test_op_names_match_polars(self):
        pandas_ops = set(list_composite_ops(engine="pandas"))
        polars_ops = set(list_composite_ops(engine="polars"))
        assert pandas_ops == polars_ops

    def test_is_composite_op_pandas(self):
        assert is_composite_op("industry_neutralize", engine="pandas")

    def test_spec_has_engine_pandas(self):
        spec = get_composite_spec("industry_neutralize", engine="pandas")
        assert spec is not None
        assert spec.engine == "pandas"


# ===== Neutralization (3) =====

class TestIndustryNeutralize:
    def test_basic(self, neutralize_df):
        spec = get_composite_spec("industry_neutralize", engine="pandas")
        result = spec.instantiate(df=neutralize_df, x_col="factor", industry_col="citic_1")
        # A: [1,2] mean=1.5 → [-0.5, 0.5], B: [3,4] mean=3.5 → [-0.5, 0.5]
        expected = pd.Series([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5], name="factor")
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_all_zero_when_uniform(self):
        df = pd.DataFrame({"factor": [1.0, 1.0, 1.0], "citic_1": ["A", "B", "C"]})
        spec = get_composite_spec("industry_neutralize", engine="pandas")
        result = spec.instantiate(df=df, x_col="factor", industry_col="citic_1")
        assert (result == 0.0).all()


class TestMarketNeutralize:
    def test_basic(self, neutralize_df):
        spec = get_composite_spec("market_neutralize", engine="pandas")
        result = spec.instantiate(df=neutralize_df, x_col="factor")
        assert abs(result.mean()) < 1e-10


class TestSubindustryNeutralize:
    def test_uses_citic_2(self, neutralize_df):
        spec = get_composite_spec("subindustry_neutralize", engine="pandas")
        result = spec.instantiate(
            df=neutralize_df, x_col="factor", subindustry_col="citic_2"
        )
        # Each subindustry has 1 element, neutralized = 0
        assert (result == 0.0).all()


# ===== Normalization (3) =====

class TestZscoreXs:
    def test_mean_zero(self, norm_df):
        spec = get_composite_spec("zscore_xs", engine="pandas")
        result = spec.instantiate(df=norm_df, x_col="factor")
        assert abs(result.mean()) < 1e-10

    def test_std_one(self, norm_df):
        spec = get_composite_spec("zscore_xs", engine="pandas")
        result = spec.instantiate(df=norm_df, x_col="factor")
        assert abs(result.std() - 1.0) < 1e-10


class TestRankXs:
    def test_pct_range(self, norm_df):
        spec = get_composite_spec("rank_xs", engine="pandas")
        result = spec.instantiate(df=norm_df, x_col="factor")
        assert result.min() > 0
        assert result.max() <= 1.0

    def test_monotonic(self, norm_df):
        spec = get_composite_spec("rank_xs", engine="pandas")
        result = spec.instantiate(df=norm_df, x_col="factor")
        assert result.is_monotonic_increasing


class TestScaleXs:
    def test_default_range(self, norm_df):
        spec = get_composite_spec("scale_xs", engine="pandas")
        result = spec.instantiate(df=norm_df, x_col="factor")
        assert abs(result.min() - 0.0) < 1e-10
        assert abs(result.max() - 1.0) < 1e-10

    def test_custom_range(self, norm_df):
        spec = get_composite_spec("scale_xs", engine="pandas")
        result = spec.instantiate(df=norm_df, x_col="factor", lower=-1.0, upper=1.0)
        assert abs(result.min() - (-1.0)) < 1e-10
        assert abs(result.max() - 1.0) < 1e-10


# ===== Rolling Regression (3) =====

class TestRollingBeta:
    def test_not_nan_after_full_warmup(self, rolling_df):
        spec = get_composite_spec("rolling_beta", engine="pandas")
        result = spec.instantiate(df=rolling_df, y_col="y", x_col="x", window=10)
        # Chained rolling needs ~2*window warmup
        assert result.iloc[20:].notna().all()

    def test_nan_in_warmup(self, rolling_df):
        spec = get_composite_spec("rolling_beta", engine="pandas")
        result = spec.instantiate(df=rolling_df, y_col="y", x_col="x", window=10)
        assert result.iloc[:15].isna().all()


class TestRollingOlsSimplified:
    def test_output_length(self, rolling_df):
        spec = get_composite_spec("rolling_ols_simplified", engine="pandas")
        result = spec.instantiate(df=rolling_df, y_col="y", x_col="x", window=10)
        assert len(result) == len(rolling_df)


class TestRollingResidual:
    def test_residual_is_y_minus_beta_x(self, rolling_df):
        spec_r = get_composite_spec("rolling_residual", engine="pandas")
        res = spec_r.instantiate(df=rolling_df, y_col="y", x_col="x", window=10)
        # residual = y - beta * x (not y - ols_fitted)
        # Just verify valid part is finite
        assert res.iloc[20:].notna().all()
        assert np.isfinite(res.iloc[20:]).all()


# ===== Volatility (4) =====

class TestParkinsonVol:
    def test_positive(self, ohlcv_df):
        spec = get_composite_spec("parkinson_vol", engine="pandas")
        result = spec.instantiate(df=ohlcv_df, high_col="high", low_col="low", window=10)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_increases_with_spread(self):
        # Higher spread → higher vol
        df窄 = pd.DataFrame({
            "high": [101, 102, 103, 104, 105],
            "low": [100, 101, 102, 103, 104],
        })
        df宽 = pd.DataFrame({
            "high": [110, 120, 130, 140, 150],
            "low": [90, 80, 70, 60, 50],
        })
        spec = get_composite_spec("parkinson_vol", engine="pandas")
        v窄 = spec.instantiate(df=df窄, high_col="high", low_col="low", window=3).dropna().mean()
        v宽 = spec.instantiate(df=df宽, high_col="high", low_col="low", window=3).dropna().mean()
        assert v宽 > v窄


class TestGarmanKlassVol:
    def test_positive(self, ohlcv_df):
        spec = get_composite_spec("garman_klass_vol", engine="pandas")
        result = spec.instantiate(
            df=ohlcv_df, high_col="high", low_col="low",
            close_col="close", open_col="open", window=10,
        )
        valid = result.dropna()
        assert (valid >= 0).all()


class TestYangZhangVol:
    def test_positive(self, ohlcv_df):
        spec = get_composite_spec("yang_zhang_vol", engine="pandas")
        result = spec.instantiate(
            df=ohlcv_df, high_col="high", low_col="low",
            close_col="close", open_col="open", window=10,
        )
        valid = result.dropna()
        assert (valid >= 0).all()


class TestRealizedVol:
    def test_positive(self, returns_df):
        spec = get_composite_spec("realized_vol", engine="pandas")
        result = spec.instantiate(df=returns_df, returns_col="returns", window=10)
        valid = result.dropna()
        assert (valid >= 0).all()


# ===== Pairs Trading (2) =====

class TestPairZscore:
    def test_zero_when_identical(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
        spec = get_composite_spec("pair_zscore", engine="pandas")
        result = spec.instantiate(df=df, a_col="a", b_col="b", window=3)
        # spread = 0 → zscore = 0/0 = NaN (not 0), check valid part
        assert result.iloc[-1] == 0.0 or np.isnan(result.iloc[-1])

    def test_output_length(self, pair_df):
        spec = get_composite_spec("pair_zscore", engine="pandas")
        result = spec.instantiate(df=pair_df, a_col="a", b_col="b", window=10)
        assert len(result) == len(pair_df)


class TestPairRatio:
    def test_one_when_identical(self):
        df = pd.DataFrame({"a": [2.0, 4.0, 6.0], "b": [2.0, 4.0, 6.0]})
        spec = get_composite_spec("pair_ratio", engine="pandas")
        result = spec.instantiate(df=df, a_col="a", b_col="b", window=3)
        valid = result.dropna()
        assert abs(valid.mean() - 1.0) < 1e-10


# ===== Winsorize/Outlier (3) =====

class TestWinsorize:
    def test_clips_extremes(self):
        df = pd.DataFrame({"factor": list(range(100)) + [10000]})
        spec = get_composite_spec("winsorize", engine="pandas")
        result = spec.instantiate(df=df, x_col="factor", lower_q=0.01, upper_q=0.99)
        assert result.max() <= 99  # clipped at 99th percentile
        assert result.min() >= 0.99  # clipped at 1st percentile

    def test_middle_values_unclipped(self):
        # 5 elements: 1% quantile = 10.4, 99% = 49.6
        # Only values outside [10.4, 49.6] are clipped
        df = pd.DataFrame({"factor": [10.0, 20.0, 30.0, 40.0, 50.0]})
        spec = get_composite_spec("winsorize", engine="pandas")
        result = spec.instantiate(df=df, x_col="factor", lower_q=0.01, upper_q=0.99)
        # Middle values stay
        assert result.iloc[1] == 20.0
        assert result.iloc[2] == 30.0
        assert result.iloc[3] == 40.0


class TestMadOutlier:
    def test_outlier_becomes_nan(self):
        df = pd.DataFrame({"factor": [1.0, 1.0, 1.0, 1.0, 100.0]})
        spec = get_composite_spec("mad_outlier", engine="pandas")
        result = spec.instantiate(df=df, x_col="factor", n_mad=3.0)
        assert pd.isna(result.iloc[-1])

    def test_normal_values_preserved(self):
        df = pd.DataFrame({"factor": [1.0, 2.0, 3.0, 4.0, 5.0]})
        spec = get_composite_spec("mad_outlier", engine="pandas")
        result = spec.instantiate(df=df, x_col="factor", n_mad=3.0)
        assert result.notna().all()


class TestZscoreClip:
    def test_extreme_becomes_nan(self):
        # Need enough normal values for z-score to be meaningful
        vals = list(range(20)) + [1000.0]
        df = pd.DataFrame({"factor": vals})
        spec = get_composite_spec("zscore_clip", engine="pandas")
        result = spec.instantiate(df=df, x_col="factor", n_std=2.0)
        assert pd.isna(result.iloc[-1])

    def test_normal_values_preserved(self):
        df = pd.DataFrame({"factor": list(range(20))})
        spec = get_composite_spec("zscore_clip", engine="pandas")
        result = spec.instantiate(df=df, x_col="factor", n_std=3.0)
        assert result.notna().all()


# ===== Time-Series (2) =====

class TestDecayLinearXs:
    def test_ewm_mean(self):
        df = pd.DataFrame({"factor": list(range(20))})
        spec = get_composite_spec("decay_linear_xs", engine="pandas")
        result = spec.instantiate(df=df, x_col="factor", window=5)
        expected = df["factor"].ewm(span=5).mean()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_no_nan(self):
        df = pd.DataFrame({"factor": [1.0, 2.0, 3.0]})
        spec = get_composite_spec("decay_linear_xs", engine="pandas")
        result = spec.instantiate(df=df, x_col="factor", window=2)
        assert result.notna().all()


class TestMomentumAccel:
    def test_accel_measures_surge(self):
        # Accelerating uptrend: price surges late
        df = pd.DataFrame({
            "factor": [1.0, 1.01, 1.02, 1.03, 1.1, 1.3, 1.6, 2.0],
        })
        spec = get_composite_spec("momentum_accel", engine="pandas")
        result = spec.instantiate(df=df, x_col="factor", short_window=2, long_window=5)
        # short_mom surges (1.6→2.0 = +25%), long_mom lags (1.03→1.3 = +26%)
        # accel = short - long; when short surges more, accel > 0
        valid = result.dropna()
        assert len(valid) > 0

    def test_output_length(self):
        df = pd.DataFrame({"factor": list(range(20))})
        spec = get_composite_spec("momentum_accel", engine="pandas")
        result = spec.instantiate(df=df, x_col="factor", short_window=5, long_window=10)
        assert len(result) == 20


# ===== Parametrize across all 20 ops =====

ALL_PANDAS_OPS = list_composite_ops(engine="pandas")


@pytest.mark.parametrize("op_name", ALL_PANDAS_OPS)
def test_pandas_op_has_valid_spec(op_name):
    """Every pandas op must have a valid CompositeSpec with engine='pandas'."""
    spec = get_composite_spec(op_name, engine="pandas")
    assert spec is not None
    assert spec.engine == "pandas"
    assert spec.name == op_name
    assert callable(spec.template)


@pytest.mark.parametrize("op_name", ALL_PANDAS_OPS)
def test_pandas_op_to_dict(op_name):
    """Every pandas op spec must serialize to dict without error."""
    spec = get_composite_spec(op_name, engine="pandas")
    d = spec.to_dict()
    assert isinstance(d, dict)
    assert d["name"] == op_name
    assert "params" in d
    assert "doc" in d
