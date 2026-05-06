# coding=utf-8
"""QuantNodes.operators.time_series 单元测试"""
import numpy as np
import polars as pl
import pytest

from QuantNodes.operators.time_series import TimeSeriesOperators


@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "b": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
    })


class TestTimeSeriesRolling:
    def test_ts_mean(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_mean("a", window=3))
        values = result["a"].to_list()
        assert values[2] == pytest.approx(2.0)

    def test_ts_mean_min_periods(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_mean("a", window=5, min_periods=3))
        values = result["a"].to_list()
        assert values[3] is not None

    def test_ts_std(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_std("a", window=3))
        values = result["a"].to_list()
        assert values[2] == pytest.approx(np.std([1.0, 2.0, 3.0], ddof=1))

    def test_ts_max(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_max("a", window=3))
        values = result["a"].to_list()
        assert values[2] == 3.0

    def test_ts_min(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_min("a", window=3))
        values = result["a"].to_list()
        assert values[2] == 1.0

    def test_ts_sum(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_sum("a", window=3))
        values = result["a"].to_list()
        assert values[2] == pytest.approx(6.0)

    def test_ts_prod(self, sample_df):
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        result = df.select(TimeSeriesOperators.ts_prod("a", window=3))
        values = result["a"].to_list()
        assert values[2] == pytest.approx(6.0)


class TestTimeSeriesRollingMedian:
    def test_ts_median(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_median("a", window=3))
        values = result["a"].to_list()
        assert values[2] == 2.0


class TestTimeSeriesRollingCorr:
    def test_ts_corr(self, sample_df):
        result = sample_df.select(
            TimeSeriesOperators.ts_corr("a", "b", window=5)
        )
        values = result["a"].to_list()
        assert all(v is None or -1 <= v <= 1 for v in values if v is not None)

    def test_ts_corr_with_expr(self, sample_df):
        result = sample_df.select(
            TimeSeriesOperators.ts_corr(pl.col("a"), pl.col("b"), window=5)
        )
        values = result["a"].to_list()
        assert len(values) == 10


class TestTimeSeriesRollingCov:
    def test_ts_cov(self, sample_df):
        result = sample_df.select(
            TimeSeriesOperators.ts_cov("a", "b", window=5)
        )
        values = result["a"].to_list()
        assert len(values) == 10


class TestTimeSeriesRollingRank:
    def test_ts_rank(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_rank("a", window=3))
        values = result["a"].to_list()
        assert values[2] is not None


class TestTimeSeriesDelta:
    def test_ts_delta(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_delta("a", periods=1))
        values = result["a"].to_list()
        assert values[1] == 1.0

    def test_ts_delta_periods_2(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_delta("a", periods=2))
        values = result["a"].to_list()
        assert values[2] == 2.0


class TestTimeSeriesPctChange:
    def test_ts_pct_change(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_pct_change("a", periods=1))
        values = result["a"].to_list()
        assert values[1] == pytest.approx(1.0)


class TestTimeSeriesLagShift:
    def test_ts_lag(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_lag("a", periods=1))
        values = result["a"].to_list()
        assert values[1] == 1.0
        assert values[0] is None

    def test_ts_lag_2(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_lag("a", periods=2))
        values = result["a"].to_list()
        assert values[2] == 1.0
        assert values[0] is None
        assert values[1] is None

    def test_ts_shift_alias(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_shift("a", periods=1))
        values = result["a"].to_list()
        assert values[1] == 1.0


class TestTimeSeriesLead:
    def test_ts_lead(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_lead("a", periods=1))
        values = result["a"].to_list()
        assert values[0] == 2.0
        assert values[9] is None

    def test_ts_lead_2(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_lead("a", periods=2))
        values = result["a"].to_list()
        assert values[0] == 3.0
        assert values[8] is None
        assert values[9] is None


class TestTimeSeriesEwm:
    def test_ewm_mean(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ewm_mean("a", alpha=0.5))
        values = result["a"].to_list()
        assert len(values) == 10
        assert values[-1] is not None

    def test_ewm_std(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ewm_std("a", alpha=0.5))
        values = result["a"].to_list()
        assert len(values) == 10

    def test_ewm_corr(self, sample_df):
        result = sample_df.select(
            TimeSeriesOperators.ewm_corr("a", "b", alpha=0.5)
        )
        values = result["a"].to_list()
        assert len(values) == 10


class TestTimeSeriesStringInput:
    def test_ts_mean_str(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_mean("a", window=3))
        assert len(result) == 10

    def test_ts_std_str(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_std("a", window=3))
        assert len(result) == 10

    def test_ts_delta_str(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_delta("a", periods=1))
        assert len(result) == 10

    def test_ts_lag_str(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ts_lag("a", periods=1))
        assert len(result) == 10

    def test_ewm_mean_str(self, sample_df):
        result = sample_df.select(TimeSeriesOperators.ewm_mean("a", alpha=0.5))
        assert len(result) == 10
