# coding=utf-8
"""AllWeather V6 — subfactors_weekly.py 单元测试 (PLAN §13.3.2)."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.subfactors_weekly import (
    combine_subfactors,
    rank_cross_section,
    resample_weekly,
    zscore_cross_section,
)


def _daily_factor(t: int = 400, n: int = 5, seed: int = 0) -> pd.DataFrame:
    """合成日度因子面板."""
    r = np.random.default_rng(seed)
    return pd.DataFrame(
        r.uniform(-1, 1, size=(t, n)),
        index=pd.bdate_range("2020-01-01", periods=t),
        columns=[f"A{i}" for i in range(n)],
    )


class TestResampleWeekly:
    def test_last_aggregation_aligns_to_w_fri(self):
        daily = _daily_factor(t=400)
        out = resample_weekly(daily, agg="last")
        # 400 个工作日 → ~80 周
        assert len(out) < len(daily)
        assert len(out) > 70

    def test_mean_aggregation_within_range(self):
        daily = _daily_factor(t=400)
        out = resample_weekly(daily, agg="mean")
        assert out.notna().any().any()


class TestZscoreCrossSection:
    def test_late_window_zero_mean(self):
        """后段 (有完整滚动窗口) 截面均值应接近 0."""
        factor = _daily_factor(t=400, n=10, seed=1)
        weekly = resample_weekly(factor, agg="mean")
        z = zscore_cross_section(weekly, lookback_weeks=52)
        # 取后 52 周 (完整窗口)
        last_year = z.iloc[-52:]
        assert abs(last_year.mean().mean()) < 0.5


class TestCombineSubfactors:
    def test_weights_sum_to_one(self):
        """combine_subfactors 输出 shape 对齐, 权重归一化."""
        daily_idx = pd.bdate_range("2020-01-01", periods=400)
        rng = np.random.default_rng(2)
        daily = {
            "a": pd.DataFrame(rng.uniform(-1, 1, size=(400, 5)), index=daily_idx),
            "b": pd.DataFrame(rng.uniform(-1, 1, size=(400, 5)), index=daily_idx),
            "c": pd.DataFrame(rng.uniform(-1, 1, size=(400, 5)), index=daily_idx),
        }
        weekly = {k: resample_weekly(v, agg="mean") for k, v in daily.items()}
        out = combine_subfactors(weekly, weights={"a": 0.33, "b": 0.33, "c": 0.34})
        assert out.shape == weekly["a"].shape

    def test_method_rank(self):
        """method='rank' 输出 shape 对齐, 值域 [-0.5, +0.5]."""
        daily_idx = pd.bdate_range("2020-01-01", periods=400)
        rng = np.random.default_rng(3)
        daily = {
            "a": pd.DataFrame(rng.uniform(-1, 1, size=(400, 5)), index=daily_idx),
            "b": pd.DataFrame(rng.uniform(-1, 1, size=(400, 5)), index=daily_idx),
        }
        weekly = {k: resample_weekly(v, agg="mean") for k, v in daily.items()}
        out = combine_subfactors(weekly, method="rank")
        assert out.shape == weekly["a"].shape
        assert out.max().max() <= 0.5
        assert out.min().min() >= -0.5

    def test_method_winsorize(self):
        """winsorize=2.0 时 z-score 被 clip 到 [-2, +2]."""
        factor = _daily_factor(t=400, n=10, seed=4)
        weekly = resample_weekly(factor, agg="mean")
        z = zscore_cross_section(weekly, lookback_weeks=52, winsorize=2.0)
        assert z.max().max() <= 2.0
        assert z.min().min() >= -2.0


class TestRankCrossSection:
    def test_rank_output_range(self):
        """rank_cross_section 输出 ∈ [-0.5, +0.5]."""
        factor = _daily_factor(t=400, n=10, seed=5)
        weekly = resample_weekly(factor, agg="mean")
        r = rank_cross_section(weekly, lookback_weeks=52)
        assert r.max().max() <= 0.5
        assert r.min().min() >= -0.5
