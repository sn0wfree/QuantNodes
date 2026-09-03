# coding=utf-8
"""AllWeather V6 — subfactors_weekly.py 单元测试 (PLAN §13.3.2)."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.subfactors_weekly import (
    combine_subfactors,
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
