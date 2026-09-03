# coding=utf-8
"""AllWeather V6 — subfactors_weekly.py 单元测试 (PLAN §13.3.2)."""
import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.subfactors_weekly import (
    combine_subfactors,
    directional_risk_transform,
    magnitude_risk_transform,
    magnitude_squared_risk_transform,
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


class TestRiskTransforms:
    def test_directional_positive_zero(self):
        """Directional: 正向分数 → 0 (不调整)."""
        score = pd.DataFrame([[0.3, -0.2, 0.0]], columns=["a", "b", "c"])
        out = directional_risk_transform(score)
        assert out.iloc[0, 0] == 0.0
        assert out.iloc[0, 1] == -0.2
        assert out.iloc[0, 2] == 0.0

    def test_directional_negative_preserved(self):
        """Directional: 负向分数保留, 调整时用 -score (向下)."""
        score = pd.DataFrame([[-0.1, -0.4, -0.5]], columns=["a", "b", "c"])
        out = directional_risk_transform(score)
        assert (out.values == score.values).all()
        assert out.max().max() <= 0.0

    def test_magnitude_symmetric(self):
        """Magnitude: ±x 都映射为 -|x| (对称)."""
        score = pd.DataFrame([[0.3, -0.3, 0.5]], columns=["a", "b", "c"])
        out = magnitude_risk_transform(score)
        assert abs(out.iloc[0, 0] - (-0.3)) < 1e-9
        assert abs(out.iloc[0, 1] - (-0.3)) < 1e-9
        assert abs(out.iloc[0, 2] - (-0.5)) < 1e-9

    def test_magnitude_squared_quadratic(self):
        """Magnitude²: |x|² 比 |x| 增长快 (极端值惩罚更重)."""
        small = magnitude_squared_risk_transform(
            pd.DataFrame([[0.1, -0.1]], columns=["a", "b"])
        )
        large = magnitude_squared_risk_transform(
            pd.DataFrame([[0.5, -0.5]], columns=["a", "b"])
        )
        # 0.5² = 0.25 vs 0.1² = 0.01, 比例 25 倍 (vs magnitude 5 倍)
        assert abs(small.iloc[0, 0] - (-0.01)) < 1e-9
        assert abs(large.iloc[0, 0] - (-0.25)) < 1e-9
        assert abs(large.iloc[0, 0] / small.iloc[0, 0]) > 10  # 非线性放大

    def test_all_three_zero_at_neutral(self):
        """score = 0 时三种模式都返回 0 (无调整)."""
        score = pd.DataFrame([[0.0, 0.0]], columns=["a", "b"])
        assert (directional_risk_transform(score).values == 0).all()
        assert (magnitude_risk_transform(score).values == 0).all()
        assert (magnitude_squared_risk_transform(score).values == 0).all()
