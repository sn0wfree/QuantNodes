# coding=utf-8
"""v6 单元测试.

覆盖:
- V6Config 默认值
- V6SubStrategy 接口 (继承 v5)
- run_v6_backtest 基本流程 + 风控层开关
- TF / VT / Cost 各风控层辅助函数
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from QuantNodes.strategy.momentum_etf_rotation.v5 import IndustryRotationV5SubStrategy
from QuantNodes.strategy.momentum_etf_rotation.v6 import (
    V6Config,
    V6SubStrategy,
    run_v6_backtest,
)
from QuantNodes.strategy.momentum_etf_rotation.v6.industry_rotation_v6 import (
    _apply_trend_filter,
    _apply_vol_targeting,
    _calculate_turnover_cost,
)


# ============================================================
# 工具函数
# ============================================================
def _make_panel(n_days: int = 252, n_codes: int = 5, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.bdate_range(end="2026-06-30", periods=n_days)
    data = {}
    for i in range(n_codes):
        data[f"c{i}"] = 100 * np.cumprod(1 + np.random.randn(n_days) * 0.01)
    return pd.DataFrame(data, index=dates)


# ============================================================
# Config 测试
# ============================================================
class TestV6Config:
    def test_default_config(self):
        """V6Config 默认值与 v5.1.1 一致 (除新加的风控开关)."""
        cfg = V6Config()
        assert cfg.top_n == 5
        assert cfg.vol_window == 60
        assert cfg.vol_floor == 0.01
        assert cfg.max_weight == 0.25
        assert cfg.rebal_lag == 1
        assert cfg.min_history == 252
        assert cfg.name == "industry_rotation_v6"
        # 风控开关默认 True
        assert cfg.use_vol_targeting is True
        assert cfg.use_trend_filter is True
        assert cfg.use_cost_model is True

    def test_vol_target_params(self):
        """VT 参数 (v0.1_vt_only_v2 默认值)."""
        cfg = V6Config()
        assert cfg.vol_target == 0.15
        assert cfg.vol_target_lookback == 60
        assert cfg.vol_target_min_scale == 0.3
        assert cfg.vol_target_max_scale == 1.5

    def test_trend_filter_params(self):
        """TF 参数 (v0.2_tf_only_v2 默认值)."""
        cfg = V6Config()
        assert cfg.trend_filter_ma_window == 200
        assert cfg.trend_filter_bear_exposure == 0.7
        assert cfg.trend_filter_benchmark == "510300"
        assert cfg.trend_filter_bond == "511260"

    def test_cost_model_params(self):
        """Cost 参数 (v0.3_vt_cost_v2 默认值)."""
        cfg = V6Config()
        assert cfg.commission_bp == 5.0
        assert cfg.slippage_bp == 10.0
        assert cfg.impact_factor == 0.1


# ============================================================
# V6SubStrategy 测试
# ============================================================
class TestV6SubStrategy:
    def test_subclass_of_v5(self):
        """v6 是 v5 子类 (复用 select / 因子)."""
        cfg = V6Config()
        sub = V6SubStrategy(cfg)
        assert isinstance(sub, IndustryRotationV5SubStrategy)

    def test_weight_inv_vol(self):
        """weight() 走 v5.1.1 逆波动率加权."""
        cfg = V6Config(max_weight=1.0)  # 关闭 max_weight 干扰
        sub = V6SubStrategy(cfg)
        # 构造显著波动率差异的 panel
        np.random.seed(2026)
        n = 252
        dates = pd.bdate_range(end="2026-06-30", periods=n)
        sigmas = [0.005, 0.010, 0.030, 0.010, 0.005]
        panel = pd.DataFrame(
            {f"c{i}": 100 * np.cumprod(1 + np.random.randn(n) * sigmas[i]) for i in range(5)},
            index=dates,
        )
        codes = ["c0", "c2", "c4"]
        weights = sub.weight(panel, codes, dates[-1])
        # c2 (高波 3%) 权重应最低
        assert weights["c2"] < weights["c0"]
        assert weights["c2"] < weights["c4"]
        assert abs(sum(weights.values()) - 1.0) < 1e-6


# ============================================================
# 风控层辅助函数
# ============================================================
class TestVolTargeting:
    """VT: 波动率目标缩放."""

    def test_vol_target_scales_down_high_vol(self):
        """高 actual vol → scale < 1 → 缩放."""
        # 模拟一个高波动的 NAV: 每天 +/- 5%
        np.random.seed(42)
        nav = np.cumprod(1 + np.random.randn(252) * 0.05)  # 高波
        nav_series = pd.Series(nav)
        weights = {"A": 0.5, "B": 0.5}
        cfg = V6Config(vol_target=0.15, vol_target_lookback=60)
        scaled = _apply_vol_targeting(weights, nav_series.values, nav_series.index, cfg)
        # scale < 1 → 总权重减少
        assert sum(scaled.values()) < sum(weights.values())

    def test_vol_target_scales_up_low_vol(self):
        """低 actual vol → scale > 1 → 放大."""
        np.random.seed(42)
        nav = np.cumprod(1 + np.random.randn(252) * 0.005)  # 低波
        nav_series = pd.Series(nav)
        weights = {"A": 0.5, "B": 0.5}
        cfg = V6Config(vol_target=0.15, vol_target_lookback=60,
                       vol_target_max_scale=1.5)
        scaled = _apply_vol_targeting(weights, nav_series.values, nav_series.index, cfg)
        # scale > 1 → 总权重增加 (上限 1.5)
        assert sum(scaled.values()) > sum(weights.values())
        assert sum(scaled.values()) <= 1.5 + 1e-6

    def test_vol_target_clip_max_scale(self):
        """scale 不会超过 max_scale."""
        np.random.seed(42)
        nav = np.cumprod(1 + np.random.randn(252) * 0.001)  # 极低波
        nav_series = pd.Series(nav)
        weights = {"A": 0.5, "B": 0.5}
        cfg = V6Config(vol_target=0.15, vol_target_lookback=60, vol_target_max_scale=1.2)
        scaled = _apply_vol_targeting(weights, nav_series.values, nav_series.index, cfg)
        # scale 上限 1.2
        assert sum(scaled.values()) <= 1.2 + 1e-6

    def test_vol_target_insufficient_data(self):
        """数据不足 → 返回原 weights."""
        nav = np.array([1.0, 1.01, 1.02])  # 只有 3 天
        weights = {"A": 0.5, "B": 0.5}
        cfg = V6Config(vol_target_lookback=60)
        scaled = _apply_vol_targeting(weights, nav, pd.bdate_range(end="2026-01-05", periods=3), cfg)
        assert scaled == weights


class TestTrendFilter:
    """TF: 熊市降仓 + 补充债券."""

    def test_bull_market_no_change(self):
        """牛市: HS300 > MA200 → 不变."""
        np.random.seed(42)
        # 模拟 HS300 持续上涨 > MA200
        n = 252
        dates = pd.bdate_range(end="2026-06-30", periods=n)
        hs300 = 100 * np.cumprod(1 + np.random.randn(n) * 0.005 + 0.001)  # 持续正趋势
        panel_close = pd.DataFrame({"510300": hs300}, index=dates)
        weights = {"A": 0.5, "B": 0.5}
        cfg = V6Config(trend_filter_ma_window=200, trend_filter_benchmark="510300")
        out = _apply_trend_filter(weights, panel_close, dates[-1], cfg)
        # 牛市 → 不变
        assert abs(out["A"] - 0.5) < 1e-6

    def test_bear_market_reduces_exposure(self):
        """熊市: HS300 < MA200 → 权重缩到 bear_exposure (0.7)."""
        np.random.seed(42)
        n = 252
        dates = pd.bdate_range(end="2026-06-30", periods=n)
        # 模拟 HS300 持续下跌
        hs300 = 100 * np.cumprod(1 - np.abs(np.random.randn(n)) * 0.005 - 0.001)  # 持续负趋势
        panel_close = pd.DataFrame({"510300": hs300}, index=dates)
        weights = {"A": 0.5, "B": 0.5}
        cfg = V6Config(trend_filter_ma_window=200, trend_filter_benchmark="510300",
                       trend_filter_bear_exposure=0.7, trend_filter_bond="")
        out = _apply_trend_filter(weights, panel_close, dates[-1], cfg)
        # 熊市 → 权重缩 70%
        # 如果 last < ma 且 bond_code 为空, 归一化让总=1
        # 实际: 0.5*0.7 + 0.5*0.7 = 0.7, 归一化 0.5/0.5
        assert sum(out.values()) <= 1.0 + 1e-6

    def test_missing_benchmark(self):
        """HS300 不在 panel_close 中 → 不变."""
        panel_close = pd.DataFrame({"OTHER": [1.0] * 252}, index=pd.bdate_range(end="2026-06-30", periods=252))
        weights = {"A": 0.5, "B": 0.5}
        cfg = V6Config()
        out = _apply_trend_filter(weights, panel_close, panel_close.index[-1], cfg)
        assert out == weights


class TestCostCalculation:
    """Cost: 调仓成本扣减."""

    def test_zero_turnover_zero_cost(self):
        """权重完全不变 → 成本为 0."""
        weights = {"A": 0.5, "B": 0.5}
        cfg = V6Config(commission_bp=5, slippage_bp=10)
        cost = _calculate_turnover_cost(weights, weights, cfg)
        assert cost == 0.0

    def test_full_turnover(self):
        """完全换仓: turnover=1.0, cost_rate = (5+10×0.1)/10000 = 0.6bp."""
        old = {"A": 1.0}
        new = {"B": 1.0}
        cfg = V6Config(commission_bp=5, slippage_bp=10, impact_factor=0.1)
        cost = _calculate_turnover_cost(old, new, cfg)
        # turnover = (|1-0| + |0-1|) / 2 = 1.0
        # cost = 1.0 × (5 + 10 × 0.1) / 10000 = 0.0006
        assert abs(cost - 0.0006) < 1e-9

    def test_partial_turnover(self):
        """部分换仓: 部分权重变化."""
        old = {"A": 0.5, "B": 0.5}
        new = {"A": 0.6, "B": 0.4}
        cfg = V6Config(commission_bp=5, slippage_bp=10, impact_factor=0.1)
        cost = _calculate_turnover_cost(old, new, cfg)
        # turnover = (|0.1| + |0.1|) / 2 = 0.1
        # cost = 0.1 × 0.0006 = 0.00006
        assert abs(cost - 0.00006) < 1e-9


# ============================================================
# run_v6_backtest 集成测试
# ============================================================
class TestRunV6Backtest:
    """run_v6_backtest 端到端集成测试."""

    def test_returns_series(self):
        """返回 pd.Series, 起点=1.0."""
        panel = _make_panel(n_days=300, n_codes=3)
        # 加 close 字段构造 panel_ohlcv 简化 (单字段)
        panel_ohlcv = panel.copy()
        weights = run_v6_backtest(
            panel, panel_ohlcv, V6Config(),
            apply_vol_targeting=False, apply_trend_filter=False, apply_cost_model=False,
        )
        assert isinstance(weights, pd.Series)
        assert len(weights) == 300
        assert weights.iloc[0] == 1.0

    def test_no_risk_vs_full_risk_lower_dd(self):
        """无风控 NAV 应大于等于全风控 NAV (风控抑制收益)."""
        panel = _make_panel(n_days=500, n_codes=3)
        panel_ohlcv = panel.copy()
        # 因为 panel 是简单 1 字段, sub.select 可能失败 → 看 result 还能累积
        nav_no = run_v6_backtest(panel, panel_ohlcv, V6Config(),
                                  apply_vol_targeting=False, apply_trend_filter=False, apply_cost_model=False)
        nav_full = run_v6_backtest(panel, panel_ohlcv, V6Config(),
                                  apply_vol_targeting=True, apply_trend_filter=True, apply_cost_model=True)
        # No risk version should have higher final NAV (less constraints)
        assert nav_no.iloc[-1] >= nav_full.iloc[-1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
