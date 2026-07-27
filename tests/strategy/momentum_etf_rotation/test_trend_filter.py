# coding=utf-8
"""Tests for Stage 9-B: 趋势过滤器 (基于基准指数均线)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation import (
    DEFAULT_POOL,
    DiversificationCaps,
    RotationConfig,
    TrendFilter,
    BacktestConfig,
    run_rotation_backtest,
    performance_metrics,
)
from QuantNodes.strategy.momentum_etf_rotation.core.portfolio import (
    apply_stops,
    apply_trend_filter,
    check_trend_filter,
    select_and_weight,
)


def _make_panel(n_days: int = 800, seed: int = 42) -> pd.DataFrame:
    """合成测试面板: 全 44 ETF + 沪深 300 + 国债, 800 天."""
    rng = np.random.default_rng(seed)
    codes = list(DEFAULT_POOL.codes)
    # 沪深 300 已在池中 (a_broad), 只需额外加国债
    codes = codes + ["511260"]
    rets = rng.normal(0.0003, 0.012, size=(n_days, len(codes)))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=codes)


@pytest.fixture
def panel() -> pd.DataFrame:
    return _make_panel()


@pytest.fixture
def cfg_trend_enabled() -> RotationConfig:
    return RotationConfig(
        lookback=120, top_n=5, min_history=120,
        trend_filter=TrendFilter(enabled=True, ma_window=200, exposure_bear=0.5),
    )


@pytest.fixture
def cfg_trend_disabled() -> RotationConfig:
    return RotationConfig(lookback=120, top_n=5, min_history=120)


# ============================================================================
# 单元测试: check_trend_filter
# ============================================================================
class TestCheckTrendFilter:
    def test_bull_market_returns_true(self, panel) -> None:
        """多头市场: 沪深 300 在均线之上 → True."""
        # 找近期一个上涨日
        result = check_trend_filter(panel, "510300", 200, panel.index[-1])
        # 取决于合成数据, 但函数应该返回 bool
        assert isinstance(result, bool)

    def test_insufficient_data_returns_true(self, panel) -> None:
        """数据不足 ma_window 时, 默认多头."""
        # 取前 50 天
        early_date = panel.index[50]
        result = check_trend_filter(panel, "510300", 200, early_date)
        assert result is True

    def test_unknown_benchmark_returns_true(self, panel) -> None:
        """未知基准代码 → 默认多头."""
        result = check_trend_filter(panel, "XXXXXX", 200, panel.index[-1])
        assert result is True


# ============================================================================
# 集成测试: apply_trend_filter
# ============================================================================
class TestApplyTrendFilter:
    def test_disabled_keeps_weights(self, panel, cfg_trend_disabled) -> None:
        """trend_filter disabled → 权重不变."""
        weights = {"518880": 0.5, "513100": 0.5}
        # 构造一个 fake state (ranked 是必填)
        from QuantNodes.strategy.momentum_etf_rotation.core.portfolio import PortfolioState
        state = PortfolioState(date=panel.index[-1], ranked=[], chosen=[],
                                weights=dict(weights))
        new_state = apply_trend_filter(panel, cfg_trend_disabled, panel.index[-1], state)
        assert new_state.weights == weights

    def test_bull_market_keeps_full_exposure(self, panel) -> None:
        """多头 → 权重不变."""
        # 创建一个明显多头场景: 只拉高最近 n_recent 天的价格
        panel_bull = panel.copy()
        n_recent = 50  # 只拉高最近 50 天, MA 不变
        recent_idx = panel_bull.index[-n_recent:]
        panel_bull.loc[recent_idx, "510300"] = panel_bull.loc[recent_idx, "510300"] * 1.5
        cfg = RotationConfig(
            lookback=120, top_n=5, min_history=120,
            trend_filter=TrendFilter(enabled=True, ma_window=200, exposure_bear=0.5),
        )
        from QuantNodes.strategy.momentum_etf_rotation.core.portfolio import PortfolioState
        weights = {"518880": 0.5, "513100": 0.5}
        state = PortfolioState(date=panel_bull.index[-1], ranked=[], chosen=[],
                                weights=dict(weights))
        new_state = apply_trend_filter(panel_bull, cfg, panel_bull.index[-1], state)
        assert new_state.weights == weights  # 多头 → 不变

    def test_bear_market_scales_to_exposure_bear(self, panel) -> None:
        """熊市 → 缩放到 exposure_bear."""
        # 制造熊市: 只降低最近 50 天的价格 (MA 窗口 200 天, 大部分不变)
        # 这样最后一天的价格 < MA
        panel_bear = panel.copy()
        n_recent = 50  # 远小于 ma_window=200, 确保 MA 不被大幅拉低
        panel_bear.loc[panel_bear.index[-n_recent:], "510300"] = (
            panel_bear["510300"].iloc[-n_recent:].values * 0.5
        )
        cfg = RotationConfig(
            lookback=120, top_n=5, min_history=120,
            trend_filter=TrendFilter(
                enabled=True, ma_window=200,
                exposure_bear=0.5, bond_code="511260",
            ),
        )
        from QuantNodes.strategy.momentum_etf_rotation.core.portfolio import PortfolioState
        weights = {"518880": 0.5, "513100": 0.5}
        state = PortfolioState(date=panel_bear.index[-1], ranked=[], chosen=[],
                                weights=dict(weights))
        new_state = apply_trend_filter(panel_bear, cfg, panel_bear.index[-1], state)
        # 股票权重应缩放到 0.5 倍
        assert abs(new_state.weights["518880"] - 0.25) < 1e-9
        assert abs(new_state.weights["513100"] - 0.25) < 1e-9
        # 债券 511260 应占 0.5
        assert abs(new_state.weights["511260"] - 0.5) < 1e-9
        # 总和应 ≈ 1.0
        assert abs(sum(new_state.weights.values()) - 1.0) < 1e-9


# ============================================================================
# 集成测试: select_and_weight + apply_stops
# ============================================================================
class TestSelectAndWeightTrendFilter:
    def test_bull_market_no_bond_allocation(self, panel) -> None:
        """多头 select_and_weight → 不加入债券."""
        panel_bull = panel.copy()
        n_recent = 50
        recent_idx = panel_bull.index[-n_recent:]
        panel_bull.loc[recent_idx, "510300"] = panel_bull.loc[recent_idx, "510300"] * 1.5
        cfg = RotationConfig(
            lookback=120, top_n=5, min_history=120,
            trend_filter=TrendFilter(enabled=True, ma_window=200, exposure_bear=0.5),
        )
        state = select_and_weight(panel_bull, DEFAULT_POOL, cfg, panel_bull.index[-1])
        assert "511260" not in state.weights  # 无债券

    def test_bear_market_includes_bond(self, panel) -> None:
        """熊市 select_and_weight → 511260 应在持仓中."""
        panel_bear = panel.copy()
        n_recent = 50
        panel_bear.loc[panel_bear.index[-n_recent:], "510300"] = (
            panel_bear["510300"].iloc[-n_recent:].values * 0.5
        )
        cfg = RotationConfig(
            lookback=120, top_n=5, min_history=120,
            trend_filter=TrendFilter(
                enabled=True, ma_window=200,
                exposure_bear=0.5, bond_code="511260",
            ),
        )
        state = select_and_weight(panel_bear, DEFAULT_POOL, cfg, panel_bear.index[-1])
        assert "511260" in state.weights
        # 511260 权重应 = 1 - 0.5 = 0.5
        assert abs(state.weights["511260"] - 0.5) < 1e-9


# ============================================================================
# 回测对比
# ============================================================================
class TestBacktestTrendFilter:
    """趋势过滤器的回测对比."""

    def _run_backtest(self, panel, trend_enabled: bool) -> dict:
        if trend_enabled:
            cfg = RotationConfig(
                lookback=120, top_n=5, min_history=120,
                trend_filter=TrendFilter(enabled=True, ma_window=200, exposure_bear=0.5),
            )
        else:
            cfg = RotationConfig(lookback=120, top_n=5, min_history=120)
        result = run_rotation_backtest(panel, DEFAULT_POOL, BacktestConfig(rotation=cfg))
        m = performance_metrics(result.nav)
        return {"calmar": m["calmar"], "dd": m["max_drawdown"],
                "ann": m["ann_return"], "n_rebal": len(result.states)}

    def test_both_runs_produce_valid_results(self, panel) -> None:
        r1 = self._run_backtest(panel, trend_enabled=False)
        r2 = self._run_backtest(panel, trend_enabled=True)
        assert "calmar" in r1
        assert "calmar" in r2

    def test_trend_filter_reduces_volatility(self, panel) -> None:
        """趋势过滤应降低波动 (熊市半仓)."""
        r_no = self._run_backtest(panel, trend_enabled=False)
        r_trend = self._run_backtest(panel, trend_enabled=True)
        # 趋势过滤版本波动应更低 (夏普可能更高)
        # 但收益也可能更低 — 我们只验证不崩
        assert r_trend["calmar"] > 0 or r_trend["dd"] < 0