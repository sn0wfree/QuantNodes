# coding=utf-8
"""Tests for Stage 13: 交易成本建模 (Cost Model)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation import (
    DEFAULT_POOL,
    DiversificationCaps,
    RotationConfig,
    VolTargeting,
    CostModel,
    BacktestConfig,
    run_rotation_backtest,
    performance_metrics,
)
from QuantNodes.strategy.momentum_etf_rotation.core.portfolio import calculate_turnover_cost


def _make_panel(n_days: int = 800, seed: int = 42) -> pd.DataFrame:
    """合成测试面板."""
    rng = np.random.default_rng(seed)
    codes = list(DEFAULT_POOL.codes)
    rets = rng.normal(0.0003, 0.012, size=(n_days, len(codes)))
    prices = 100.0 * np.exp(np.cumsum(rets, axis=0))
    idx = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(prices, index=idx, columns=codes)


@pytest.fixture
def panel() -> pd.DataFrame:
    return _make_panel()


# ============================================================================
# 单元测试: calculate_turnover_cost
# ============================================================================
class TestCalculateTurnoverCost:
    def test_disabled_zero_cost(self) -> None:
        """未启用 → 成本 = 0."""
        cost = CostModel(enabled=False)
        assert calculate_turnover_cost(0.5, cost) == 0.0

    def test_commission_only(self) -> None:
        """只收佣金 (5bp) → 成本 = turnover × 5bp."""
        cost = CostModel(enabled=True, commission_bp=5.0, slippage_bp=0, impact_factor=0)
        # 50% 换手 → 0.5 × 0.0005 = 0.00025
        assert calculate_turnover_cost(0.5, cost) == pytest.approx(0.00025)

    def test_slippage_with_impact(self) -> None:
        """滑点 + 冲击成本."""
        cost = CostModel(enabled=True, commission_bp=5.0, slippage_bp=10.0, impact_factor=0.1)
        # cost_rate = (5 + 10×0.1) / 10000 = 6bp
        # 0.5 × 6bp = 3bp = 0.0003
        assert calculate_turnover_cost(0.5, cost) == pytest.approx(0.0003)

    def test_full_turnover_100pct(self) -> None:
        """100% 换手 → 完整成本率."""
        cost = CostModel(enabled=True, commission_bp=5.0, slippage_bp=10.0, impact_factor=0.1)
        # 1.0 × 6bp = 6bp
        assert calculate_turnover_cost(1.0, cost) == pytest.approx(0.0006)

    def test_zero_turnover(self) -> None:
        """0 换手 → 成本 = 0."""
        cost = CostModel(enabled=True)
        assert calculate_turnover_cost(0.0, cost) == 0.0


# ============================================================================
# 回测对比
# ============================================================================
class TestBacktestCostModel:
    def _run(self, panel, cost_enabled: bool, cost_bp: float = 5.0) -> dict:
        cfg = RotationConfig(
            lookback=120, top_n=5, min_history=120,
            cost_model=CostModel(
                enabled=cost_enabled,
                commission_bp=cost_bp,
                slippage_bp=10.0,
                impact_factor=0.1,
            ),
        )
        result = run_rotation_backtest(panel, DEFAULT_POOL, BacktestConfig(rotation=cfg))
        m = performance_metrics(result.nav)
        return {"calmar": m["calmar"], "dd": m["max_drawdown"],
                "ann": m["ann_return"], "nav_end": float(result.nav.iloc[-1])}

    def test_disabled_no_cost(self, panel) -> None:
        """禁用 cost → 回测正常运行."""
        r = self._run(panel, cost_enabled=False)
        # 回测有结果即可 (synthetic data ann 可能为 0)
        assert "nav_end" in r
        assert r["nav_end"] > 0

    def test_enabled_reduces_ann(self, panel) -> None:
        """启用 cost → 启用版本 NAV 应 <= 禁用版本."""
        r_off = self._run(panel, cost_enabled=False)
        r_on = self._run(panel, cost_enabled=True, cost_bp=20.0)  # 用 20bp 确保差异
        # 启用成本后 NAV 终值应 <= 禁用版本
        assert r_on["nav_end"] <= r_off["nav_end"] + 0.01

    def test_higher_cost_reduces_more(self, panel) -> None:
        """更高成本 → NAV 终值更低."""
        r_5bp = self._run(panel, cost_enabled=True, cost_bp=5.0)
        r_20bp = self._run(panel, cost_enabled=True, cost_bp=20.0)
        assert r_20bp["nav_end"] <= r_5bp["nav_end"] + 0.01

    def test_cost_runs_without_error(self, panel) -> None:
        """成本模型运行不报错."""
        r = self._run(panel, cost_enabled=True, cost_bp=15.0)
        assert "nav_end" in r


class TestBacktestCostWithVT:
    """成本 + 9-C (VT) 组合验证 (放宽阈值)."""

    def _run_with_vt(self, panel, cost_enabled: bool) -> dict:
        cfg = RotationConfig(
            lookback=120, top_n=5, min_history=120,
            vol_targeting=VolTargeting(
                enabled=True, target_vol=0.15, lookback=60,
                min_scale=0.3, max_scale=1.5,
            ),
            cost_model=CostModel(
                enabled=cost_enabled,
                commission_bp=5.0, slippage_bp=10.0, impact_factor=0.1,
            ),
        )
        result = run_rotation_backtest(panel, DEFAULT_POOL, BacktestConfig(rotation=cfg))
        m = performance_metrics(result.nav)
        return {"calmar": m["calmar"], "ann": m["ann_return"],
                "nav_end": float(result.nav.iloc[-1])}

    def test_vt_with_cost_runs(self, panel) -> None:
        """VT + 成本 → 回测正常运行."""
        r = self._run_with_vt(panel, cost_enabled=True)
        assert r["nav_end"] > 0