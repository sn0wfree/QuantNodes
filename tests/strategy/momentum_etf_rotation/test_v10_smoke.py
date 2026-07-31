# coding=utf-8
"""v10 冒烟测试 — 确保核心模块可导入 + 基本功能正常.

覆盖:
- dual_momentum: 信号计算 + NAV 计算 (纯函数, 不依赖真实数据)
- dynamic_weight_schemes: compute_nav (通用 NAV 计算)
- 各模块可导入性
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ============================================================
# 工具函数
# ============================================================
def _make_weekly_prices(n_weeks: int = 60, seed: int = 42) -> pd.DataFrame:
    """构造 4 资产周频价格数据."""
    np.random.seed(seed)
    dates = pd.date_range('2020-01-03', periods=n_weeks, freq='W-FRI')
    codes = ['510300', '513100', '518880', '511260']
    data = {}
    for code in codes:
        data[code] = 100.0 * np.cumprod(1 + np.random.randn(n_weeks) * 0.02)
    return pd.DataFrame(data, index=dates)


def _make_daily_prices(n_days: int = 252, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.bdate_range(end='2026-06-30', periods=n_days)
    codes = ['510300', '513100', '518880', '511260']
    data = {}
    for code in codes:
        data[code] = 100.0 * np.cumprod(1 + np.random.randn(n_days) * 0.01)
    return pd.DataFrame(data, index=dates)


# ============================================================
# 导入性测试
# ============================================================
class TestV10Imports:
    """v10 所有公开接口可导入."""

    def test_import_v10_package(self):
        from QuantNodes.strategy.momentum_etf_rotation import v10
        assert hasattr(v10, 'dual_momentum_signal')
        assert hasattr(v10, 'dual_compute_nav')
        assert hasattr(v10, 'dual_metrics')
        assert hasattr(v10, 'compute_nav')
        assert hasattr(v10, 'BASE_WEIGHTS')
        assert hasattr(v10, 'STRATS')

    def test_import_dual_momentum(self):
        from QuantNodes.strategy.momentum_etf_rotation.v10.dual_momentum import (  # noqa: F401
            dual_momentum_signal, compute_nav, ASSETS, BOND_CODE,
        )
        assert BOND_CODE == '511260'
        assert len(ASSETS) == 4

    def test_import_dynamic_weight_schemes(self):
        from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (  # noqa: F401
            compute_nav, BASE_WEIGHTS, STRATS,
            scheme_a_regime, scheme_b_vol_target, scheme_c_drawdown,
            scheme_d_signal_weighted, scheme_e_hybrid,
        )
        assert 'v1.0' in BASE_WEIGHTS
        assert 'v7.10' in BASE_WEIGHTS
        assert 'v9macro' in BASE_WEIGHTS
        assert 'DualMom' in BASE_WEIGHTS
        assert sum(BASE_WEIGHTS.values()) == pytest.approx(1.0, abs=0.01)

    def test_import_rrg_rotation(self):
        from QuantNodes.strategy.momentum_etf_rotation.v10 import rrg_main
        assert callable(rrg_main)

    def test_import_epo_momentum(self):
        from QuantNodes.strategy.momentum_etf_rotation.v10 import epo_main
        assert callable(epo_main)


# ============================================================
# Dual Momentum 测试
# ============================================================
class TestDualMomentumSignal:
    """dual_momentum_signal 纯函数测试."""

    def test_signal_sum_to_one(self):
        from QuantNodes.strategy.momentum_etf_rotation.v10.dual_momentum import (
            dual_momentum_signal,
        )
        prices = _make_weekly_prices(n_weeks=30)
        weights = dual_momentum_signal(prices, lookback_weeks=4)
        assert isinstance(weights, pd.Series)
        assert len(weights) == 4
        assert weights.sum() == pytest.approx(1.0)
        assert all(w >= 0 for w in weights)

    def test_signal_all_bear_holds_bond(self):
        from QuantNodes.strategy.momentum_etf_rotation.v10.dual_momentum import (
            dual_momentum_signal, BOND_CODE,
        )
        # 所有风险资产 52 周收益为负
        n_weeks = 60
        dates = pd.date_range('2020-01-03', periods=n_weeks, freq='W-FRI')
        prices = pd.DataFrame({
            '510300': np.linspace(100, 80, n_weeks),    # 下跌
            '513100': np.linspace(100, 85, n_weeks),    # 下跌
            '518880': np.linspace(100, 90, n_weeks),    # 下跌
            '511260': np.linspace(100, 105, n_weeks),   # 微涨
        }, index=dates)
        weights = dual_momentum_signal(prices, lookback_weeks=52)
        assert weights[BOND_CODE] == pytest.approx(1.0)

    def test_signal_bull_picks_best_risk_asset(self):
        from QuantNodes.strategy.momentum_etf_rotation.v10.dual_momentum import (
            dual_momentum_signal, BOND_CODE,
        )
        # 所有资产都涨，513100 涨最多
        n_weeks = 60
        dates = pd.date_range('2020-01-03', periods=n_weeks, freq='W-FRI')
        prices = pd.DataFrame({
            '510300': np.linspace(100, 110, n_weeks),
            '513100': np.linspace(100, 130, n_weeks),   # 最高
            '518880': np.linspace(100, 120, n_weeks),
            '511260': np.linspace(100, 103, n_weeks),
        }, index=dates)
        weights = dual_momentum_signal(prices, lookback_weeks=52)
        assert weights['513100'] == pytest.approx(1.0)
        assert weights[BOND_CODE] == pytest.approx(0.0)

    def test_insufficient_history(self):
        from QuantNodes.strategy.momentum_etf_rotation.v10.dual_momentum import (
            dual_momentum_signal,
        )
        prices = _make_weekly_prices(n_weeks=5)
        # 5 周数据 + lookback 52 周 = 全部 NaN，所有收益为负 → 持有国债
        weights = dual_momentum_signal(prices, lookback_weeks=52)
        assert weights.sum() == pytest.approx(1.0)


class TestDualMomentumComputeNav:
    """dual_momentum compute_nav 测试."""

    def test_nav_starts_at_one(self):
        from QuantNodes.strategy.momentum_etf_rotation.v10.dual_momentum import (
            compute_nav,
        )
        daily = _make_daily_prices(n_days=100)
        weekly = daily.resample('W-FRI').last()
        rebal_dates = pd.DatetimeIndex(weekly.index[-10:])

        nav = compute_nav(daily, weekly, rebal_dates, cost_bp=0)
        assert isinstance(nav, pd.Series)
        assert nav.iloc[0] == pytest.approx(1.0)
        assert len(nav) == len(daily)
        assert all(v > 0 for v in nav)

    def test_nav_with_zero_cost_grows_with_market(self):
        from QuantNodes.strategy.momentum_etf_rotation.v10.dual_momentum import (
            compute_nav,
        )
        # 平稳上涨市场，NAV 应该增加
        n_days = 100
        dates = pd.bdate_range(end='2026-06-30', periods=n_days)
        daily = pd.DataFrame({
            '510300': np.linspace(100, 115, n_days),
            '513100': np.linspace(100, 120, n_days),
            '518880': np.linspace(100, 110, n_days),
            '511260': np.linspace(100, 102, n_days),
        }, index=dates)
        weekly = daily.resample('W-FRI').last()
        rebal_dates = pd.DatetimeIndex(weekly.index[10:])

        nav = compute_nav(daily, weekly, rebal_dates, cost_bp=0)
        # 长期看涨，最终 NAV > 初始
        assert nav.iloc[-1] > 1.0


# ============================================================
# Dynamic Weight Schemes 测试
# ============================================================
class TestDynamicWeightSchemes:
    """dynamic_weight_schemes 通用函数测试."""

    def test_compute_nav_no_cost(self):
        from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (
            compute_nav,
        )
        # 等权持有，收益简单
        n_days = 50
        dates = pd.bdate_range(end='2026-06-30', periods=n_days)
        prices = pd.DataFrame({
            's1': np.linspace(100, 105, n_days),
            's2': np.linspace(100, 110, n_days),
        }, index=dates)
        weights = pd.DataFrame(0.5, index=dates, columns=['s1', 's2'])

        nav = compute_nav(prices, weights, cost_bp=0)
        assert nav.iloc[0] == pytest.approx(1.0)
        assert nav.iloc[-1] > 1.0
        assert len(nav) == n_days

    def test_compute_nav_with_cost_lower_than_no_cost(self):
        from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (
            compute_nav,
        )
        n_days = 50
        dates = pd.bdate_range(end='2026-06-30', periods=n_days)
        np.random.seed(42)
        prices = pd.DataFrame({
            's1': 100 * np.cumprod(1 + np.random.randn(n_days) * 0.01),
            's2': 100 * np.cumprod(1 + np.random.randn(n_days) * 0.015),
        }, index=dates)
        # 每天都调仓 → 换手高 → 成本高
        weights = pd.DataFrame(np.random.rand(n_days, 2), index=dates, columns=['s1', 's2'])
        weights = weights.div(weights.sum(axis=1), axis=0)

        nav_no_cost = compute_nav(prices, weights, cost_bp=0)
        nav_with_cost = compute_nav(prices, weights, cost_bp=50)
        assert nav_with_cost.iloc[-1] <= nav_no_cost.iloc[-1]


# ============================================================
# 真实数据回测 (可选，无数据时跳过)
# ============================================================
class TestV10RealData:
    """真实数据上的冒烟测试 — 数据缺失时自动跳过."""

    @pytest.fixture(scope="class")
    def etf_daily(self):
        path = Path("data/real/per_etf/510300.parquet")
        if not path.exists():
            pytest.skip("真实ETF数据未找到, 跳过")
        return path

    def test_load_etf_daily(self, etf_daily):
        from QuantNodes.strategy.momentum_etf_rotation.v10.dual_momentum import (
            load_etf_daily,
        )
        s = load_etf_daily('510300')
        assert isinstance(s, pd.Series)
        assert len(s) > 0
        assert s.name == '510300'

    def test_load_all_assets_daily(self, etf_daily):
        from QuantNodes.strategy.momentum_etf_rotation.v10.dual_momentum import (
            load_all_assets_daily,
        )
        df = load_all_assets_daily()
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == 4
        assert len(df) > 0
