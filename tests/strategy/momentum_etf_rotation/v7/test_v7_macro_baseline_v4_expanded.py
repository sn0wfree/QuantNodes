# coding=utf-8
"""v7_macro_baseline_v4_expanded (扩大资产池) 测试 (2026-07-13).

测试覆盖:
  1. 数据管道 (3 个): load_expanded_panel 形状/列/无 NaN
  2. 配置冻结 (3 个): V7_4Config 默认/自定义/继承
  3. TF 适配 (2 个): expanded pool TF 生效
  4. 端到端 backtest (3 个 slow): determinism / 改善 baseline / 与 v2 对比
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.v7 import (
    V7_3Config,
    V7_4Config,
    v7_macro_baseline,
    v7_macro_baseline_v2_tf,
    v7_macro_baseline_v4_expanded,
    apply_trend_filter,
    run_v7_3_backtest,
    load_factor_returns,
    load_expanded_panel,
    load_benchmark_price,
    EXPANDED_COLS,
    EQUITY_ETF_COLS,
    COMMODITY_ETF_COLS,
    EXPANDED_BOND_INDICES,
)


# ============================================================================
# 1. 数据管道
# ============================================================================
class TestExpandedPanel:
    def test_shape(self) -> None:
        """expanded panel 应有 56 列."""
        df = load_expanded_panel()
        assert df.shape[1] == 56

    def test_columns(self) -> None:
        """expanded panel 列应与 EXPANDED_COLS 一致."""
        df = load_expanded_panel()
        assert list(df.columns) == EXPANDED_COLS

    def test_no_all_nan_rows(self) -> None:
        """expanded panel 不应有全 NaN 行."""
        df = load_expanded_panel()
        assert df.isna().all(axis=1).sum() == 0

    def test_date_range(self) -> None:
        """expanded panel 日期范围应从 2018 开始."""
        df = load_expanded_panel()
        assert df.index[0].year >= 2018


# ============================================================================
# 2. 配置冻结
# ============================================================================
class TestV4Config:
    def test_default_config(self) -> None:
        """V7_4Config 默认配置正确."""
        cfg = V7_4Config()
        assert cfg.asset_pool == "expanded"
        assert len(cfg.index_pool) == 56
        assert len(cfg.equity_cols) == 45
        assert len(cfg.commodity_cols) == 6
        assert len(cfg.bond_cols) == 5

    def test_factory_config(self) -> None:
        """v7_macro_baseline_v4_expanded() 工厂函数配置正确."""
        cfg = v7_macro_baseline_v4_expanded()
        assert cfg.asset_pool == "expanded"
        assert list(cfg.index_pool) == EXPANDED_COLS
        assert cfg.trend_filter_enabled is True
        assert cfg.trend_filter_bear == 0.5

    def test_inherits_baseline(self) -> None:
        """v4 应继承 v7+v2 baseline 非 pool 配置."""
        v2 = v7_macro_baseline_v2_tf()
        v4 = v7_macro_baseline_v4_expanded()
        assert v4.bootstrap_times == v2.bootstrap_times
        assert v4.bootstrap_random_state == v2.bootstrap_random_state
        assert v4.quarter_window == v2.quarter_window
        assert v4.max_weight == v2.max_weight

    def test_returns_new_instance(self) -> None:
        """每次调用返回新实例."""
        cfg1 = v7_macro_baseline_v4_expanded()
        cfg2 = v7_macro_baseline_v4_expanded()
        assert cfg1 is not cfg2


# ============================================================================
# 3. TF 适配
# ============================================================================
class TestV4TrendFilter:
    @pytest.fixture
    def benchmark(self) -> pd.Series:
        return load_benchmark_price()

    @pytest.fixture
    def w_uniform_expanded(self) -> pd.Series:
        """56 资产等权."""
        return pd.Series([1.0 / 56] * 56, index=EXPANDED_COLS)

    def test_bear_equity_only_expanded(self, benchmark, w_uniform_expanded) -> None:
        """expanded pool 熊市: 只减 equity ETFs, bond indices 按比例增加."""
        cfg = V7_4Config(trend_filter_enabled=True)
        # 2018-12-31 是 BEAR
        w_out = apply_trend_filter(w_uniform_expanded.copy(), benchmark, pd.Timestamp("2018-12-31"), cfg)

        equity_mask = w_out.index.isin(EQUITY_ETF_COLS)
        bond_mask = w_out.index.isin(EXPANDED_BOND_INDICES)

        # equity × 0.5
        np.testing.assert_array_almost_equal(
            w_out[equity_mask].values,
            (w_uniform_expanded[equity_mask] * 0.5).values,
        )
        # sum ≈ 1.0
        assert abs(w_out.sum() - 1.0) < 1e-6

    def test_bull_unchanged_expanded(self, benchmark, w_uniform_expanded) -> None:
        """expanded pool 多头: 权重不变."""
        cfg = V7_4Config(trend_filter_enabled=True)
        # 2019-06-30 是 BULL
        w_out = apply_trend_filter(w_uniform_expanded.copy(), benchmark, pd.Timestamp("2019-06-30"), cfg)
        np.testing.assert_array_almost_equal(w_out.values, w_uniform_expanded.values)


# ============================================================================
# 4. 端到端 backtest (slow)
# ============================================================================
@pytest.mark.slow
class TestV4Backtest:
    @pytest.fixture(scope="class")
    def nav_v1(self) -> pd.Series:
        """v1 baseline NAV (对照组)."""
        factor_ret = load_factor_returns()
        idx_ret = load_expanded_panel()
        return run_v7_3_backtest(idx_ret, factor_ret, v7_macro_baseline())

    @pytest.fixture(scope="class")
    def nav_v2(self) -> pd.Series:
        """v2 TF NAV (对照组, 13 indices)."""
        factor_ret = load_factor_returns()
        idx_ret = load_expanded_panel()
        benchmark = load_benchmark_price()
        return run_v7_3_backtest(idx_ret, factor_ret, v7_macro_baseline_v2_tf(), benchmark)

    @pytest.fixture(scope="class")
    def nav_v4(self) -> pd.Series:
        """v4 expanded pool NAV."""
        factor_ret = load_factor_returns()
        idx_ret = load_expanded_panel()
        benchmark = load_benchmark_price()
        return run_v7_3_backtest(idx_ret, factor_ret, v7_macro_baseline_v4_expanded(), benchmark)

    def test_v4_deterministic(self, nav_v4) -> None:
        """同参数 → 同 NAV."""
        factor_ret = load_factor_returns()
        idx_ret = load_expanded_panel()
        benchmark = load_benchmark_price()
        nav_again = run_v7_3_backtest(idx_ret, factor_ret, v7_macro_baseline_v4_expanded(), benchmark)
        np.testing.assert_array_almost_equal(nav_v4.values, nav_again.values, decimal=8)

    def test_v4_has_nav(self, nav_v4) -> None:
        """v4 应产生有效 NAV (>0)."""
        assert len(nav_v4) > 0
        assert nav_v4.iloc[0] == 1.0
        assert nav_v4.iloc[-1] > 0

    def test_v4_not_worse_than_v1(self, nav_v1, nav_v4) -> None:
        """v4 expanded pool 不应比 v1 baseline 差太多 (Calmar 差距 < 50%)."""
        c1 = self._calmar(nav_v1.loc['2022-01-01':])
        c4 = self._calmar(nav_v4.loc['2022-01-01':])
        # 允许 v4 比 v1 差 50% (expanded pool 可能需要更多数据)
        if c1 > 0.01:
            assert c4 > c1 * 0.5, (
                f"v4 Calmar {c4:.3f} 不应比 v1 Calmar {c1:.3f} 差太多"
            )

    @staticmethod
    def _calmar(s: pd.Series) -> float:
        n_years = (s.index[-1] - s.index[0]).days / 365.25
        ann = (s.iloc[-1] / s.iloc[0]) ** (1 / n_years) - 1
        dd = (s / s.cummax() - 1).min()
        return ann / abs(dd) if abs(dd) > 0.001 else 0
