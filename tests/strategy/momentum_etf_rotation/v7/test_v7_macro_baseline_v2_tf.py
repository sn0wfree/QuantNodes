# coding=utf-8
"""v7_macro_baseline_v2_tf (趋势过滤增强版) 锁定测试 (2026-07-13).

测试覆盖:
  1. 配置冻结 (3 个)
  2. apply_trend_filter 单元测试 (4 个, 多场景)
  3. 端到端 backtest (3 个 slow 测试)
  4. 跨 random_state 稳定性 (1 个)

TF 修复 ROI 验证:
  - 2018 熊市: DD 应 < v7_macro_baseline (无 TF)
  - 2022 熊市: DD 应 < v7_macro_baseline
  - OOS 2023-至今: Calmar 应 > v7_macro_baseline (0.620)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.v7 import (
    V7_3Config,
    v7_macro_baseline,
    v7_macro_baseline_v2_tf,
    apply_trend_filter,
    run_v7_3_backtest,
    load_factor_returns,
    load_index_panel,
    load_benchmark_price,
    INDEX_COLS,
)


# ============================================================================
# 1. 配置冻结
# ============================================================================
class TestV2TFConfig:
    def test_v2_config_inherits_v1(self) -> None:
        """v2 必须继承 v1 所有非 TF 配置 (锁定兼容)."""
        v1 = v7_macro_baseline()
        v2 = v7_macro_baseline_v2_tf()
        assert v2.bootstrap_times == v1.bootstrap_times == 500
        assert v2.bootstrap_random_state == v1.bootstrap_random_state == 42
        assert v2.quarter_window == v1.quarter_window == 8
        assert v2.max_weight == v1.max_weight
        assert v2.sum_lower == v1.sum_lower
        assert v2.sum_upper == v1.sum_upper
        assert v2.index_pool == v1.index_pool
        assert v2.factor_cols == v1.factor_cols

    def test_v2_tf_enabled(self) -> None:
        """v2 必须启用 TF, 3 个 TF 字段为预期值."""
        cfg = v7_macro_baseline_v2_tf()
        assert cfg.trend_filter_enabled is True
        assert cfg.trend_filter_ma == 200
        assert cfg.trend_filter_bear == 0.5
        assert cfg.trend_filter_benchmark == "沪深300指数"
        assert cfg.trend_filter_defensive == "中债10年期国债指数"

    def test_v1_baseline_unchanged(self) -> None:
        """v1 baseline TF 必须保持 False (锁定不变)."""
        cfg = v7_macro_baseline()
        assert cfg.trend_filter_enabled is False

    def test_v2_returns_new_instance(self) -> None:
        """每次调用返回新 V7_3Config, 避免污染."""
        cfg1 = v7_macro_baseline_v2_tf()
        cfg2 = v7_macro_baseline_v2_tf()
        assert cfg1 is not cfg2


# ============================================================================
# 2. apply_trend_filter 单元测试
# ============================================================================
class TestApplyTrendFilter:
    @pytest.fixture
    def benchmark(self) -> pd.Series:
        return load_benchmark_price()

    @pytest.fixture
    def w_uniform(self) -> pd.Series:
        """13 资产等权 (各 1/13 ≈ 0.0769), sum=1.0."""
        return pd.Series([1.0 / 13] * 13, index=INDEX_COLS)

    def test_bull_unchanged(self, benchmark, w_uniform) -> None:
        """多头 (沪深300 > MA200): 权重不变."""
        cfg = V7_3Config(trend_filter_enabled=True)
        # 2019-06-30 是 BULL
        w_out = apply_trend_filter(w_uniform.copy(), benchmark, pd.Timestamp("2019-06-30"), cfg)
        np.testing.assert_array_almost_equal(w_out.values, w_uniform.values)

    def test_bear_scale_and_defensive(self, benchmark, w_uniform) -> None:
        """熊市 (沪深300 < MA200): 缩放 + 配防御资产."""
        cfg = V7_3Config(trend_filter_enabled=True)
        # 2018-12-31 是 BEAR (ratio=0.860)
        w_out = apply_trend_filter(w_uniform.copy(), benchmark, pd.Timestamp("2018-12-31"), cfg)
        # 各资产 × 0.5
        np.testing.assert_array_almost_equal(
            w_out.drop(cfg.trend_filter_defensive).values,
            (w_uniform.drop(cfg.trend_filter_defensive) * 0.5).values,
        )
        # 中债10年 = 0.5 * (1/13) + 0.5 ≈ 0.5385
        assert abs(w_out[cfg.trend_filter_defensive] - 0.5 * (1/13) - 0.5) < 1e-6
        # sum 应 ≈ 0.5 (各资产 0.5) + 0.5 (中债10年加) = 1.0
        assert abs(w_out.sum() - 1.0) < 1e-6

    def test_disabled_unchanged(self, benchmark, w_uniform) -> None:
        """TF disabled: 权重永远不变."""
        cfg = V7_3Config(trend_filter_enabled=False)
        w_out = apply_trend_filter(w_uniform.copy(), benchmark, pd.Timestamp("2018-12-31"), cfg)
        np.testing.assert_array_almost_equal(w_out.values, w_uniform.values)

    def test_insufficient_data_unchanged(self, benchmark, w_uniform) -> None:
        """数据不足 (< ma_window) 时默认多头."""
        cfg = V7_3Config(trend_filter_enabled=True, trend_filter_ma=200)
        # 用一个早期日期, 数据 < 200
        w_out = apply_trend_filter(w_uniform.copy(), benchmark, pd.Timestamp("2002-06-30"), cfg)
        np.testing.assert_array_almost_equal(w_out.values, w_uniform.values)


# ============================================================================
# 3. 端到端 backtest (slow)
# ============================================================================
@pytest.mark.slow
class TestV2TFBacktest:
    @pytest.fixture(scope="class")
    def nav_v1(self) -> pd.Series:
        """v1 baseline NAV (无 TF, 对照组)."""
        factor_ret = load_factor_returns()
        idx_ret = load_index_panel()
        return run_v7_3_backtest(idx_ret, factor_ret, v7_macro_baseline())

    @pytest.fixture(scope="class")
    def nav_v2(self) -> pd.Series:
        """v2 baseline NAV (有 TF)."""
        factor_ret = load_factor_returns()
        idx_ret = load_index_panel()
        benchmark = load_benchmark_price()
        return run_v7_3_backtest(idx_ret, factor_ret, v7_macro_baseline_v2_tf(), benchmark)

    def test_v2_deterministic(self, nav_v2) -> None:
        """同 random_state → 同 NAV."""
        factor_ret = load_factor_returns()
        idx_ret = load_index_panel()
        benchmark = load_benchmark_price()
        nav_again = run_v7_3_backtest(idx_ret, factor_ret, v7_macro_baseline_v2_tf(), benchmark)
        np.testing.assert_array_almost_equal(nav_v2.values, nav_again.values, decimal=8)

    def test_v2_2018_dd_reduced(self, nav_v1, nav_v2) -> None:
        """2018 熊市: v2 DD 应 < v1 DD (TF 应减仓降回撤)."""
        dd_v1 = self._max_dd(nav_v1.loc['2018-01-01':'2019-06-30'])
        dd_v2 = self._max_dd(nav_v2.loc['2018-01-01':'2019-06-30'])
        assert dd_v2 > dd_v1, (
            f"v2 DD {dd_v2*100:.2f}% 应 > v1 DD {dd_v1*100:.2f}% (TF 应减仓降回撤)"
        )

    def test_v2_2022_dd_reduced(self, nav_v1, nav_v2) -> None:
        """2022 熊市: v2 DD 应 < v1 DD (TF 应减仓降回撤)."""
        dd_v1 = self._max_dd(nav_v1.loc['2022-01-01':'2023-06-30'])
        dd_v2 = self._max_dd(nav_v2.loc['2022-01-01':'2023-06-30'])
        assert dd_v2 > dd_v1, (
            f"v2 DD {dd_v2*100:.2f}% 应 > v1 DD {dd_v1*100:.2f}% (TF 应减仓降回撤)"
        )

    def test_v2_oos_2023_calmar_better(self, nav_v1, nav_v2) -> None:
        """OOS 2023-至今: v2 Calmar 应 > v1 (0.620)."""
        c1 = self._calmar(nav_v1.loc['2023-01-01':])
        c2 = self._calmar(nav_v2.loc['2023-01-01':])
        assert c2 > c1, (
            f"v2 Calmar {c2:.3f} 应 > v1 Calmar {c1:.3f} (TF 修复 ROI)"
        )

    def test_v2_bull_market_no_regression(self, nav_v1, nav_v2) -> None:
        """2020 牛市: TF 不触发, v2 ≈ v1."""
        # 2020-01-01 到 2021-06-30 (沪深300 强势)
        n1 = nav_v1.loc['2020-01-01':'2021-06-30']
        n2 = nav_v2.loc['2020-01-01':'2021-06-30']
        # 允许 5% 误差 (TF 信号可能在边界触发)
        diff = abs(n1.iloc[-1] - n2.iloc[-1]) / n1.iloc[-1]
        assert diff < 0.05, f"2020 牛市 NAV 差异 {diff*100:.1f}% > 5%"

    @staticmethod
    def _max_dd(s: pd.Series) -> float:
        return float((s / s.cummax() - 1).min())

    @staticmethod
    def _calmar(s: pd.Series) -> float:
        r = s.pct_change().dropna()
        n_years = len(r) / 252
        ann = (s.iloc[-1] / s.iloc[0]) ** (1 / n_years) - 1
        dd = (s / s.cummax() - 1).min()
        return ann / abs(dd) if abs(dd) > 0.001 else 0


# ============================================================================
# 4. 跨 random_state 稳定性 (slow)
# ============================================================================
@pytest.mark.slow
class TestV2TFStability:
    """v2 跨 3 seeds 平均 Ann CV% 应 < 10% (与 v1 一致)."""

    SEEDS = [42, 7, 123]
    CV_LIMIT = 0.10

    @pytest.fixture(scope="class")
    def navs_by_seed(self) -> dict[int, pd.Series]:
        factor_ret = load_factor_returns()
        idx_ret = load_index_panel()
        benchmark = load_benchmark_price()
        out = {}
        for seed in self.SEEDS:
            cfg = v7_macro_baseline_v2_tf()
            cfg.bootstrap_random_state = seed
            out[seed] = run_v7_3_backtest(idx_ret, factor_ret, cfg, benchmark)
        return out

    def test_stability_ann_cv(self, navs_by_seed) -> None:
        """跨 seed Ann CV% < 10%."""
        anns = []
        for seed, nav in navs_by_seed.items():
            sub = nav.loc['2023-01-01':]
            r = sub.pct_change().dropna()
            n_years = len(r) / 252
            ann = (sub.iloc[-1] / sub.iloc[0]) ** (1 / n_years) - 1
            anns.append(ann * 100)
        anns = np.array(anns)
        cv = abs(anns.std() / anns.mean()) if anns.mean() else 0
        assert cv < self.CV_LIMIT, (
            f"v2 跨 seed Ann CV={cv:.1%} ({anns.round(2)}), 期望 < 10%"
        )
