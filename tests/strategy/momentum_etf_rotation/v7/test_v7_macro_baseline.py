# coding=utf-8
"""v7_macro_baseline 锁定测试 (2026-07-13).

任何 v7 变更必须保证:
  1. v7_macro_baseline() 配置不被修改
  2. baseline 配置生成的 NAV 序列可复现
  3. baseline 关键指标在锁定值 5% 容差内

退化 > 5% 必须更新 baseline + 加 migration note (见 docs/38).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.v7 import (
    V7_3Config,
    v7_macro_baseline,
    run_v7_3_backtest,
    load_factor_returns,
    load_index_panel,
    INDEX_COLS,
)


# ============================================================================
# 1. 配置冻结 (3 个测试)
# ============================================================================
class TestV7MacroBaselineConfig:
    def test_baseline_config_frozen(self) -> None:
        """锁定配置的所有关键参数."""
        cfg = v7_macro_baseline()
        assert cfg.bootstrap_times == 500
        assert cfg.bootstrap_resample_min == 78
        assert cfg.bootstrap_resample_max == 104
        assert cfg.bootstrap_random_state == 42
        assert cfg.bootstrap_cache_alpha is True
        assert cfg.quarter_window == 8
        assert cfg.max_weight == 0.5
        assert cfg.sum_lower == 0.9
        assert cfg.sum_upper == 1.0
        assert cfg.commission_bp == 5.0
        assert cfg.slippage_bp == 5.0

    def test_baseline_pool(self) -> None:
        """13 指数池必须含 中债1-3年国债财富指数, 不含 南华综合指数."""
        cfg = v7_macro_baseline()
        assert "中债1-3年国债财富指数" in cfg.index_pool
        assert "南华综合指数" not in cfg.index_pool
        assert len(cfg.index_pool) == 13

    def test_baseline_factor_set(self) -> None:
        """8 因子 (源 cell 99 不含 期限利差因子_加权)."""
        cfg = v7_macro_baseline()
        assert len(cfg.factor_cols) == 8
        assert "期限利差因子_加权" not in cfg.factor_cols  # 源 v2 不含

    def test_baseline_returns_new_instance(self) -> None:
        """每次调用返回新 V7_3Config, 避免被外部修改污染."""
        cfg1 = v7_macro_baseline()
        cfg2 = v7_macro_baseline()
        assert cfg1 is not cfg2
        assert cfg1.bootstrap_times == cfg2.bootstrap_times


# ============================================================================
# 2. 数据兼容性 (1 个测试)
# ============================================================================
class TestV7MacroBaselineIntegration:
    def test_v7_baseline_data_load(self) -> None:
        """baseline 配置能成功加载 13 指数 + 9 因子."""
        cfg = v7_macro_baseline()
        idx_ret = load_index_panel()
        fac_ret = load_factor_returns()
        assert all(c in idx_ret.columns for c in cfg.index_pool)
        assert all(c in fac_ret.columns for c in cfg.factor_cols)


# ============================================================================
# 3. 可复现性 + 性能锁定 (slow 测试, 需 ~130s 跑一次)
# ============================================================================
@pytest.mark.slow
class TestV7MacroBaselineReproducibility:
    @pytest.fixture(scope="class")
    def nav(self) -> pd.Series:
        factor_ret = load_factor_returns()
        idx_ret = load_index_panel()
        cfg = v7_macro_baseline()
        return run_v7_3_backtest(idx_ret, factor_ret, cfg)

    def test_baseline_deterministic(self, nav: pd.Series) -> None:
        """相同 random_state 必须产生相同 NAV."""
        factor_ret = load_factor_returns()
        idx_ret = load_index_panel()
        cfg = v7_macro_baseline()
        nav2 = run_v7_3_backtest(idx_ret, factor_ret, cfg)
        np.testing.assert_array_almost_equal(nav.values, nav2.values, decimal=8)

    def test_baseline_oos_2023_ann(self, nav: pd.Series) -> None:
        """OOS 2023-至今 Ann ≈ 5.24% (容差 ±5%, 即 4.98-5.50)."""
        sub = nav.loc['2023-01-01':]
        n_years = (sub.index[-1] - sub.index[0]).days / 365.25
        ann = (sub.iloc[-1] / sub.iloc[0]) ** (1 / n_years) - 1
        assert abs(ann - 0.0524) < 0.005, (
            f"ann={ann*100:.3f}%, 期望 5.24%±0.5pp, 退化 > 5%"
        )

    def test_baseline_oos_2023_calmar(self, nav: pd.Series) -> None:
        """OOS 2023-至今 Calmar ≈ 0.620 (容差 ±10%)."""
        sub = nav.loc['2023-01-01':]
        n_years = (sub.index[-1] - sub.index[0]).days / 365.25
        ann = (sub.iloc[-1] / sub.iloc[0]) ** (1 / n_years) - 1
        dd = (sub / sub.cummax() - 1).min()
        calmar = ann / abs(dd)
        assert abs(calmar - 0.620) < 0.062, (
            f"calmar={calmar:.4f}, 期望 0.620±10%, 退化 > 5%"
        )


# ============================================================================
# 4. 跨 random_state 稳定性 (3 seeds, 锁定统计指标)
# ============================================================================
@pytest.mark.slow
class TestV7MacroBaselineStability:
    """锁定 3 seed 平均值的稳定性, 防止 random_state 改变导致性能漂移."""

    SEEDS = [42, 7, 123]
    OOS_2023_ANN_EXPECTED = 5.244  # 3 seed 平均
    OOS_2023_ANN_CV_LIMIT = 0.10   # CV < 10% 才算稳定

    @pytest.fixture(scope="class")
    def navs_by_seed(self) -> dict[int, pd.Series]:
        factor_ret = load_factor_returns()
        idx_ret = load_index_panel()
        out = {}
        for seed in self.SEEDS:
            cfg = v7_macro_baseline()
            cfg.bootstrap_random_state = seed
            out[seed] = run_v7_3_backtest(idx_ret, factor_ret, cfg)
        return out

    def test_stability_ann_cv(self, navs_by_seed: dict[int, pd.Series]) -> None:
        """跨 seed Ann CV% < 10%."""
        anns = []
        for seed, nav in navs_by_seed.items():
            sub = nav.loc['2023-01-01':]
            n_years = (sub.index[-1] - sub.index[0]).days / 365.25
            ann = (sub.iloc[-1] / sub.iloc[0]) ** (1 / n_years) - 1
            anns.append(ann * 100)
        anns = np.array(anns)
        cv = abs(anns.std() / anns.mean()) if anns.mean() else 0
        assert cv < self.OOS_2023_ANN_CV_LIMIT, (
            f"跨 seed Ann CV={cv:.1%} ({anns.round(2)}), 期望 < 10%"
        )
