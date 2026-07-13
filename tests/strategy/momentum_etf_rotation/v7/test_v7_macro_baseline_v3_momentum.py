# coding=utf-8
"""v7_macro_baseline_v3_momentum 测试."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.v7 import (
    load_factor_returns, load_index_panel, load_benchmark_price,
    v7_macro_baseline, run_v7_3_backtest,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.v3_momentum_backtest import (
    v3_momentum_config, run_v3_momentum_backtest,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.momentum_overlay import (
    compute_momentum_score, apply_momentum_tilt_a, scores_to_weights,
    EQUITY_INDICES, COMMODITY_INDICES, BOND_INDICES, MOMENTUM_UNIVERSE,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader import INDEX_COLS


# ============================================================================
# 1. 动量计算测试
# ============================================================================
class TestMomentumScore:
    @pytest.fixture
    def idx_ret(self) -> pd.DataFrame:
        return load_index_panel()

    def test_price_momentum(self, idx_ret) -> None:
        """price 动量: P(T)/P(T-90)-1, bond=0."""
        scores = compute_momentum_score(idx_ret, pd.Timestamp("2025-12-31"), 90, "price")
        assert len(scores) == 13
        for col in BOND_INDICES:
            assert scores[col] == 0.0, f"bond {col} should be 0"

    def test_slope_r2_momentum(self, idx_ret) -> None:
        """slope_r2 动量: 10000*slope*R², bond=0."""
        scores = compute_momentum_score(idx_ret, pd.Timestamp("2025-12-31"), 90, "slope_r2")
        assert len(scores) == 13
        for col in BOND_INDICES:
            assert scores[col] == 0.0

    def test_hybrid_momentum(self, idx_ret) -> None:
        """hybrid: 归一化混合, bond=0, 值域 [-1,1]."""
        scores = compute_momentum_score(idx_ret, pd.Timestamp("2025-12-31"), 90, "hybrid")
        assert len(scores) == 13
        for col in BOND_INDICES:
            assert scores[col] == 0.0
        active = scores[scores.index.isin(MOMENTUM_UNIVERSE)]
        assert active.abs().max() <= 1.0 + 1e-6

    def test_insufficient_data(self, idx_ret) -> None:
        """数据不足返回全零."""
        scores = compute_momentum_score(idx_ret, pd.Timestamp("2002-01-01"), 90, "hybrid")
        assert (scores == 0).all()


# ============================================================================
# 2. Option A 测试
# ============================================================================
class TestMomentumTiltA:
    @pytest.fixture
    def idx_ret(self) -> pd.DataFrame:
        return load_index_panel()

    def test_bull_unchanged(self, idx_ret) -> None:
        """α=0 时权重不变."""
        w = pd.Series([1.0 / 13] * 13, index=INDEX_COLS)
        w_out = apply_momentum_tilt_a(
            w, idx_ret, pd.Timestamp("2019-06-30"), 90, "hybrid", alpha=0.0,
        )
        np.testing.assert_array_almost_equal(w_out.values, w.values)

    def test_full_momentum(self, idx_ret) -> None:
        """α=1 时 equity+commodity 完全用动量权重, 输出合法."""
        w = pd.Series([1.0 / 13] * 13, index=INDEX_COLS)
        w_out = apply_momentum_tilt_a(
            w, idx_ret, pd.Timestamp("2025-12-31"), 90, "hybrid", alpha=1.0,
        )
        # 输出合法: sum=1, 无负权重
        assert abs(w_out.sum() - 1.0) < 1e-6
        assert (w_out >= 0).all(), "no negative weights"

    def test_sum_to_one(self, idx_ret) -> None:
        """输出权重 sum=1."""
        w = pd.Series([1.0 / 13] * 13, index=INDEX_COLS)
        w_out = apply_momentum_tilt_a(
            w, idx_ret, pd.Timestamp("2025-12-31"), 90, "hybrid", alpha=0.3,
        )
        assert abs(w_out.sum() - 1.0) < 1e-6


# ============================================================================
# 3. Config 测试
# ============================================================================
class TestV3MomentumConfig:
    def test_default_config(self) -> None:
        """默认配置: hybrid, 90d, α=0.3, Option A."""
        cfg = v3_momentum_config()
        assert cfg.momentum_enabled is True
        assert cfg.momentum_type == "hybrid"
        assert cfg.momentum_lookback == 90
        assert cfg.momentum_alpha == 0.3
        assert cfg.momentum_option == "A"
        assert cfg.trend_filter_enabled is False

    def test_custom_config(self) -> None:
        """自定义配置."""
        cfg = v3_momentum_config(
            momentum_type="slope_r2", lookback=144, alpha=0.1,
            option="B", tf_enabled=True,
        )
        assert cfg.momentum_type == "slope_r2"
        assert cfg.momentum_lookback == 144
        assert cfg.momentum_alpha == 0.1
        assert cfg.momentum_option == "B"
        assert cfg.trend_filter_enabled is True

    def test_inherits_v7_defaults(self) -> None:
        """继承 v7 默认配置."""
        cfg = v3_momentum_config()
        v7 = v7_macro_baseline()
        assert cfg.bootstrap_times == v7.bootstrap_times
        assert cfg.quarter_window == v7.quarter_window
        assert cfg.index_pool == v7.index_pool


# ============================================================================
# 4. 端到端 backtest (slow)
# ============================================================================
@pytest.mark.slow
class TestV3MomentumBacktest:
    @pytest.fixture(scope="class")
    def data(self):
        return load_factor_returns(), load_index_panel(), load_benchmark_price()

    def test_option_a_deterministic(self, data) -> None:
        """Option A 同参数同结果."""
        fr, ir, bp = data
        cfg = v3_momentum_config(momentum_type="hybrid", lookback=90, alpha=0.05, option="A")
        nav1 = run_v3_momentum_backtest(ir, fr, cfg, bp)
        nav2 = run_v3_momentum_backtest(ir, fr, cfg, bp)
        np.testing.assert_array_almost_equal(nav1.values, nav2.values, decimal=8)

    def test_option_a_improves_over_baseline(self, data) -> None:
        """Option A (with TF) 应优于 v7 baseline."""
        fr, ir, bp = data
        cfg_base = v7_macro_baseline()
        nav_base = run_v7_3_backtest(ir, fr, cfg_base)
        cfg_mom = v3_momentum_config(momentum_type="slope_r2", lookback=90, alpha=0.05, option="A", tf_enabled=True)
        nav_mom = run_v3_momentum_backtest(ir, fr, cfg_mom, bp)
        # Calmar 应 > baseline
        def calmar(nav):
            n = nav.loc["2022-01-01":]
            y = (n.index[-1] - n.index[0]).days / 365.25
            ann = (n.iloc[-1] / n.iloc[0]) ** (1/y) - 1
            dd = (n / n.cummax() - 1).min()
            return ann / abs(dd) if dd != 0 else 0
        assert calmar(nav_mom) > calmar(nav_base), "v3+mom should beat v7 baseline"

    def test_option_b_matches_tf_only(self, data) -> None:
        """Option B (第10因子) with TF 应与 TF only 一致."""
        fr, ir, bp = data
        cfg_tf = v7_macro_baseline()
        cfg_tf.trend_filter_enabled = True
        cfg_tf.trend_filter_benchmark = "沪深300指数"
        cfg_tf.trend_filter_ma = 200
        cfg_tf.trend_filter_bear = 0.5
        nav_tf = run_v7_3_backtest(ir, fr, cfg_tf, bp)

        cfg_b = v3_momentum_config(momentum_type="hybrid", lookback=90, alpha=0.05, option="B", tf_enabled=True)
        nav_b = run_v3_momentum_backtest(ir, fr, cfg_b, bp)
        # Option B 应与 TF only 非常接近 (可能因 float 精度微小差异)
        np.testing.assert_array_almost_equal(
            nav_tf.loc["2022-01-01":].values,
            nav_b.loc["2022-01-01":].values,
            decimal=4,
        )
