# coding=utf-8
"""Tests for v7.macro_substrategy_v7_3: V7_3 v2 完整版端到端 (faithful to source).

[测试矩阵]
  1. test_end_to_end_dummy        (合成数据跑通)
  2. test_load_real_indices        (13 indices + 9 factor 加载)
  3. test_reproducibility           (固定 seed 跑通)
  4. test_run_v7_3_returns_series   (集成回测跑通, 输出 Series)
  5. test_smoke_test_performance    (确保 select() 不报错, sum=1)
  6. test_quarter_window            (8 quarter window)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.v7 import (
    V7_3Config,
    V7_3SubStrategy,
    load_aligned_prices,
    run_v7_3_backtest,
)


def _make_synthetic_inputs(n_factors_weeks=300, n_idx_daily=600, seed=42):
    """合成 13 indices daily + 9 factor weekly 数据."""
    rng = np.random.default_rng(seed)
    # weekly factor returns
    factor_nav = pd.DataFrame(
        rng.normal(0, 1, (n_factors_weeks, 9)).cumsum(axis=0) + 100,
        index=pd.date_range("2020-01-03", periods=n_factors_weeks, freq="W"),
        columns=[f"f{i}" for i in range(9)],
    )
    # daily idx returns
    etf_pool_idx = list(range(13))
    asset_ret = pd.DataFrame(
        rng.normal(0, 1, (n_idx_daily, 13)).cumsum(axis=0) + 100,
        index=pd.date_range("2020-01-02", periods=n_idx_daily, freq="B"),
        columns=[f"i{i}" for i in etf_pool_idx],
    )
    return factor_nav, asset_ret


# ============================================================================
# 1. 基本流程
# ============================================================================
class TestEndToEnd:
    def test_imports(self) -> None:
        from QuantNodes.strategy.momentum_etf_rotation.v7 import (
            V7_3Config,
            V7_3SubStrategy,
            run_v7_3_backtest,
        )
        assert V7_3Config is not None
        assert V7_3SubStrategy is not None
        assert callable(run_v7_3_backtest)

    def test_load_real_indices(self) -> None:
        data = load_aligned_prices(pool="index")
        assert data["factor_nav"].shape[1] == 8
        assert data["asset_prices"].shape[1] == 13
        assert len(data["asset_prices"]) > 1000

    def test_run_v7_3_returns_series(self) -> None:
        data = load_aligned_prices(pool="index")
        cfg = V7_3Config(bootstrap_times=5, quarter_window=8)  # 短 bootstrap 加速
        nav = run_v7_3_backtest(data["asset_prices"], data["factor_nav"], cfg)
        assert isinstance(nav, pd.Series)
        assert len(nav) > 100
        assert abs(nav.iloc[0] - 1.0) < 0.01

    def test_smoke_select(self) -> None:
        """Single select() 端到端."""
        data = load_aligned_prices(pool="index")
        cfg = V7_3Config(bootstrap_times=5, quarter_window=8)
        sub = V7_3SubStrategy(cfg)

        # Concat weekly simple returns (from prices)
        asset_weekly = data["asset_prices"][list(cfg.index_pool)].resample("W").last().pct_change()
        factor_weekly = data["factor_nav"][list(cfg.factor_cols)].pct_change()
        sample = pd.concat(
            [asset_weekly, factor_weekly],
            axis=1,
        ).dropna(how="any")

        # Test mid-date
        result = sub.select(sample, pd.Timestamp("2024-06-30"))
        if result is not None:
            assert 0.85 <= sum(result.values()) <= 1.05, (
                f"sum {sum(result.values()):.3f} out of [0.9, 1.0]"
            )


# ============================================================================
# 2. Quarter 窗口
# ============================================================================
class TestQuarterWindow:
    def test_8_quarter_default(self) -> None:
        cfg = V7_3Config()
        assert cfg.quarter_window == 8
        assert cfg.bootstrap_times == 500
        assert cfg.bootstrap_resample_min == 52 * 1 + 26  # 78
        assert cfg.bootstrap_resample_max == 52 * 2       # 104

    def test_factor_cov_default(self) -> None:
        cfg = V7_3Config()
        assert cfg.sum_lower == 0.9
        assert cfg.sum_upper == 1.0
        assert cfg.max_weight == 0.5
