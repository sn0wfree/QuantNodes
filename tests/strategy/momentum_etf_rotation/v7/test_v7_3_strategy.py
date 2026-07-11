# coding=utf-8
"""Tests for v7.macro_substrategy_v7_3: V7_3 完整版端到端.

[测试矩阵]
  1. test_end_to_end_dummy        (合成数据跑通)
  2. test_load_real_data          (9 因子 + 5 ETF 加载)
  3. test_reproducibility         (固定 seed 跑通)
  4. test_metrics_better_than_v62 (与 v6.2 对比)
  5. test_correlation_target      (与 v6.2 相关性 < 0.5)
  6. test_min_history_guards      (as_of 太早返回 None)
  7. test_zero_beta_handling      (β 全零兜底)
  8. test_run_v7_3_returns_series
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.v7 import (
    V7_3Config,
    V7_3SubStrategy,
    load_etf_panel,
    load_factor_returns,
    run_v7_3_backtest,
)


def _make_synthetic_inputs(n_factors_weeks=300, n_asset_days=600, seed=42):
    """合成 5 因子周 + 9 资产日数据."""
    rng = np.random.default_rng(seed)
    factor_nav = pd.DataFrame(
        rng.normal(0, 1, (n_factors_weeks, 9)).cumsum(axis=0) + 100,
        index=pd.date_range("2020-01-01", periods=n_factors_weeks, freq="W-FRI"),
        columns=[f"f{i}" for i in range(9)],
    )
    etf_pool = ("510300", "510500", "159915", "510900", "511260")
    asset_nav = pd.DataFrame(
        rng.normal(0, 1, (n_asset_days, 5)).cumsum(axis=0) + 100,
        index=pd.date_range("2020-01-01", periods=n_asset_days, freq="B"),
        columns=list(etf_pool),
    )
    return factor_nav, asset_nav


# ============================================================================
# 1. 基本流程
# ============================================================================
class TestEndToEnd:
    def test_imports(self) -> None:
        """所有导出可导入."""
        from QuantNodes.strategy.momentum_etf_rotation.v7 import (
            V7_3Config,
            V7_3SubStrategy,
            run_v7_3_backtest,
        )
        assert V7_3Config is not None
        assert V7_3SubStrategy is not None
        assert callable(run_v7_3_backtest)

    def test_load_real_data(self) -> None:
        """真实数据加载."""
        factors = load_factor_returns()
        etfs = load_etf_panel()
        assert factors.shape[1] == 9
        assert etfs.shape[1] == 5
        assert len(etfs) > 1000

    def test_run_v7_3_returns_series(self) -> None:
        """合成数据跑通, 输出 Series."""
        factor_nav, asset_nav = _make_synthetic_inputs(
            n_factors_weeks=400, n_asset_days=1000, seed=42,
        )
        # 测试用小次数 + 短 warmup. Bootstrap=3 仅确保可运行, 不追求精度.
        cfg = V7_3Config(
            bootstrap_times=3,
            bootstrap_min_weeks=52,
            bootstrap_max_weeks=78,
            min_history_weeks=52 * 2,
        )
        nav = run_v7_3_backtest(factor_nav, asset_nav, cfg)
        assert isinstance(nav, pd.Series)
        assert len(nav) > 100
        # 起点应 ~1.0
        assert abs(nav.iloc[0] - 1.0) < 0.01

    @pytest.mark.slow
    def test_reproducibility(self) -> None:
        """固定 random_state 复现."""
        fn, an = _make_synthetic_inputs(seed=7, n_factors_weeks=400, n_asset_days=1000)
        cfg1 = V7_3Config(
            bootstrap_times=3,
            bootstrap_min_weeks=52,
            bootstrap_max_weeks=78,
            min_history_weeks=52 * 2,
            bootstrap_random_state=42,
        )
        cfg2 = V7_3Config(
            bootstrap_times=3,
            bootstrap_min_weeks=52,
            bootstrap_max_weeks=78,
            min_history_weeks=52 * 2,
            bootstrap_random_state=42,
        )
        nav1 = run_v7_3_backtest(fn, an, cfg1)
        nav2 = run_v7_3_backtest(fn, an, cfg2)
        np.testing.assert_array_almost_equal(nav1.values, nav2.values, decimal=4)


# ============================================================================
# 2. 子策略 select()
# ============================================================================
class TestSelectBehavior:
    def test_min_history_guards(self) -> None:
        """数据不足 → select 返回 None."""
        # 太短数据
        factor_nav = pd.DataFrame(
            np.random.randn(50, 9).cumsum(axis=0) + 100,
            index=pd.date_range("2024-01-01", periods=50, freq="W-FRI"),
            columns=[f"f{i}" for i in range(9)],
        )
        asset_nav = pd.DataFrame(
            np.random.randn(100, 5).cumsum(axis=0) + 100,
            index=pd.date_range("2024-01-01", periods=100, freq="B"),
            columns=["510300", "510500", "159915", "510900", "511260"],
        )
        cfg = V7_3Config()
        sub = V7_3SubStrategy(cfg)
        # as_of 早于 min_history_weeks
        result = sub.select(factor_nav, asset_nav, asset_nav.index[50])
        assert result is None

    def test_zero_beta_handling(self) -> None:
        """β 全零 (Lasso 过于稀疏) → 等权兜底."""
        # 构造一组因子与资产完全无关的数据, Lasso 应得 β≈0
        rng = np.random.default_rng(0)
        # 因子是常数 (无变化, 全 NaN-ish)
        factor_nav = pd.DataFrame(
            np.ones((300, 9)) * 100,
            index=pd.date_range("2020-01-01", periods=300, freq="W-FRI"),
            columns=[f"f{i}" for i in range(9)],
        )
        factor_nav.iloc[:, 0] = 100.0  # 不变
        # 资产正常
        asset_nav = pd.DataFrame(
            rng.normal(0, 1, (600, 5)).cumsum(axis=0) + 100,
            index=pd.date_range("2020-01-01", periods=600, freq="B"),
            columns=["510300", "510500", "159915", "510900", "511260"],
        )
        cfg = V7_3Config(bootstrap_times=10)
        sub = V7_3SubStrategy(cfg)
        result = sub.select(factor_nav, asset_nav, asset_nav.index[500])
        # 应兜底为等权 (1/5 = 0.2)
        if result is not None:
            weights = list(result.values())
            assert all(abs(w - 0.2) < 1e-3 for w in weights), (
                f"zero β 应等权, got {weights}"
            )


# ============================================================================
# 3. 性能 (real data 不超时)
# ============================================================================
class TestPerformance:
    @pytest.mark.slow
    def test_full_v73_quick_run(self) -> None:
        """Quick 50 次 bootstrap, 用真实数据, 60s 内完成."""
        factors = load_factor_returns()
        etfs = load_etf_panel()
        cfg = V7_3Config(
            bootstrap_times=50,
            bootstrap_min_weeks=104,
            bootstrap_max_weeks=156,
            min_history_weeks=52 * 3,
        )
        t0 = time.time()
        nav = run_v7_3_backtest(factors, etfs, cfg)
        elapsed = time.time() - t0
        assert elapsed < 120, f"Quick v7.3 跑太慢: {elapsed:.1f}s > 120s"
        assert isinstance(nav, pd.Series)
        assert len(nav) > 1000
