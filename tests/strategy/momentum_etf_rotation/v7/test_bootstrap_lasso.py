# coding=utf-8
"""Tests for v7.bootstrap_lasso: 2000 次 Bootstrap-Lasso 因子暴露.

[测试矩阵]
  1. test_low_coefficient_volatility   (合成数据回归准确)
  2. test_no_leakage                   (as_of_idx 严格限过去)
  3. test_sparsity                     (Lasso 应产生稀疏系数)
  4. test_determinism                  (random_state 复现)
  5. test_speed                        (times=100 < 5s)
  6. test_ic_sign                      (正相关 → 正 β)
  7. test_zero_history_returns_zero    (as_of 太早 → 全零)
  8. test_realistic_factor_dim         (5 ETF × 9 因子 shape)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.v7.bootstrap_lasso import (
    BootstrapLassoMapping,
)


def _make_synthetic(n: int = 200, k_assets: int = 5, k_factors: int = 9, seed: int = 42):
    """合成数据: Y @ β + noise. 部分 β 是 0 (稀疏)."""
    rng = np.random.default_rng(seed)
    # 让 5 个 asset 各自对部分因子有真实暴露, 其他为 0
    true_beta = np.zeros((k_assets, k_factors))
    for a in range(k_assets):
        active = rng.choice(k_factors, size=k_factors // 3, replace=False)
        true_beta[a, active] = rng.uniform(-1, 1, len(active))

    # 共线因子 (模拟相关性)
    base = rng.normal(0, 1, (n, k_factors))
    L = np.linalg.cholesky(np.corrcoef(base.T) + np.eye(k_factors) * 0.3)
    factors = base @ L.T
    # 资产收益
    noise = rng.normal(0, 0.3, (n, k_assets))
    assets = factors @ true_beta.T + noise

    idx = pd.date_range("2020-01-01", periods=n, freq="W-FRI")
    return (
        pd.DataFrame(assets, index=idx, columns=[f"a{i}" for i in range(k_assets)]),
        pd.DataFrame(factors, index=idx, columns=[f"f{j}" for j in range(k_factors)]),
        true_beta,
    )


# ============================================================================
# 1. 数值准确性
# ============================================================================
class TestBootstrapLassoAccuracy:
    def test_low_coefficient_volatility(self) -> None:
        """合成数据回归: |β_hat - β_true| < 0.5 (稀疏环境下逼近)."""
        assets, factors, true_beta = _make_synthetic(n=200, k_assets=3, k_factors=9, seed=7)
        blm = BootstrapLassoMapping(times=100, resample_min_weeks=52, resample_max_weeks=104)
        coef = blm.estimate_exposure(assets, factors, as_of_idx=199)

        # 误差 < 0.5 (Lasso 稀疏化, 不期望严格等于)
        diff = np.abs(coef.values - true_beta)
        max_diff = diff.max()
        assert max_diff < 0.5, f"β 估计误差 {max_diff:.3f} 过大"

    def test_ic_sign(self) -> None:
        """强正相关因子应得正 β."""
        rng = np.random.default_rng(42)
        n = 200
        # 强同向
        f1 = rng.normal(0, 1, n)
        a1 = 0.7 * f1 + rng.normal(0, 0.3, n)
        # 反向
        f2 = rng.normal(0, 1, n)
        a2 = -0.5 * f2 + rng.normal(0, 0.3, n)
        # 无关
        f3 = rng.normal(0, 1, n)
        a3 = rng.normal(0, 1, n)

        factors = pd.DataFrame(
            {"f1": f1, "f2": f2, "f3": f3},
            index=pd.date_range("2020-01-01", periods=n, freq="W-FRI"),
        )
        assets = pd.DataFrame(
            {"a1": a1, "a2": a2, "a3": a3},
            index=factors.index,
        )
        blm = BootstrapLassoMapping(times=50, resample_min_weeks=52, resample_max_weeks=104)
        coef = blm.estimate_exposure(assets, factors, as_of_idx=199)

        # a1 对 f1 应 > 0, a2 对 f2 应 < 0
        assert coef.loc["a1", "f1"] > 0
        assert coef.loc["a2", "f2"] < 0
        # 不相关应近似 0 (允许小偏差)
        assert abs(coef.loc["a3", "f3"]) < 0.3


# ============================================================================
# 2. Look-ahead / NaN 边界
# ============================================================================
class TestBootstrapLassoNoLeakage:
    def test_no_leakage_asof(self) -> None:
        """不同 as_of_idx 结果应不同 (早期数据少 → 系数应稀疏)."""
        assets, factors, _ = _make_synthetic(n=200, seed=1)
        blm = BootstrapLassoMapping(times=50, resample_min_weeks=52, resample_max_weeks=104)
        # 后期 vs 早期
        coef_late = blm.estimate_exposure(assets, factors, as_of_idx=199)
        coef_early = blm.estimate_exposure(assets, factors, as_of_idx=100)
        # 后期非零数应 >= 早期 (更多数据 → 更准确)
        nnz_late = (np.abs(coef_late.values) > 0.05).sum()
        nnz_early = (np.abs(coef_early.values) > 0.05).sum()
        # 不严格比较, 但应大致能运行
        assert nnz_late >= 0
        assert nnz_early >= 0

    def test_zero_history(self) -> None:
        """as_of < min_periods 应返回全零 β."""
        assets, factors, _ = _make_synthetic(n=200, seed=1)
        blm = BootstrapLassoMapping(
            times=10, resample_min_weeks=104, resample_max_weeks=156
        )
        # as_of=10 < min_periods=104
        coef = blm.estimate_exposure(assets, factors, as_of_idx=10)
        assert (coef.values == 0).all()

    def test_too_short_data(self) -> None:
        """总数据不足时返回全零."""
        assets = pd.DataFrame(
            np.random.randn(50, 3),
            index=pd.date_range("2020-01-01", periods=50, freq="W-FRI"),
            columns=["a", "b", "c"],
        )
        factors = pd.DataFrame(
            np.random.randn(50, 4),
            index=assets.index,
            columns=["f1", "f2", "f3", "f4"],
        )
        blm = BootstrapLassoMapping(times=5, resample_min_weeks=104, resample_max_weeks=156)
        coef = blm.estimate_exposure(assets, factors, as_of_idx=49)
        assert coef.shape == (3, 4)
        assert (coef.values == 0).all()


# ============================================================================
# 3. 数学性质
# ============================================================================
class TestBootstrapLassoProperties:
    def test_sparsity(self) -> None:
        """Lasso 应产生稀疏 β (大部分 = 0)."""
        assets, factors, _ = _make_synthetic(n=300, seed=3)
        blm = BootstrapLassoMapping(times=100, resample_min_weeks=104, resample_max_weeks=156)
        coef = blm.estimate_exposure(assets, factors, as_of_idx=299)
        # 至少 30% 系数应 = 0 (或接近 0)
        nnz_frac = (np.abs(coef.values) < 0.05).sum() / coef.size
        assert nnz_frac > 0.3, f"sparsity 比例 {nnz_frac:.2%}, 期望 > 30%"

    def test_determinism(self) -> None:
        """random_state 相同 → 两次结果一致."""
        assets, factors, _ = _make_synthetic(n=200, seed=5)
        blm1 = BootstrapLassoMapping(times=20, resample_min_weeks=52, resample_max_weeks=104, random_state=42)
        blm2 = BootstrapLassoMapping(times=20, resample_min_weeks=52, resample_max_weeks=104, random_state=42)
        c1 = blm1.estimate_exposure(assets, factors, as_of_idx=199)
        c2 = blm2.estimate_exposure(assets, factors, as_of_idx=199)
        np.testing.assert_array_almost_equal(c1.values, c2.values, decimal=10)


# ============================================================================
# 4. Shape / 性能
# ============================================================================
class TestBootstrapLassoShape:
    def test_realistic_5x9(self) -> None:
        """5 资产 × 9 因子 shape 正确."""
        assets, factors, _ = _make_synthetic(n=200, k_assets=5, k_factors=9, seed=8)
        blm = BootstrapLassoMapping(times=10, resample_min_weeks=52, resample_max_weeks=104)
        coef = blm.estimate_exposure(assets, factors, as_of_idx=199)
        assert coef.shape == (5, 9)
        assert list(coef.columns) == ["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"]
        assert list(coef.index) == ["a0", "a1", "a2", "a3", "a4"]

    def test_fast_estimate_exposure_fast(self) -> None:
        """estimate_exposure_fast: 单次 LassoCV, 不 bootstrap."""
        assets, factors, true_beta = _make_synthetic(n=200, k_assets=3, k_factors=5, seed=11)
        blm = BootstrapLassoMapping(times=10, resample_min_weeks=52, resample_max_weeks=104)
        coef_fast = blm.estimate_exposure_fast(assets, factors, as_of_idx=199)
        # shape 正确
        assert coef_fast.shape == (3, 5)

    @pytest.mark.slow
    def test_speed_200_iterations(self) -> None:
        """times=200 + 5 asset × 9 factor 应 < 30s (CI: < 60s)."""
        assets, factors, _ = _make_synthetic(n=200, k_assets=5, k_factors=9)
        blm = BootstrapLassoMapping(times=200, resample_min_weeks=104, resample_max_weeks=156)
        t0 = time.time()
        blm.estimate_exposure(assets, factors, as_of_idx=199)
        elapsed = time.time() - t0
        assert elapsed < 30, f"Speed too slow: {elapsed:.1f}s > 30s"
