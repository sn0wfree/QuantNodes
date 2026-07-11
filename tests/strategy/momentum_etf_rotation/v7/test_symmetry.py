# coding=utf-8
"""Tests for v7.symmetry: RollingSymmetry (Klein 2013) — 严格 no-lookahead.

[关键] v7.3 的 look-ahead 风险点:
  Rolling Symmetry t 时刻结果必须独立于 [t+1, ] 数据.

[测试矩阵]
  1. test_t1_independent_of_t2  (无 look-ahead, 核心)
  2. test_idempotent             (Symmetry² = Symmetry)
  3. test_uncorrelated           (cov ≈ I 矩阵)
  4. test_min_periods            (< min_periods → None)
  5. test_panel_consistency      (rolling 全样本 vs fit_transform_full)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.v7.symmetry import (
    RollingSymmetry,
)


def _make_factors(n: int = 200, k: int = 9, seed: int = 42) -> pd.DataFrame:
    """合成相关因子 (有共线性, 正定协方差)."""
    rng = np.random.default_rng(seed)
    # 用更强正则化保证协方差矩阵正定
    A = rng.normal(0, 1, (k, k))
    cov = A @ A.T + np.eye(k) * k  # 严格正定 (Gram + scale*I)
    L = np.linalg.cholesky(cov)
    base = rng.normal(0, 1, (n, k))
    factors = base @ L.T
    return pd.DataFrame(
        factors,
        index=pd.date_range("2020-01-01", periods=n, freq="W-FRI"),
        columns=[f"f{i}" for i in range(k)],
    )


# ============================================================================
# 1. No-lookahead (核心)
# ============================================================================
class TestNoLookahead:
    """修改未来数据, 不影响 t 时刻结果."""

    def test_t1_independent_of_t2(self) -> None:
        factors = _make_factors(n=200, k=9)
        sym = RollingSymmetry(window=52, min_periods=26)
        result_t = sym.fit_transform(factors, idx=100)

        # 大幅修改 t+1 (idx=101)
        factors_mod = factors.copy()
        factors_mod.iloc[101] += 1000.0
        sym2 = RollingSymmetry(window=52, min_periods=26)
        result_t_modified = sym2.fit_transform(factors_mod, idx=100)

        np.testing.assert_array_almost_equal(
            result_t.values,
            result_t_modified.values,
            decimal=6,
            err_msg="t+1 数据修改导致 t 时刻结果变化 → look-ahead!",
        )

    def test_window_end_protected(self) -> None:
        """修改 t 之前很久的数据 (早于窗口) 不影响 t 时刻结果.

        window=10 → t=15 时刻窗口 = [6, 15].
        修改 idx=0 (在窗口外) 不应影响 t=15 的对称变换.
        """
        sym = RollingSymmetry(window=10, min_periods=5)
        factors = _make_factors(n=20, k=3, seed=1)
        res_t = sym.fit_transform(factors, idx=15)

        # 修改 idx=0 (在 t=15 的窗口 [6, 15] 之外)
        factors_mod = factors.copy()
        factors_mod.iloc[0] += 1000
        sym2 = RollingSymmetry(window=10, min_periods=5)
        res_t_mod = sym2.fit_transform(factors_mod, idx=15)

        np.testing.assert_array_almost_equal(res_t.values, res_t_mod.values)

    def test_panel_consistent_with_fit_transform(self) -> None:
        """transform_panel 在 idx=i 处应 = fit_transform idx=i."""
        factors = _make_factors(n=100, k=5)
        sym = RollingSymmetry(window=30, min_periods=10)
        panel = sym.transform_panel(factors)

        # 测试 5 个不同 idx
        for idx in [30, 50, 70, 90, 99]:
            direct = sym.fit_transform(factors, idx)
            from_panel = panel.iloc[idx]
            np.testing.assert_array_almost_equal(
                direct.values,
                from_panel.values,
                decimal=8,
                err_msg=f"idx={idx}: panel 与 fit_transform 不一致",
            )


# ============================================================================
# 2. 数学性质
# ============================================================================
class TestSymmetryProperties:
    def test_idempotent(self) -> None:
        """全样本 Symmetry 是 symmetric / involutive: Symmetry(x) 后再 cov → I."""
        # 全样本 Symmetry 后应该已经 cov=I, 再 Symmetry 仍 cov=I (近似)
        sym = RollingSymmetry(window=52, min_periods=26)
        factors = _make_factors(n=200, k=5, seed=7)
        first = sym.fit_transform_full(factors)
        second = sym.fit_transform_full(first)
        # 两次 cov 都应约 I
        np.testing.assert_array_almost_equal(
            first.cov().values, np.eye(5), decimal=2,
            err_msg="第一次 Symmetry 后 cov ≠ I"
        )
        np.testing.assert_array_almost_equal(
            second.cov().values, np.eye(5), decimal=2,
            err_msg="第二次 Symmetry 后 cov ≠ I (idempotent 破坏)"
        )

    def test_uncorrelated_full_sample(self) -> None:
        """全样本 Symmetry: cov(output) ≈ I (单位阵)."""
        sym = RollingSymmetry(window=52, min_periods=26)
        factors = _make_factors(n=300, k=9, seed=11)
        full = sym.fit_transform_full(factors)
        cov_full = full.cov().values
        # cov ≈ I
        np.testing.assert_array_almost_equal(
            cov_full,
            np.eye(9),
            decimal=2,
            err_msg=f"全样本 Symmetry cov 应接近单位阵:\n{cov_full}",
        )

    def test_equal_variance_each_factor(self) -> None:
        """Symmetry 后每个因子方差约等于 1."""
        sym = RollingSymmetry(window=52, min_periods=26)
        factors = _make_factors(n=300, k=9, seed=12)
        full = sym.fit_transform_full(factors)
        std_per_factor = full.std().values
        np.testing.assert_array_almost_equal(
            std_per_factor,
            np.ones(9),
            decimal=1,
        )


# ============================================================================
# 3. 边界 / 配置
# ============================================================================
class TestConfigAndBounds:
    def test_min_periods_not_reached(self) -> None:
        """idx < min_periods - 1 返回 None."""
        sym = RollingSymmetry(window=52, min_periods=26)
        factors = _make_factors(n=100)
        # idx=10 < 25 = min_periods-1
        assert sym.fit_transform(factors, idx=10) is None
        # idx=25 应有结果
        result = sym.fit_transform(factors, idx=25)
        assert result is not None

    def test_window_smaller_than_min_periods_raises(self) -> None:
        with pytest.raises(ValueError):
            RollingSymmetry(window=10, min_periods=20)

    def test_window_too_small_raises(self) -> None:
        with pytest.raises(ValueError):
            RollingSymmetry(window=2)

    def test_transform_panel_nan_before_min(self) -> None:
        """transform_panel 前 min_periods-1 行应为 NaN."""
        sym = RollingSymmetry(window=20, min_periods=10)
        factors = _make_factors(n=50, k=3)
        out = sym.transform_panel(factors)
        # 索引 0-8 应是 NaN
        assert out.iloc[:9].isna().all().all()
        # 索引 9 起有值
        assert out.iloc[9:].notna().any().all()
