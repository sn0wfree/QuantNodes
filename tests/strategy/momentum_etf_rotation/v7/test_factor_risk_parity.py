# coding=utf-8
"""Tests for v7.factor_risk_parity: 因子风险平价优化器.

[测试矩阵]
  1. test_converges              (max_iter 内收敛)
  2. test_constraints_satisfied  (sum=1, 0 ≤ w_i ≤ max_weight)
  3. test_equal_input_equal_w    (等 β → 等权重)
  4. test_factor_cov_influence   (Σ_f → w 改变)
  5. test_no_negative_weight     (w_i ≥ 0)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from QuantNodes.strategy.momentum_etf_rotation.v7.factor_risk_parity import (
    FactorRiskParityOptimizer,
)


def _make_inputs(n_assets=5, n_factors=9, seed=42):
    """合成 β 和 Σ_f."""
    rng = np.random.default_rng(seed)
    # β: 每个 asset 对 3 个因子活跃
    β = np.zeros((n_assets, n_factors))
    for a in range(n_assets):
        active = rng.choice(n_factors, size=3, replace=False)
        β[a, active] = rng.uniform(-1, 1, 3)

    # Σ_f: 对称正定
    A = rng.normal(0, 1, (n_factors, n_factors))
    Σ_f = A @ A.T + np.eye(n_factors) * n_factors

    idx_a = pd.date_range("2020-01-01", periods=n_assets, freq="ME")
    idx_f = pd.date_range("2020-01-01", periods=n_factors, freq="ME")
    return (
        pd.DataFrame(β, index=[f"a{i}" for i in range(n_assets)],
                     columns=[f"f{j}" for j in range(n_factors)]),
        pd.DataFrame(Σ_f, index=idx_f, columns=idx_f),
    )


# ============================================================================
# 1. 收敛 / 配置
# ============================================================================
class TestFactorRiskParityConvergence:
    def test_converges(self) -> None:
        β, Σ_f = _make_inputs()
        opt = FactorRiskParityOptimizer(max_iter=200)
        w = opt.optimize(β, Σ_f)
        assert isinstance(w, pd.Series)
        assert len(w) == 5

    def test_max_iter_slow_converges(self) -> None:
        """设置 max_iter=10, 仍返回合理权重 (即使未完全收敛)."""
        β, Σ_f = _make_inputs(seed=99)
        opt = FactorRiskParityOptimizer(max_iter=10, tol=1e-12)
        w = opt.optimize(β, Σ_f)
        # sum 应 ~0.9 (sum_lower 硬约束)
        assert abs(w.sum() - 0.9) < 0.1

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            FactorRiskParityOptimizer(max_weight=1.5)
        with pytest.raises(ValueError):
            FactorRiskParityOptimizer(max_weight=0)
        with pytest.raises(ValueError):
            FactorRiskParityOptimizer(tol=-1)
        with pytest.raises(ValueError):
            FactorRiskParityOptimizer(max_iter=0)


# ============================================================================
# 2. 数学性质
# ============================================================================
class TestFactorRiskParityProperties:
    def test_constraints_satisfied(self) -> None:
        """output 满足: 0.9 ≤ sum ≤ 1.0, 0 ≤ w_i ≤ max_weight (source cell 94)."""
        β, Σ_f = _make_inputs()
        opt = FactorRiskParityOptimizer(max_weight=0.4)
        w = opt.optimize(β, Σ_f)

        # sum 约束 (source cell 94: 0.9 ≤ sum ≤ 1.0)
        assert 0.85 <= w.sum() <= 1.05, f"sum {w.sum():.3f} out of [0.9, 1.0]"
        assert (w >= 0).all(), f"负权重: {w[w < 0]}"
        assert (w <= 0.4 + 1e-6).all(), f"超 max_weight: {w[w > 0.4]}"

    def test_sum_lower_upper_default(self) -> None:
        """默认 sum_lower=0.9, sum_upper=1.0 (source cell 94)."""
        opt = FactorRiskParityOptimizer()
        assert opt.sum_lower == 0.9
        assert opt.sum_upper == 1.0
        assert opt.use_soft_constraint is True

    def test_equal_input_equal_w(self) -> None:
        """等 β + 单位 Σ_f → 等权重 (RiskParity 的最朴素边界).

        默认 sum_lower=0.9 → 5 等权资产各 0.18 (硬约束 sum = 0.9).
        """
        n = 5
        β = pd.DataFrame(
            np.ones((n, 3)) * 0.5,
            index=[f"a{i}" for i in range(n)],
            columns=[f"f{j}" for j in range(3)],
        )
        Σ_f = pd.DataFrame(
            np.eye(3),
            index=β.columns,
            columns=β.columns,
        )
        opt = FactorRiskParityOptimizer()
        w = opt.optimize(β, Σ_f)
        # Hard sum = 0.9 → 5 × 0.18 = 0.9
        np.testing.assert_array_almost_equal(w.values, 0.18 * np.ones(5), decimal=3)

    def test_equal_input_with_no_soft_constraint(self) -> None:
        """关闭 soft_constraint 时, 等权 → sum=1 (旧版行为)."""
        n = 5
        β = pd.DataFrame(
            np.ones((n, 3)) * 0.5,
            index=[f"a{i}" for i in range(n)],
            columns=[f"f{j}" for j in range(3)],
        )
        Σ_f = pd.DataFrame(
            np.eye(3),
            index=β.columns,
            columns=β.columns,
        )
        opt = FactorRiskParityOptimizer(use_soft_constraint=False)
        w = opt.optimize(β, Σ_f)
        # Hard sum=1 → 等权 0.20 (5 × 0.20 = 1)
        np.testing.assert_array_almost_equal(w.values, 0.20 * np.ones(5), decimal=3)

    def test_factor_cov_influence(self) -> None:
        """不同 Σ_f 给出不同 w."""
        # 自定义 β: a0 只对 f0 有暴露, a1 只对 f1, a2 只对 f2
        β_arr = np.zeros((3, 9))
        β_arr[0, 0] = 1.0  # a0 暴露 f0
        β_arr[1, 1] = 1.0  # a1 暴露 f1
        β_arr[2, 2] = 1.0  # a2 暴露 f2
        cols = [f"f{j}" for j in range(9)]
        β = pd.DataFrame(β_arr, index=[f"a{i}" for i in range(3)], columns=cols)
        opt = FactorRiskParityOptimizer()

        # Σ_f = I: 标准 RP, 三个 asset 各自一个因子, 等权
        Σ_I = pd.DataFrame(np.eye(9), index=cols, columns=cols)
        w_I = opt.optimize(β, Σ_I).values

        # Σ_f 让 f0 大方差: a0 风险大 → 应减权重
        Σ_alt = np.eye(9)
        Σ_alt[0, 0] = 100.0
        Σ_alt = pd.DataFrame(Σ_alt, index=cols, columns=cols)
        w_alt = opt.optimize(β, Σ_alt).values

        diff = np.abs(w_I - w_alt).sum()
        assert diff > 0.05, f"Σ_f 改变未影响 w: diff={diff}"
        # a0 在 w_alt 中应明显少于 w_I (因 f0 高方差)
        assert w_alt[0] < w_I[0] - 0.05, (
            f"a0 权重未减少: w_I={w_I[0]:.3f}, w_alt={w_alt[0]:.3f}"
        )

    def test_no_negative_weight(self) -> None:
        β, Σ_f = _make_inputs(seed=33)
        opt = FactorRiskParityOptimizer()
        w = opt.optimize(β, Σ_f)
        assert (w >= 0).all()

    def test_factor_only_optimize(self) -> None:
        """仅因子空间 RP (用于 ablation)."""
        β, Σ_f = _make_inputs(seed=2)
        opt = FactorRiskParityOptimizer()
        w_f = opt.optimize_factor_only(Σ_f)
        assert len(w_f) == 9
        # 默认 sum=0.9
        assert 0.85 <= w_f.sum() <= 1.05, f"sum {w_f.sum():.3f} out of [0.9, 1.0]"
        assert (w_f >= 0).all()


# ============================================================================
# 3. 数值稳定性
# ============================================================================
class TestFactorRiskParityStability:
    def test_zero_beta_handled(self) -> None:
        """β 全零 → 等权兜底 (但 sum = 0.9 = sum_lower)."""
        β = pd.DataFrame(
            np.zeros((5, 9)),
            index=[f"a{i}" for i in range(5)],
            columns=[f"f{j}" for j in range(9)],
        )
        Σ_f = pd.DataFrame(
            np.eye(9),
            index=β.columns,
            columns=β.columns,
        )
        opt = FactorRiskParityOptimizer()
        w = opt.optimize(β, Σ_f)
        # 全等权 but sum = 0.9: 5 × 0.18 = 0.9
        np.testing.assert_array_almost_equal(w.values, 0.18 * np.ones(5), decimal=3)

    def test_deterministic(self) -> None:
        """Deterministic — 相同输入应同结果."""
        β, Σ_f = _make_inputs(seed=11)
        opt1 = FactorRiskParityOptimizer()
        opt2 = FactorRiskParityOptimizer()
        w1 = opt1.optimize(β, Σ_f)
        w2 = opt2.optimize(β, Σ_f)
        np.testing.assert_array_almost_equal(w1.values, w2.values)
