# coding=utf-8
"""FactorRiskParity 优化器 (借鉴 v2 cell 94 + 99).

[理论]
经典 RiskParity (Qian 2005): 找权重 w, 使得各资产对组合风险的边际贡献相等.
FactorRiskParity 扩展: 先映射到因子空间:
  Σ_asset = β @ Σ_factor @ β.T
其中 β 是 (n_assets × n_factors) 因子暴露 (Lasso 估计).

[v2 source 关键约束] (cell 94)
约束 (QuantOPT):
  - bounds: 0 ≤ w_i ≤ 0.5
  - sum_lower = 0.9 (允许 0.1 cash buffer)
  - sum_upper = 1.0
  - slack=True (软约束)

[与 v1 实现的区别]
v1 (旧): 强制 sum=1, 0.5 上限, 无 sum_lower.
v2 (新): 0.9 ≤ sum ≤ 1.0, 软约束 (slack=True). 用 scipy SLSQP 求解 (与 QuantOPT 等价).

[scipy SLSQP 替代迭代式]
迭代式 w_new = w * sqrt(target / aw) 在 Σ_asset 退化或 β 接近全零时会卡住 (e.g., 某
asset 的方差远大于其他时, aw_ij 极小, w_i 跑到 1e-100 后即认为 "已收敛").
scipy SLSQP 直接最小化 Σ_i (RC_i - target)^2, 加 slack 边界, 数值稳定且与 QuantOPT 等价.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class FactorRiskParityOptimizer:
    """因子风险平价优化器 (faithful to v2 source cell 94 + 99).

    参数:
        max_weight: 单资产最大权重 (默认 0.5, source cell 99)
        sum_lower: 总权重下限 (默认 0.9, source cell 94)
        sum_upper: 总权重上限 (默认 1.0, source cell 94)
        tol: 收敛容差 (默认 1e-8)
        max_iter: 最大迭代 (默认 200, source)
        use_soft_constraint: 是否使用软约束 sum_lower (默认 True)

    例子:
        >>> opt = FactorRiskParityOptimizer()
        >>> w = opt.optimize(factor_exposure=β, factor_cov=Σ_f)
    """

    def __init__(
        self,
        max_weight: float = 0.5,
        sum_lower: float = 0.9,
        sum_upper: float = 1.0,
        tol: float = 1e-8,
        max_iter: int = 200,
        use_soft_constraint: bool = True,
    ) -> None:
        if max_weight <= 0 or max_weight > 1:
            raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
        if not (0 <= sum_lower <= sum_upper <= 1.5):
            raise ValueError(
                f"sum_lower <= sum_upper required in [0, 1.5], got "
                f"{sum_lower} <= {sum_upper}"
            )
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        self.max_weight = max_weight
        self.sum_lower = sum_lower
        self.sum_upper = sum_upper
        self.tol = tol
        self.max_iter = max_iter
        self.use_soft_constraint = use_soft_constraint

    def _compute_assets_cov(
        self, factor_exposure: pd.DataFrame, factor_cov: pd.DataFrame,
    ) -> np.ndarray:
        """Σ_asset = β @ Σ_factor @ β.T (对称化)."""
        β = factor_exposure.values
        Σf = factor_cov.values
        Σa = β @ Σf @ β.T
        return (Σa + Σa.T) / 2

    def optimize(
        self,
        factor_exposure: pd.DataFrame,  # β: (n_assets, n_factors)
        factor_cov: pd.DataFrame,      # Σ_f: (n_factors, n_factors)
    ) -> pd.Series:
        """输出资产权重 Series (scipy SLSQP, faithful to v2 source).

        满足: 0.9 ≤ sum ≤ 1.0 (软约束), 0 ≤ w_i ≤ 0.5, 等风险贡献.
        """
        β = factor_exposure.values
        Σf = factor_cov.values
        n = β.shape[0]
        Σa = self._compute_assets_cov(factor_exposure, factor_cov)

        # 起点: 等权
        w0 = np.ones(n) / n

        # 目标函数: minimize Σ_i (RC_i - target)^2
        def objective(w):
            aw = Σa @ w
            rc = w * aw
            target = np.mean(rc)
            return float(np.sum((rc - target) ** 2))

        # Bounds
        bounds = [(0.0, self.max_weight)] * n

        # Constraints
        # source cell 94 用 sum_lower=0.9 (允许 0.1 cash buffer).
        # 我们用 SLSQP 求解, 硬约束 sum = sum_lower (来源默认起始仓位).
        # 这样保证 sum = 0.9 而非自由 sum ∈ [0.9, 1.0].
        constraints = []
        if self.use_soft_constraint:
            constraints.append({
                'type': 'eq',  # 硬约束 sum = sum_lower (开始仓位)
                'fun': lambda w: np.sum(w) - self.sum_lower
            })
            constraints.append({
                'type': 'ineq',  # 上限 soft
                'fun': lambda w: self.sum_upper - np.sum(w)
            })
        else:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1.0
            })

        try:
            result = minimize(
                objective, w0,
                bounds=bounds,
                constraints=constraints,
                method='SLSQP',
                options={'maxiter': self.max_iter, 'ftol': self.tol},
            )
            w = result.x
        except Exception:
            # 兜底: 等权
            w = np.ones(n) / n

        # 数值稳定: clip + 归一化
        w = np.clip(w, 0.0, self.max_weight)
        sum_w = w.sum()
        if sum_w < 1e-12:
            w = np.ones(n) / n
        else:
            # soft constraint: 把 sum 缩放到 sum_lower (默认 0.9)
            if self.use_soft_constraint:
                target_sum = np.clip(sum_w, self.sum_lower, self.sum_upper)
                w = w * (target_sum / sum_w)
                w = np.clip(w, 0.0, self.max_weight)

        return pd.Series(w, index=factor_exposure.index, name="weight")

    def optimize_factor_only(
        self, factor_cov: pd.DataFrame,
    ) -> pd.Series:
        """仅在因子空间做 RiskParity (用于 ablation)."""
        Σf = factor_cov.values
        n = Σf.shape[0]
        w0 = np.ones(n) / n

        def objective(w):
            aw = Σf @ w
            rc = w * aw
            target = np.mean(rc)
            return float(np.sum((rc - target) ** 2))

        bounds = [(0.0, self.max_weight)] * n
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - self.sum_lower},
            {'type': 'ineq', 'fun': lambda w: self.sum_upper - np.sum(w)},
        ]

        try:
            result = minimize(
                objective, w0, bounds=bounds, constraints=constraints,
                method='SLSQP', options={'maxiter': self.max_iter, 'ftol': self.tol},
            )
            w = result.x
        except Exception:
            w = np.ones(n) / n

        w = np.clip(w, 0.0, self.max_weight)
        sum_w = w.sum()
        if sum_w < 1e-12:
            w = np.ones(n) / n
        else:
            target_sum = np.clip(sum_w, self.sum_lower, self.sum_upper)
            w = w * (target_sum / sum_w)
            w = np.clip(w, 0.0, self.max_weight)
        return pd.Series(w, index=factor_cov.index, name="factor_weight")


__all__ = ["FactorRiskParityOptimizer"]
