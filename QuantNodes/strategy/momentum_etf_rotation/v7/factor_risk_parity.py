# coding=utf-8
"""FactorRiskParity 优化器 (借鉴 v2 cell 114 RiskParity).

[理论]
经典 RiskParity (Qian 2005): 找权重 w, 使得各资产对组合风险的边际贡献相等.
FactorRiskParity 扩展: 先映射到因子空间:
  Σ_asset = β @ Σ_factor @ β.T
其中 β 是 (n_assets × n_factors) 因子暴露 (Lasso 估计).

[迭代公式]
w_{k+1} = sqrt(w_k * (Σ_asset @ w_k) / risk_target)
其中 risk_target = mean(w_k * (Σ_asset @ w_k))
收敛条件: max|w_new - w_old| < tol

[参考] v2 cell 114 的 QuantOPT.utils.RunOpt(method='FactorRiskParity')
[简化] 本实现不调 QuantOPT, 直接 numpy 求解, 加速在线场景.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class FactorRiskParityOptimizer:
    """因子风险平价优化器.

    参数:
        max_weight: 单资产最大权重 (默认 0.5)
        tol: 收敛容差 (默认 1e-8)
        max_iter: 最大迭代 (默认 200)

    例子:
        >>> opt = FactorRiskParityOptimizer()
        >>> w = opt.optimize(factor_exposure=β, factor_cov=Σ_f)
        >>> # w: pd.Series shape (n_assets,), sum=1, w_i ∈ [0, max_weight]
    """

    def __init__(
        self,
        max_weight: float = 0.5,
        tol: float = 1e-8,
        max_iter: int = 200,
    ) -> None:
        if max_weight <= 0 or max_weight > 1:
            raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        self.max_weight = max_weight
        self.tol = tol
        self.max_iter = max_iter

    def optimize(
        self,
        factor_exposure: pd.DataFrame,  # β: (n_assets, n_factors)
        factor_cov: pd.DataFrame,      # Σ_f: (n_factors, n_factors)
    ) -> pd.Series:
        """输出资产权重 Series.

        Returns:
            pd.Series shape (n_assets,), sum=1, w_i ∈ [0, max_weight]
        """
        β = factor_exposure.values
        Σf = factor_cov.values
        n = β.shape[0]

        # 资产协方差
        Σa = β @ Σf @ β.T  # (n, n)
        Σa = (Σa + Σa.T) / 2  # 强制对称

        # 等权起点
        w = np.ones(n) / n
        for it in range(self.max_iter):
            # 资产协方差下的边际风险
            aw = Σa @ w
            risk_contrib = w * aw
            target = risk_contrib.mean()
            if target <= 0:
                # 兜底: 等权
                break

            # 风险平价迭代 (Qian 2005 / Spinu 2013):
            #   w_i^(k+1) = w_i^k * sqrt(target / (w_i^k * (Σa @ w^k)_i))
            #   即 w_i ∝ 1/sqrt((Σa @ w)_i)
            # 高方差因子暴露 → 权重应减小
            w_new = w * np.sqrt(np.maximum(target, 1e-12) / np.maximum(aw, 1e-12))

            # 约束: bounds
            w_new = np.clip(w_new, 0, self.max_weight)
            sum_w = w_new.sum()
            if sum_w < 1e-12:
                # 全归零, 退化为等权
                w = np.ones(n) / n
                break
            w_new = w_new / sum_w

            # 收敛判据
            if np.abs(w_new - w).max() < self.tol:
                w = w_new
                break
            w = w_new

        return pd.Series(w, index=factor_exposure.index, name="weight")

    def optimize_factor_only(
        self, factor_cov: pd.DataFrame,
    ) -> pd.Series:
        """仅在因子空间做 RiskParity (用于 ablation).

        Returns:
            pd.Series 因子权重 (n_factors,)
        """
        Σf = factor_cov.values
        n = Σf.shape[0]
        w = np.ones(n) / n
        for _ in range(self.max_iter):
            aw = Σf @ w
            contrib = w * aw
            target = contrib.mean()
            if target <= 0:
                break
            w_new = w * np.sqrt(np.maximum(target, 1e-12) / np.maximum(aw, 1e-12))
            w_new = np.clip(w_new, 0, self.max_weight)
            sum_w = w_new.sum()
            if sum_w < 1e-12:
                break
            w_new = w_new / sum_w
            if np.abs(w_new - w).max() < self.tol:
                break
            w = w_new
        return pd.Series(w, index=factor_cov.index, name="factor_weight")


__all__ = ["FactorRiskParityOptimizer"]
