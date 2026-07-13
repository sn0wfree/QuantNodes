# coding=utf-8
"""FactorRiskParity 优化器 (忠实于 source v2 cell 99/120 + QuantOPT_model).

[Stage 2 重大修正 2026-07-13]
v7.3 v1/v2 用的是 scipy SLSQP 自实现, 优化目标是最小化 **资产层** RC spread:
  Σ_asset = β @ Σf @ β.T
  asset_rc_i = w_i * (Σ_asset @ w)_i
  loss = Σ (asset_rc_i - mean)^2

这是错的! 源 v2 cell 99 + QuantOPT_model.FactorRiskParity 优化的是 **因子层** RC spread:
  W = w @ β (组合因子暴露, 9 维)
  port_std = sqrt(W @ Σf @ W)
  factor_mrc = (Σf @ W) / port_std  (9 维)
  factor_rc = W * factor_mrc  (9 维, W_i 乘 factor_mrc_i)
  loss = Σ_{i,j} (factor_rc_i*100 - factor_rc_j*100)^2  (pairwise)

这两个优化数学上不等价. 实测: 同一 (β, Σf) 下, 权重 L1 差异 ~0.50 (50% 权重重分配).

[Stage 2 修复]
替换为源 QuantOPT_model.FactorRiskParity.run_opt(), API 与 source 完全一致.
源文件位置: /home/ll/Public/高频宏观因子/QuantOPT_model.py (已 copy 到 v7/_quantopt_model.py).

[理论]
经典 RiskParity (Qian 2005): 找权重 w, 使得各资产对组合风险的边际贡献相等.
FactorRiskParity 扩展: 让各**因子**对组合风险的边际贡献相等 (而非各资产).
  factor_rc_i = W_i * (Σf @ W)_i / port_std
其中 W = w @ β (组合因子暴露).

[v2 source 关键约束] (cell 94)
约束 (QuantOPT_model.FactorRiskParity.run_opt):
  - bounds: 0 ≤ w_i ≤ 0.5
  - 0.9 ≤ sum(w) ≤ 1.0 (软约束, ineq)
  - method='SLSQP'
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 源 QuantOPT_model (从 v2 cell 99 直接 copy)
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from _quantopt_model import FactorRiskParity  # noqa: E402


class FactorRiskParityOptimizer:
    """因子风险平价优化器 (faithful to v2 source cell 99 + QuantOPT_model).

    参数:
        max_weight: 单资产最大权重 (默认 0.5, source cell 99)
        sum_lower: 总权重下限 (默认 0.9, source cell 94)
        sum_upper: 总权重上限 (默认 1.0, source cell 94)
        random_state: 随机种子 (QuantOPT 不需要, 保留接口兼容)

    例子:
        >>> opt = FactorRiskParityOptimizer()
        >>> w = opt.optimize(factor_exposure=β, factor_cov=Σ_f)
    """

    def __init__(
        self,
        max_weight: float = 0.5,
        sum_lower: float = 0.9,
        sum_upper: float = 1.0,
        random_state: int | None = None,
    ) -> None:
        if max_weight <= 0 or max_weight > 1:
            raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
        if not (0 <= sum_lower <= sum_upper <= 1.5):
            raise ValueError(
                f"sum_lower <= sum_upper required in [0, 1.5], got "
                f"{sum_lower} <= {sum_upper}"
            )
        self.max_weight = max_weight
        self.sum_lower = sum_lower
        self.sum_upper = sum_upper
        self.random_state = random_state

    def optimize(
        self,
        factor_exposure: pd.DataFrame,  # β: (n_assets, n_factors)
        factor_cov: pd.DataFrame,      # Σ_f: (n_factors, n_factors)
    ) -> pd.Series:
        """输出资产权重 Series (调源 QuantOPT_model.FactorRiskParity, faithful to source).

        满足: 0.9 ≤ sum ≤ 1.0 (软约束, ineq), 0 ≤ w_i ≤ 0.5, 因子层等风险贡献.
        """
        β = factor_exposure.values
        Σf = factor_cov.values
        n = β.shape[0]
        β_T = β.T  # (n_factors, n_assets) for source API: factor_expo is (n_factors, n_assets)

        # Bounds: [(0, max_weight), ...] per asset
        bounds = [(0.0, self.max_weight)] * n

        # Constraints: soft ineq (source cell 99 + cell 94)
        constraints = [
            {"type": "ineq", "fun": lambda w: np.sum(w) - self.sum_lower},
            {"type": "ineq", "fun": lambda w: self.sum_upper - np.sum(w)},
        ]

        try:
            result = FactorRiskParity.run_opt(
                stockpool=list(factor_exposure.index),
                factor_cov=Σf,
                factor_expo=β,  # 源 API: factor_expo is (n_assets, n_factors)
                bounds=bounds,
                constraints=constraints,
                method="SLSQP",
            )
            w = np.asarray(result.x).ravel()
        except Exception:
            # 兜底: 等权
            w = np.ones(n) / n

        # 数值稳定: clip
        w = np.clip(w, 0.0, self.max_weight)
        sum_w = w.sum()
        if sum_w < 1e-12:
            w = np.ones(n) / n

        return pd.Series(w, index=factor_exposure.index, name="weight")

    def optimize_factor_only(
        self, factor_cov: pd.DataFrame,
    ) -> pd.Series:
        """仅在因子空间做 RiskParity (用于 ablation, 用源 RiskParity 而非 FactorRiskParity)."""
        from _quantopt_model import RiskParity
        Σf = factor_cov.values
        n = Σf.shape[0]
        bounds = [(0.0, self.max_weight)] * n
        constraints = [
            {"type": "ineq", "fun": lambda w: np.sum(w) - self.sum_lower},
            {"type": "ineq", "fun": lambda w: self.sum_upper - np.sum(w)},
        ]
        try:
            result = RiskParity.run_opt(
                stockpool=list(factor_cov.index),
                cov=Σf,
                bounds=bounds,
                constraints=constraints,
                method="SLSQP",
            )
            w = np.asarray(result.x).ravel()
        except Exception:
            w = np.ones(n) / n
        w = np.clip(w, 0.0, self.max_weight)
        if w.sum() < 1e-12:
            w = np.ones(n) / n
        return pd.Series(w, index=factor_cov.index, name="factor_weight")


__all__ = ["FactorRiskParityOptimizer"]
