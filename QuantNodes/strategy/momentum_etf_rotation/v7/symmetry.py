# coding=utf-8
"""Rolling Symmetry 正交 (Klein 2013, 借鉴 v2 cell 8).

[理论基础]
Klein (2013) "对称正交化" 与主成分 PCA 不同:
- PCA: 找最大方差方向 (完全正交, 但牺牲可解释性)
- Symmetry: 对每个原始变量做"对称白化", 输出保持原维度
  - 公式: F̂ = F @ U @ D^{-1/2} @ U.T
  - 性质: cov(F̂) = I (单位阵), 即等方差无相关
  - 优点: 各变量仍对应"原方向", 仅解相关

[风险] 全样本 Symmetry 有 look-ahead. 本实现用滚动 52 周:
  S_t = U_t @ D_t^{-1/2} @ U_t.T  ← 基于截至 t 的 52 周窗口协方差矩阵
  F̂_t = S_t @ F_t                ← 应用到 t 时刻向量

这样严格基于 t 时刻已可见, 不看未来。

[参考] NowCastingHelper.py / .ipynb_checkpoints 中 cell 8 的 MacroFactorHolding.Symmetry
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class RollingSymmetry:
    """滚动 Symmetry 正交 (避免未来 look-ahead).

    参数:
        window: 滚动窗口周数 (默认 52 = 1 年)
        min_periods: 最小可用期数 (默认 26 = 0.5 年). 不足时返回 None.

    示例:
        >>> sym = RollingSymmetry(window=52, min_periods=26)
        >>> orthogonalized = sym.transform_panel(factor_returns)
        >>> # 输出: 与原 DataFrame 同索引, 前 26 行为 NaN
    """

    def __init__(self, window: int = 52, min_periods: int = 26) -> None:
        if window < 4:
            raise ValueError(f"window must be >= 4, got {window}")
        if min_periods > window:
            raise ValueError(f"min_periods ({min_periods}) > window ({window})")
        self.window = window
        self.min_periods = min_periods

    def fit_transform(self, factors: pd.DataFrame, idx: int) -> pd.Series | None:
        """对 idx 时刻做 Symmetry 正交.

        严格基于 t 时刻已可见 (idx 之前的窗口期数).

        Returns:
            pd.Series of orthogonalized values, or None if insufficient history.
        """
        if idx < self.min_periods - 1:
            return None
        if idx >= len(factors):
            raise IndexError(f"idx {idx} >= len(factors) {len(factors)}")

        # 滚动窗口: [max(0, idx-window+1), idx]
        start = max(0, idx - self.window + 1)
        window_data = factors.iloc[start : idx + 1].values

        # Symmetry: F̂ = F @ U @ D^{-1/2} @ U.T
        # 等价: cov(F̂) = I
        cov = np.cov(window_data, rowvar=False)
        # 数值稳定: 处理退化矩阵
        D, U = np.linalg.eigh(cov)
        D = np.maximum(D, 1e-8)
        S = U @ np.diag(D ** -0.5) @ U.T  # (n, n) 变换矩阵

        # 应用到 t 时刻行
        current = factors.iloc[idx].values
        return pd.Series(S @ current, index=factors.columns)

    def transform_panel(self, factors: pd.DataFrame) -> pd.DataFrame:
        """对整个 panel 应用滚动 Symmetry, 输出与原 DataFrame 同 shape.

        前 min_periods-1 行为 NaN. 之后每行做 t-dependent Symmetry.
        """
        result = pd.DataFrame(
            index=factors.index,
            columns=factors.columns,
            dtype=float,
        )
        for i in range(len(factors)):
            row = self.fit_transform(factors, i)
            if row is not None:
                result.iloc[i] = row.values
        return result

    def transform_at(
        self,
        factors: pd.DataFrame,
        as_of: pd.Timestamp,
    ) -> pd.Series | None:
        """基于 as_of 之前的窗口做 Symmetry.

        Returns:
            pd.Series of as_of 时刻的正交化向量, or None.
        """
        as_of_idx = factors.index.get_indexer([as_of], method="ffill")[0]
        if as_of_idx < 0:
            return None
        return self.fit_transform(factors, as_of_idx)

    def fit_transform_full(self, factors: pd.DataFrame) -> pd.DataFrame:
        """全样本 Symmetry (含 look-ahead!) 仅用于 OOS 因子加载后的"还原".

        WARNING: 仅在因子最终 fit 时使用, 不能用于在线预测.
        """
        cov = np.cov(factors.values, rowvar=False)
        D, U = np.linalg.eigh(cov)
        D = np.maximum(D, 1e-8)
        S = U @ np.diag(D ** -0.5) @ U.T
        out = factors.values @ S
        return pd.DataFrame(out, index=factors.index, columns=factors.columns)


__all__ = ["RollingSymmetry"]
