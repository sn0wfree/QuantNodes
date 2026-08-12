# coding=utf-8
"""基差动量因子.

基于 spot_basis_daily 的近月/主力基差率:
    - BasisMomentumFactor: 基差率变化方向 (基差走强/走弱)
    - BasisLevelFactor: 基差率水平 (贴水/升水深浅)
"""
from __future__ import annotations

import pandas as pd

from .base import Factor


def _basis_panel(basis_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """基差 long-form → (date × symbol) 面板."""
    df = basis_df[["trade_date", "symbol", value_col]].copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    panel = df.pivot_table(index="trade_date", columns="symbol", values=value_col)
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()


class BasisMomentumFactor(Factor):
    """基差动量因子.

    基差率变化: 基差走强 (贴水加深/升水收窄) → 近月偏强 → 多头.
    研报: 华泰金工基差系列 (基差收敛与趋势).

    参数:
        window: 基差变化窗口
        value_col: 'near_basis_rate' (默认) 或 'dom_basis_rate'
    """

    name = "basis_momentum"

    def __init__(self, window: int = 10, value_col: str = "near_basis_rate"):
        self.window = window
        self.value_col = value_col

    def compute(self, basis_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        panel = _basis_panel(basis_df, self.value_col)
        change = panel.diff(self.window)
        sig = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
        sig[change > 0] = 1.0   # 基差走强 → 多
        sig[change < 0] = -1.0  # 基差走弱 → 空
        return sig


class BasisLevelFactor(Factor):
    """基差水平因子.

    贴水深 (near_basis_rate 显著为负) → 近月低估 → 多头;
    升水深 (显著为正) → 近月高估 → 空头.
    研报: 期现套利/基差回归策略.

    参数:
        window: 平滑窗口
        threshold: 基差率阈值 (绝对)
        value_col: 'near_basis_rate' 或 'dom_basis_rate'
    """

    name = "basis_level"

    def __init__(self, window: int = 5, threshold: float = 0.005,
                 value_col: str = "near_basis_rate"):
        self.window = window
        self.threshold = threshold
        self.value_col = value_col

    def compute(self, basis_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        panel = _basis_panel(basis_df, self.value_col)
        smooth = panel.rolling(self.window).mean()
        sig = pd.DataFrame(0.0, index=smooth.index, columns=smooth.columns)
        # 注意: basis_rate = (S-F)/F, 负 = 现货贴水 = 期货升水 → 基差率负 = 近月相对现货低估
        # 但跨期视角: 负基差率 = 现货<期货 = 近月贴水 → 反向. 采用研报口径:
        # near_basis_rate 显著正 → 现货升水 → 多头信号
        sig[smooth > self.threshold] = 1.0
        sig[smooth < -self.threshold] = -1.0
        return sig
