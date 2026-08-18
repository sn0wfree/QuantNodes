# coding=utf-8
"""基差动量因子.

基于 spot_basis_daily 的近月/主力基差率:
    - BasisMomentumFactor: 基差率变化方向 (基差走强/走弱)
    - BasisLevelFactor: 基差率水平 (贴水/升水深浅)
"""
from __future__ import annotations

import numpy as np
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


class BasisTrendComposite(Factor):
    """基差+趋势复合信号.

    将基差信号作为趋势信号的过滤器/增强器:
    - 基差和趋势方向一致 → 放大信号
    - 基差和趋势方向相反 → 缩小或取消信号
    - 基差极端时 → 独立产生回归信号

    解决了:
    - 纯基差信号在国内商品因持续贴水失效
    - 纯趋势信号在基差极端时容易反转

    参数:
        basis_window: 基差z-score窗口
        basis_threshold: 基差极端阈值
        trend_signal: 外部趋势信号 (DataFrame, date×product)
    """

    name = "basis_trend_composite"

    def __init__(self, basis_window: int = 60, basis_threshold: float = 1.5):
        self.basis_window = basis_window
        self.basis_threshold = basis_threshold

    def compute(self, basis_df: pd.DataFrame, trend_signal: pd.DataFrame = None,
                **kwargs) -> pd.DataFrame:
        """计算复合信号.

        Args:
            basis_df: 含 near_basis_rate 的 long-form
            trend_signal: 趋势信号 (date×product), 值为 +1/-1/0

        Returns:
            DataFrame (date × product), 复合信号
        """
        # 基差 z-score
        df = basis_df[["trade_date", "symbol", "near_basis_rate"]].copy()
        df["symbol"] = df["symbol"].astype(str).str.upper()
        panel = df.pivot_table(index="trade_date", columns="symbol",
                               values="near_basis_rate", aggfunc="first")
        panel.index = pd.to_datetime(panel.index)
        panel = panel.sort_index()

        mean = panel.rolling(self.basis_window, min_periods=20).mean()
        std = panel.rolling(self.basis_window, min_periods=20).std()
        basis_z = (panel - mean) / std.replace(0, np.nan)

        # 基差信号
        basis_sig = pd.DataFrame(0.0, index=basis_z.index, columns=basis_z.columns)
        basis_sig[basis_z < -self.basis_threshold] = 1.0   # 贴水深 → 多
        basis_sig[basis_z > self.basis_threshold] = -1.0    # 升水深 → 空

        if trend_signal is None or trend_signal.empty:
            return basis_sig

        # 对齐趋势信号
        trend = trend_signal.reindex_like(basis_sig).fillna(0)

        # 复合逻辑:
        # 1. 基差极端 → 独立信号
        # 2. 基差不极端 → 用趋势信号, 但基差方向加权
        composite = basis_sig.copy()

        # 非极端区域: 用趋势信号, 基差方向作为权重
        non_extreme = (basis_sig == 0)
        # 基差方向: z > 0 → +1 (升水), z < 0 → -1 (贴水)
        basis_direction = np.sign(basis_z)
        # 权重: 基差方向与趋势一致 → 1.0, 相反 → 0.3
        agreement = (basis_direction == trend).astype(float) * 0.7 + 0.3
        composite[non_extreme] = (trend * agreement)[non_extreme]

        return composite
