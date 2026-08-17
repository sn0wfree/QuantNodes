# coding=utf-8
"""期限结构因子.

基于 inter_contract_spread / term_structure_daily 派生表:
    - BasisFactor: 近月-次月价差 (展期收益率近似)
    - RollYieldFactor: 展期收益率年化
    - TermSlopeFactor: 期限结构斜率 (近月 vs 远月, 时间归一)
    - TermCurvatureFactor: 期限结构曲率 (二阶信息)
    - NearBasisFactor: 近月基差率 (现货-近月, 用 spot_basis_daily)

数据来源: DuckDB 的 inter_contract_spread / spot_basis_daily.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class BasisFactor(Factor):
    """跨期价差因子 (近月 - 次月).

    正值 (backwardation): 近月 > 次月 → 贴水 → 多头信号
    负值 (contango): 近月 < 次月 → 升水 → 空头信号
    """

    name = "basis"

    def compute(self, spread_df: pd.DataFrame, **kwargs) -> pd.Series:
        """从 inter_contract_spread 计算.

        Args:
            spread_df: 含 trade_date/product/spread_1_2 的 long-form

        Returns:
            pd.Series (multi-index: date × product), 值 = spread_1_2
        """
        df = spread_df.set_index(["trade_date", "product"])["spread_1_2"]
        return df.sort_index()


class RollYieldFactor(Factor):
    """展期收益率 (年化).

    roll_yield = (F1 - F2) / F1 × 252 / days_gap
    正值 → backwardation → 多头持有有展期收益
    """

    name = "roll_yield"

    def __init__(self, days_gap: int = 30):
        self.days_gap = days_gap

    def compute(self, spread_df: pd.DataFrame, **kwargs) -> pd.Series:
        df = spread_df.copy()
        df["roll_yield"] = (
            df["spread_1_2"] / df["close_near"] * 252.0 / self.days_gap
        )
        return df.set_index(["trade_date", "product"])["roll_yield"].sort_index()


def basis_to_signal(basis: pd.Series, threshold: float = 0.0) -> pd.Series:
    """跨期价差 → 方向信号."""
    return basis.apply(lambda x: 1.0 if x > threshold else (-1.0 if x < -threshold else 0.0))


class TermSlopeFactor(Factor):
    """期限结构斜率因子.

    斜率 = (F_far - F_near) / (T_far - T_near) (时间归一).
    正斜率 = contango 陡峭, 负斜率 = backwardation 陡峭.
    研报: 华泰金工《商品期货期限结构因子》.

    signal = +1 斜率 z-score 高 (陡峭升水), -1 低 (陡峭贴水).
    """

    name = "term_slope"

    def __init__(self, z_threshold: float = 1.0):
        self.z_threshold = z_threshold

    def compute(self, spread_df: pd.DataFrame, **kwargs) -> pd.Series:
        """从 inter_contract_spread 计算, 需含 close_near/close_near4 与期限差.

        近似: slope = (F4 - F1) / F1 (用 1-4 月价差, 期限差约 3 个月).
        """
        df = spread_df.copy()
        df["slope"] = (df["close_near4"] - df["close_near"]) / df["close_near"].replace(0, np.nan)
        return df.set_index(["trade_date", "product"])["slope"].sort_index()


class TermCurvatureFactor(Factor):
    """期限结构曲率因子 (二阶信息).

    曲率 = F1 - 2×F2 + F3 (凸性).
    研报: 华泰金工《期限结构曲率》. 曲率 > 0 → 中期相对偏高.

    signal = +1 曲率 z-score 高, -1 低.
    """

    name = "term_curvature"

    def __init__(self, z_threshold: float = 1.0):
        self.z_threshold = z_threshold

    def compute(self, spread_df: pd.DataFrame, **kwargs) -> pd.Series:
        df = spread_df.copy()
        df["curvature"] = (
            df["close_near"] - 2 * df["close_near2"] + df["close_near3"]
        ) / df["close_near"].replace(0, np.nan)
        return df.set_index(["trade_date", "product"])["curvature"].sort_index()


class NearBasisFactor(Factor):
    """近月基差率因子 (现货-近月).

    基差率 = (S - F1) / F1. 正 = 现货升水 (近月贴水).
    研报: 期现基差收敛策略常用.
    """

    name = "near_basis"

    def __init__(self, z_threshold: float = 1.0):
        self.z_threshold = z_threshold

    def compute(self, basis_df: pd.DataFrame, **kwargs) -> pd.Series:
        """从 spot_basis_daily 计算, 需含 near_basis_rate 列."""
        df = basis_df.copy()
        df["symbol"] = df["symbol"].astype(str).str.upper()
        return df.set_index(["trade_date", "symbol"])["near_basis_rate"].sort_index()


class BasisConvergenceFactor(Factor):
    """基差收敛因子 (均值回归).

    基差率偏离历史均值过远时, 做回归方向:
    - z < -z_threshold: 基差极度负 (现货贴水) → 基差将收窄 → 多头
    - z > z_threshold: 基差极度正 (现货升水) → 基差将收窄 → 空头

    与 NearBasisFactor 的区别:
    - NearBasisFactor: 方向性信号 (贴水→多头), 国内商品因持续贴水失效
    - BasisConvergenceFactor: 均值回归信号 (极端→回归), 只在极端位置交易

    参数:
        window: z-score 计算窗口
        z_threshold: 极端阈值 (默认2.0 = 2倍标准差)
        value_col: 基差率列
    """

    name = "basis_convergence"

    def __init__(self, window: int = 60, z_threshold: float = 2.0,
                 value_col: str = "near_basis_rate"):
        self.window = window
        self.z_threshold = z_threshold
        self.value_col = value_col

    def compute(self, basis_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df = basis_df[["trade_date", "symbol", self.value_col]].copy()
        df["symbol"] = df["symbol"].astype(str).str.upper()
        panel = df.pivot_table(index="trade_date", columns="symbol",
                               values=self.value_col, aggfunc="first")
        panel.index = pd.to_datetime(panel.index)
        panel = panel.sort_index()

        mean = panel.rolling(self.window, min_periods=20).mean()
        std = panel.rolling(self.window, min_periods=20).std()
        z = (panel - mean) / std.replace(0, np.nan)

        sig = pd.DataFrame(0.0, index=z.index, columns=z.columns)
        sig[z < -self.z_threshold] = 1.0   # 极度贴水 → 基差收窄 → 多头
        sig[z > self.z_threshold] = -1.0   # 极度升水 → 基差收窄 → 空头
        return sig


class CalendarSpreadMeanReversion(Factor):
    """跨期价差均值回归因子.

    近月-次月价差 (spread_1_2) 做 z-score 均值回归:
    - z < -z_threshold: 价差极度负 (近月贴水) → 做多价差 (买近卖远)
    - z > z_threshold: 价差极度正 (近月升水) → 做空价差 (卖近买远)

    与 BasisFactor 的区别:
    - BasisFactor: 方向性信号 (贴水→多头), 不做均值回归
    - CalendarSpreadMR: 均值回归信号, 只在极端位置交易

    与 BasisConvergenceFactor 的区别:
    - BasisConvergence: 用 spot-futures 基差 (需要现货)
    - CalendarSpreadMR: 用 near-far 跨期价差 (纯期货, 可执行)

    参数:
        window: 均值/标准差窗口
        z_threshold: z-score 触发阈值
    """

    name = "calendar_spread_mr"

    def __init__(self, window: int = 60, z_threshold: float = 1.5):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, spread_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """从 inter_contract_spread 计算.

        Args:
            spread_df: 含 trade_date/product/spread_1_2/close_near 的 long-form

        Returns:
            DataFrame (date × product), 信号: +1 做多价差, -1 做空价差
        """
        df = spread_df[["trade_date", "product", "spread_1_2", "close_near"]].copy()
        # 归一化价差: spread / near_price
        df["norm_spread"] = df["spread_1_2"] / df["close_near"].replace(0, np.nan)
        panel = df.pivot_table(index="trade_date", columns="product",
                               values="norm_spread", aggfunc="first")
        panel.index = pd.to_datetime(panel.index)
        panel = panel.sort_index()

        mean = panel.rolling(self.window, min_periods=20).mean()
        std = panel.rolling(self.window, min_periods=20).std()
        z = (panel - mean) / std.replace(0, np.nan)

        sig = pd.DataFrame(0.0, index=z.index, columns=z.columns)
        sig[z < -self.z_threshold] = 1.0   # 价差极低 → 做多价差
        sig[z > self.z_threshold] = -1.0   # 价差极高 → 做空价差
        return sig
