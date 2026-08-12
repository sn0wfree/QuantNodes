# coding=utf-8
"""持仓结构因子.

基于 inter_contract_spread 的 oi_near* 持仓分布:
    - OIConcentrationFactor: 主力合约持仓集中度
    - FrontMonthOIShareFactor: 近月持仓占比
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class OIConcentrationFactor(Factor):
    """持仓集中度因子.

    主力合约 OI / 前 4 合约总 OI:
    集中度高 → 投机资金集中于主力 (趋势性强).
    集中度上升 → 多头信号.
    研报: 国泰君安《持仓结构分析》.

    参数:
        window: 平滑窗口
        z_threshold: z-score 阈值
    """

    name = "oi_concentration"

    def __init__(self, window: int = 20, z_threshold: float = 1.0):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, spread_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """spread_df: inter_contract_spread (含 oi_near/oi_near2/oi_near3/oi_near4)."""
        req = ["oi_near", "oi_near2", "oi_near3", "oi_near4"]
        for c in req:
            if c not in spread_df.columns:
                raise ValueError(f"OIConcentrationFactor requires '{c}' column")
        total_oi = spread_df[req].sum(axis=1).replace(0, np.nan)
        conc = spread_df["oi_near"] / total_oi
        # 转面板: (date × product)
        df = spread_df[["trade_date", "product"]].copy()
        df["conc"] = conc.values
        panel = df.pivot_table(index="trade_date", columns="product", values="conc")
        # 平滑 + z-score
        smooth = panel.rolling(self.window).mean()
        long_ma = smooth.rolling(self.window * 3).mean()
        long_std = smooth.rolling(self.window * 3).std()
        z = (smooth - long_ma) / long_std.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
        sig[z > self.z_threshold] = 1.0
        sig[z < -self.z_threshold] = -1.0
        return sig


class FrontMonthOIShareFactor(Factor):
    """近月持仓占比因子.

    近月 OI / 前 4 合约总 OI:
    近月占比高 → 现货临近交割 → 逼仓/贴水.
    研报: 期现结构分析.

    参数:
        window: 平滑窗口
        z_threshold: z-score 阈值
    """

    name = "front_month_oi"

    def __init__(self, window: int = 20, z_threshold: float = 1.0):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, spread_df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        req = ["oi_near", "oi_near2", "oi_near3", "oi_near4"]
        total_oi = spread_df[req].sum(axis=1).replace(0, np.nan)
        share = spread_df["oi_near"] / total_oi
        df = spread_df[["trade_date", "product"]].copy()
        df["share"] = share.values
        panel = df.pivot_table(index="trade_date", columns="product", values="share")
        smooth = panel.rolling(self.window).mean()
        long_ma = smooth.rolling(self.window * 3).mean()
        long_std = smooth.rolling(self.window * 3).std()
        z = (smooth - long_ma) / long_std.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
        sig[z > self.z_threshold] = 1.0
        sig[z < -self.z_threshold] = -1.0
        return sig
