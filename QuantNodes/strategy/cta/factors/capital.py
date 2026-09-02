# coding=utf-8
"""资金面因子.

基于主力连续面板的成交量/持仓量:
    - CapitalInflowFactor: 资金沉淀变化 (OI×价格)
    - TurnoverFactor: 成交额趋势 (volume×price)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class CapitalInflowFactor(Factor):
    """资金沉淀因子.

    沉淀资金 = OI × 价格 (近似保证金规模).
    资金持续流入 (沉淀资金上升) → 趋势确认 → 多头.
    研报: 中信期货/国泰君安 CTA 资金面研究.

    参数:
        window: 沉淀资金变化窗口
        threshold: 变化阈值 (z-score)
    """

    name = "capital_inflow"

    def __init__(self, window: int = 20, z_threshold: float = 1.0):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, prices: pd.DataFrame, open_interest: pd.DataFrame,
                **kwargs) -> pd.DataFrame:
        capital = prices * open_interest
        change = capital.pct_change(self.window, fill_method=None)
        mean = change.rolling(self.window).mean()
        std = change.rolling(self.window).std()
        z = (change - mean) / std.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[z > self.z_threshold] = 1.0
        sig[z < -self.z_threshold] = -1.0
        return sig


class TurnoverFactor(Factor):
    """成交额趋势因子.

    成交额 = volume × price. 成交额放大 → 市场活跃.
    研报: 活跃度/热度因子.

    参数:
        window: 成交额趋势窗口
        threshold: 放量倍数阈值
    """

    name = "turnover"

    def __init__(self, window: int = 20, threshold: float = 1.5):
        self.window = window
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame, volumes: pd.DataFrame,
                **kwargs) -> pd.DataFrame:
        turnover = prices * volumes
        ma = turnover.rolling(self.window).mean()
        ratio = turnover / ma.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[ratio > self.threshold] = 1.0
        sig[ratio < 1 / self.threshold] = -1.0
        return sig
