# coding=utf-8
"""风险类因子.

基于收益分布的尾部/下行风险:
    - DownsideVolFactor: 下行波动率 (仅负收益波动)
    - MaxDrawdownFactor: 滚动最大回撤
    - TailRiskFactor: 尾部风险 (VaR 近似)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class DownsideVolFactor(Factor):
    """下行波动率因子.

    仅负收益的标准差 (年化): 衡量下跌风险.
    下行波动率上升 → 风险加大 → 降仓/空头.
    研报: Sortino 体系, 风险预算.

    参数:
        window: 波动窗口
        z_threshold: z-score 阈值
    """

    name = "downside_vol"

    def __init__(self, window: int = 20, z_threshold: float = 1.0):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = prices.pct_change(fill_method=None)
        # 仅负收益: 正收益填充 0, 保持序列连续 (避免 NaN 空洞)
        downside = ret.clip(upper=0)
        dvol = downside.rolling(self.window).std() * np.sqrt(252)
        mean = dvol.rolling(self.window * 3).mean()
        std = dvol.rolling(self.window * 3).std()
        z = (dvol - mean) / std.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[z > self.z_threshold] = 1.0    # 下行风险放大
        sig[z < -self.z_threshold] = -1.0
        return sig


class MaxDrawdownFactor(Factor):
    """滚动最大回撤因子.

    N 日滚动最大回撤: 回撤深 → 风险释放/趋势破坏.
    研报: 风控体系, 回撤控制.

    参数:
        window: 回撤窗口
        z_threshold: z-score 阈值
    """

    name = "max_drawdown"

    def __init__(self, window: int = 60, z_threshold: float = 1.0):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        roll_max = prices.rolling(self.window).max()
        drawdown = prices / roll_max.replace(0, np.nan) - 1  # ≤ 0
        mean = drawdown.rolling(self.window * 2).mean()
        std = drawdown.rolling(self.window * 2).std()
        z = (drawdown - mean) / std.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[z < -self.z_threshold] = 1.0    # 深回撤 → 风险释放 → 多 (反转)
        sig[z > self.z_threshold] = -1.0
        return sig


class TailRiskFactor(Factor):
    """尾部风险因子 (VaR 近似).

    N 日 5% 分位收益 (历史 VaR): 尾部风险上升 → 规避.
    研报: 风险度量 (VaR/ES).

    参数:
        window: VaR 窗口
        quantile: 分位 (默认 0.05)
    """

    name = "tail_risk"

    def __init__(self, window: int = 60, quantile: float = 0.05):
        self.window = window
        self.quantile = quantile

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = prices.pct_change(fill_method=None)
        var = ret.rolling(self.window).quantile(self.quantile)
        # var 通常为负; 越负 = 尾部风险越大
        mean = var.rolling(self.window).mean()
        std = var.rolling(self.window).std()
        z = (var - mean) / std.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[z < -1.0] = 1.0    # 尾部风险加剧 → 规避 (空/减仓信号)
        sig[z > 1.0] = -1.0
        return sig
