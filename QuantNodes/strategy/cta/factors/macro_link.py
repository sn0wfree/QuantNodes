# coding=utf-8
"""宏观联动因子.

商品期货与宏观指标的联动关系 (VIX/DXY/利率):
    - VIXLinkFactor: VIX 变化对商品收益的影响 (风险偏好)
    - DXYLinkFactor: 美元指数对商品的影响 (负相关, 商品以美元计价)
    - MacroStateFactor: 宏观状态 (risk-on/off) 对商品方向的择时
"""
from __future__ import annotations

import pandas as pd

from .base import Factor


class VIXLinkFactor(Factor):
    """VIX 联动因子.

    VIX 上升 (恐慌) → 商品承压 (尤其工业品); VIX 下降 → 风险偏好回升.
    信号: VIX 变化方向 × 品种敏感度符号.

    参数:
        window: VIX 变化窗口
        threshold: 变化阈值 (VIX 绝对变动)
    """

    name = "vix_link"

    def __init__(self, window: int = 5, threshold: float = 1.0):
        self.window = window
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame, vix: pd.Series, **kwargs) -> pd.DataFrame:
        """vix: 与 prices 对齐的 VIX 序列."""
        vix_aligned = vix.reindex(prices.index).ffill()
        vix_change = vix_aligned.diff(self.window)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        # VIX 大涨 → 风险 off → 商品空头 (工业品敏感)
        sig[vix_change > self.threshold] = -1.0
        sig[vix_change < -self.threshold] = 1.0
        return sig


class DXYLinkFactor(Factor):
    """美元指数联动因子.

    美元走强 (DXY 升) → 以美元计价商品承压 (负相关).
    研报: 美元-商品跷跷板.

    参数:
        window: DXY 变化窗口
        threshold: 变化阈值 (DXY 绝对变动)
    """

    name = "dxy_link"

    def __init__(self, window: int = 5, threshold: float = 0.3):
        self.window = window
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame, dxy: pd.Series, **kwargs) -> pd.DataFrame:
        dxy_aligned = dxy.reindex(prices.index).ffill()
        dxy_change = dxy_aligned.diff(self.window)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        # 美元升 → 商品空
        sig[dxy_change > self.threshold] = -1.0
        sig[dxy_change < -self.threshold] = 1.0
        return sig


class MacroStateFactor(Factor):
    """宏观状态因子.

    综合 VIX 水平 + 趋势判断 risk-on/off 状态.
    VIX 高位 + 上升 → risk-off → 商品空头 (除避险金属).

    参数:
        vix_window: VIX 均线窗口
        high_threshold: VIX 高位阈值
        trend_window: VIX 趋势窗口
    """

    name = "macro_state"

    def __init__(self, vix_window: int = 20, high_threshold: float = 25.0,
                 trend_window: int = 5):
        self.vix_window = vix_window
        self.high_threshold = high_threshold
        self.trend_window = trend_window

    def compute(self, prices: pd.DataFrame, vix: pd.Series, **kwargs) -> pd.DataFrame:
        vix_aligned = vix.reindex(prices.index).ffill()
        vix_ma = vix_aligned.rolling(self.vix_window).mean()
        vix_trend = vix_aligned - vix_aligned.shift(self.trend_window)

        risk_off = (vix_ma > self.high_threshold) & (vix_trend > 0)
        risk_on = (vix_ma < self.high_threshold * 0.7) & (vix_trend < 0)

        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[risk_off] = -1.0
        sig[risk_on] = 1.0
        return sig
