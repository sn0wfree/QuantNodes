# coding=utf-8
"""微观结构类因子 (OHLC 日内分解).

基于连续主力面板的 open/high/low/close 分解:
    - OvernightReturnFactor: 隔夜收益 (开盘-前收)
    - IntradayReturnFactor: 日内收益 (收盘-开盘)
    - OvernightIntradayDivergenceFactor: 隔夜/日内背离
    - GapFactor: 跳空缺口
    - CloseLocationFactor: 收盘位置 (收盘在当日区间的强弱)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class OvernightReturnFactor(Factor):
    """隔夜收益因子.

    overnight = open / pre_close - 1.
    正隔夜 → 隔夜信息偏多 → 多头.
    研报: 华泰金工隔夜收益系列.

    参数:
        window: 平滑窗口
        threshold: 信号阈值
    """

    name = "overnight"

    def __init__(self, window: int = 5, threshold: float = 0.0):
        self.window = window
        self.threshold = threshold

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if "open" not in ohlc.columns:
            raise ValueError("OvernightReturnFactor requires 'open' column")
        pre_close = ohlc["close"].shift(1)
        overnight = ohlc["open"] / pre_close.replace(0, np.nan) - 1
        smooth = overnight.rolling(self.window).mean()
        sig = pd.DataFrame(0.0, index=ohlc.index, columns=["overnight"])
        sig.loc[smooth > self.threshold, "overnight"] = 1.0
        sig.loc[smooth < -self.threshold, "overnight"] = -1.0
        return sig


class IntradayReturnFactor(Factor):
    """日内收益因子.

    intraday = close / open - 1.
    正日内 → 日内动能强 → 多头.
    研报: 华泰金工日内动能.

    参数:
        window: 平滑窗口
        threshold: 信号阈值
    """

    name = "intraday"

    def __init__(self, window: int = 5, threshold: float = 0.0):
        self.window = window
        self.threshold = threshold

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if "open" not in ohlc.columns or "close" not in ohlc.columns:
            raise ValueError("IntradayReturnFactor requires 'open' and 'close' columns")
        intraday = ohlc["close"] / ohlc["open"].replace(0, np.nan) - 1
        smooth = intraday.rolling(self.window).mean()
        sig = pd.DataFrame(0.0, index=ohlc.index, columns=["intraday"])
        sig.loc[smooth > self.threshold, "intraday"] = 1.0
        sig.loc[smooth < -self.threshold, "intraday"] = -1.0
        return sig


class OvernightIntradayDivergenceFactor(Factor):
    """隔夜/日内背离因子.

    隔夜与日内收益方向相反 → 市场分歧.
    正背离: 隔夜涨/日内跌 (冲高回落) → 空头.
    研报: 华泰金工量价背离.

    参数:
        window: 平滑窗口
    """

    name = "oi_divergence"

    def __init__(self, window: int = 5):
        self.window = window

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if "open" not in ohlc.columns:
            raise ValueError("OvernightIntradayDivergenceFactor requires 'open' column")
        pre_close = ohlc["close"].shift(1)
        overnight = ohlc["open"] / pre_close.replace(0, np.nan) - 1
        intraday = ohlc["close"] / ohlc["open"].replace(0, np.nan) - 1
        div = overnight.rolling(self.window).mean() * intraday.rolling(self.window).mean()
        sig = pd.DataFrame(0.0, index=ohlc.index, columns=["divergence"])
        sig.loc[div < 0, "divergence"] = 1.0   # 背离存在 → 分歧信号
        return sig


class GapFactor(Factor):
    """跳空缺口因子.

    gap = open - pre_close (绝对或相对).
    正跳空 → 向上缺口 → 多头动能.

    参数:
        window: 平滑窗口
        threshold: 缺口阈值 (相对前收比例)
    """

    name = "gap"

    def __init__(self, window: int = 5, threshold: float = 0.002):
        self.window = window
        self.threshold = threshold

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if "open" not in ohlc.columns:
            raise ValueError("GapFactor requires 'open' column")
        pre_close = ohlc["close"].shift(1)
        gap = (ohlc["open"] - pre_close) / pre_close.replace(0, np.nan)
        smooth = gap.rolling(self.window).mean()
        sig = pd.DataFrame(0.0, index=ohlc.index, columns=["gap"])
        sig.loc[smooth > self.threshold, "gap"] = 1.0
        sig.loc[smooth < -self.threshold, "gap"] = -1.0
        return sig


class CloseLocationFactor(Factor):
    """收盘位置因子.

    CLV = (close - low) / (high - low), 0-1.
    收盘靠近高位 → 多头占优.
    研报: 经典日内强弱指标.

    参数:
        window: 平滑窗口
        threshold: CLV 阈值
    """

    name = "close_location"

    def __init__(self, window: int = 5, threshold: float = 0.5):
        self.window = window
        self.threshold = threshold

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if not {"high", "low", "close"}.issubset(ohlc.columns):
            raise ValueError("CloseLocationFactor requires high/low/close columns")
        rng = ohlc["high"] - ohlc["low"]
        clv = (ohlc["close"] - ohlc["low"]) / rng.replace(0, np.nan)
        smooth = clv.rolling(self.window).mean()
        sig = pd.DataFrame(0.0, index=ohlc.index, columns=["clv"])
        sig.loc[smooth > 0.5 + (self.threshold - 0.5), "clv"] = 1.0
        sig.loc[smooth < 0.5 - (self.threshold - 0.5), "clv"] = -1.0
        return sig
