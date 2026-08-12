# coding=utf-8
"""价格位置类因子.

基于 OHLCV 的价格位置/分布特征:
    - PriceLocationFactor: 收盘价在滚动区间分位
    - VWAPDeviationFactor: 收盘价偏离 VWAP
    - OpenLocationFactor: 开盘价在当日区间位置
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class PriceLocationFactor(Factor):
    """价格位置因子.

    收盘价在滚动 N 日最高-最低区间中的分位 (0-1):
    高分位 → 强势; 低分位 → 弱势.
    研报: 华泰金工价格位置因子.

    参数:
        window: 滚动区间窗口
        z_threshold: z-score 阈值
    """

    name = "price_location"

    def __init__(self, window: int = 20, z_threshold: float = 1.0):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        roll_max = prices.rolling(self.window).max()
        roll_min = prices.rolling(self.window).min()
        loc = (prices - roll_min) / (roll_max - roll_min).replace(0, np.nan)
        # 直接用位置阈值: loc > 0.5+δ → 强 (多), < 0.5-δ → 弱 (空)
        # 对持续趋势有效 (z-score 对单调趋势失效)
        delta = 0.3  # 位置偏移阈值
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[loc > 0.5 + delta] = 1.0
        sig[loc < 0.5 - delta] = -1.0
        return sig


class VWAPDeviationFactor(Factor):
    """VWAP 偏离因子.

    收盘价与当日成交量加权均价 (VWAP) 的偏离:
    close > VWAP → 买方主导 (日内强); close < VWAP → 卖方主导.
    研报: 经典日内交易, 华泰量价.

    参数:
        window: 平滑窗口
        threshold: 偏离阈值 (相对比例)
    """

    name = "vwap_deviation"

    def __init__(self, window: int = 5, threshold: float = 0.005):
        self.window = window
        self.threshold = threshold

    def compute(self, ohlcv: pd.DataFrame, **kwargs) -> pd.DataFrame:
        req = {"open", "high", "low", "close", "volume"}
        if not req.issubset(ohlcv.columns):
            raise ValueError(f"VWAPDeviationFactor requires columns {req}")
        typical = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3
        vwap = (typical * ohlcv["volume"]).rolling(self.window).sum() / \
            ohlcv["volume"].rolling(self.window).sum().replace(0, np.nan)
        deviation = ohlcv["close"] / vwap.replace(0, np.nan) - 1
        sig = pd.DataFrame(0.0, index=ohlcv.index, columns=["vwap_dev"])
        sig.loc[deviation > self.threshold, "vwap_dev"] = 1.0
        sig.loc[deviation < -self.threshold, "vwap_dev"] = -1.0
        return sig


class OpenLocationFactor(Factor):
    """开盘位置因子.

    开盘价在当日 high-low 区间的位置 (0-1):
    高开 (靠近 high) → 隔夜利好; 低开 → 隔夜利空.
    研报: 日内分解 (与 CloseLocation 互补).

    参数:
        window: 平滑窗口
        z_threshold: z-score 阈值
    """

    name = "open_location"

    def __init__(self, window: int = 5, z_threshold: float = 1.0):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        req = {"open", "high", "low"}
        if not req.issubset(ohlc.columns):
            raise ValueError(f"OpenLocationFactor requires columns {req}")
        rng = ohlc["high"] - ohlc["low"]
        loc = (ohlc["open"] - ohlc["low"]) / rng.replace(0, np.nan)
        smooth = loc.rolling(self.window).mean()
        deviation = smooth - 0.5
        mean = deviation.rolling(self.window).mean()
        std = deviation.rolling(self.window).std()
        z = (deviation - mean) / std.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=ohlc.index, columns=["open_loc"])
        sig.loc[z > self.z_threshold, "open_loc"] = 1.0
        sig.loc[z < -self.z_threshold, "open_loc"] = -1.0
        return sig
