# coding=utf-8
"""资金流类因子.

基于 OHLCV 的经典资金流指标:
    - MFIFactor: 资金流量指数 (Money Flow Index)
    - CMFFactor: 蔡金资金流 (Chaikin Money Flow)
    - PVTFactor: 价量趋势 (Price Volume Trend)
    - ForceIndexFactor: 力量指数 (Force Index)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


def _typical_price(ohlc: pd.DataFrame) -> pd.Series:
    """典型价格 (high+low+close)/3."""
    return (ohlc["high"] + ohlc["low"] + ohlc["close"]) / 3


class MFIFactor(Factor):
    """资金流量指数因子.

    MFI = 100 - 100/(1 + 正资金流/负资金流), 类似 RSI 但用量能加权.
    MFI > 80 超买 → 空; MFI < 20 超卖 → 多.
    研报: 经典量价指标 (Wilder 衍生).

    参数:
        window: MFI 窗口
        low_threshold: 超卖线
        high_threshold: 超买线
    """

    name = "mfi"

    def __init__(self, window: int = 14, low_threshold: float = 20.0,
                 high_threshold: float = 80.0):
        self.window = window
        self.low = low_threshold
        self.high = high_threshold

    def compute(self, ohlcv: pd.DataFrame, **kwargs) -> pd.DataFrame:
        req = {"high", "low", "close", "volume"}
        if not req.issubset(ohlcv.columns):
            raise ValueError(f"MFIFactor requires columns {req}")
        tp = _typical_price(ohlcv)
        money_flow = tp * ohlcv["volume"]
        tp_change = tp.diff()
        pos_flow = money_flow.where(tp_change > 0, 0.0)
        neg_flow = money_flow.where(tp_change < 0, 0.0)
        pos_sum = pos_flow.rolling(self.window).sum()
        neg_sum = neg_flow.rolling(self.window).sum()
        # 边界: 全正 → MFI=100; 全负 → MFI=0; 否则标准公式
        ratio = pos_sum / neg_sum.replace(0, np.nan)
        mfi = 100 - 100 / (1 + ratio)
        mfi = mfi.where(neg_sum > 0, 100.0)   # 无负资金流 → 100
        mfi = mfi.where(pos_sum > 0, 0.0)     # 无正资金流 → 0

        sig = pd.DataFrame(0.0, index=ohlcv.index, columns=["mfi"])
        sig.loc[mfi < self.low, "mfi"] = 1.0
        sig.loc[mfi > self.high, "mfi"] = -1.0
        return sig


class CMFFactor(Factor):
    """蔡金资金流因子.

    CMF = Σ(量×CLV) / Σ量, CLV = ((close-low)-(high-close))/(high-low).
    正值 → 资金流入; 负值 → 流出.
    研报: Chaikin (1981).

    参数:
        window: 窗口
        threshold: 信号阈值
    """

    name = "cmf"

    def __init__(self, window: int = 20, threshold: float = 0.1):
        self.window = window
        self.threshold = threshold

    def compute(self, ohlcv: pd.DataFrame, **kwargs) -> pd.DataFrame:
        req = {"high", "low", "close", "volume"}
        if not req.issubset(ohlcv.columns):
            raise ValueError(f"CMFFactor requires columns {req}")
        rng = ohlcv["high"] - ohlcv["low"]
        clv = ((ohlcv["close"] - ohlcv["low"]) -
               (ohlcv["high"] - ohlcv["close"])) / rng.replace(0, np.nan)
        cmf = (clv * ohlcv["volume"]).rolling(self.window).sum() / \
            ohlcv["volume"].rolling(self.window).sum().replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=ohlcv.index, columns=["cmf"])
        sig.loc[cmf > self.threshold, "cmf"] = 1.0
        sig.loc[cmf < -self.threshold, "cmf"] = -1.0
        return sig


class PVTFactor(Factor):
    """价量趋势因子.

    PVT = Σ (ret × volume), 累计量价趋势.
    研报: 经典量价指标 (OBV 的收益加权版).

    参数:
        window: PVT 变化窗口
        z_threshold: z-score 阈值
    """

    name = "pvt"

    def __init__(self, window: int = 20, z_threshold: float = 1.0):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, prices: pd.DataFrame, volumes: pd.DataFrame,
                **kwargs) -> pd.DataFrame:
        ret = prices.pct_change(fill_method=None)
        pvt = (ret * volumes).cumsum()
        change = pvt.diff(self.window)
        mean = change.rolling(self.window).mean()
        std = change.rolling(self.window).std()
        z = (change - mean) / std.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[z > self.z_threshold] = 1.0
        sig[z < -self.z_threshold] = -1.0
        return sig


class ForceIndexFactor(Factor):
    """力量指数因子.

    FI = 量 × (close - prev_close), 衡量价格变动背后的量能支撑.
    正且大 → 强多头; 负且大 → 强空头.
    研报: Elder (1993) 力量指数.

    参数:
        window: 平滑窗口
        z_threshold: z-score 阈值
    """

    name = "force_index"

    def __init__(self, window: int = 13, z_threshold: float = 1.0):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, prices: pd.DataFrame, volumes: pd.DataFrame,
                **kwargs) -> pd.DataFrame:
        price_change = prices.diff()
        fi = price_change * volumes
        fi_smooth = fi.ewm(span=self.window, min_periods=self.window).mean()
        mean = fi_smooth.rolling(self.window).mean()
        std = fi_smooth.rolling(self.window).std()
        z = (fi_smooth - mean) / std.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[z > self.z_threshold] = 1.0
        sig[z < -self.z_threshold] = -1.0
        return sig
