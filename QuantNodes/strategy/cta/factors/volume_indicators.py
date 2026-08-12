# coding=utf-8
"""量能技术指标类因子.

基于 OHLCV 的经典量能指标:
    - VolumeRatioFactor: 量比 (当日量/过去均量)
    - VolumeMACrossFactor: 量能均线交叉
    - EaseOfMovementFactor: 价格变动难易度 (EMV)
    - KlingerOscillatorFactor: 克凌格量能振荡器
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class VolumeRatioFactor(Factor):
    """量比因子.

    量比 = 当日量 / 过去 N 日均量: >1 放量, <1 缩量.
    与 VolumeBreakout 类似但连续 (不分档).
    研报: 经典量比指标.

    参数:
        window: 均量窗口
        high_threshold: 放量阈值
        low_threshold: 缩量阈值
    """

    name = "volume_ratio"

    def __init__(self, window: int = 5, high_threshold: float = 1.5,
                 low_threshold: float = 0.5):
        self.window = window
        self.high = high_threshold
        self.low = low_threshold

    def compute(self, volumes: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ma = volumes.rolling(self.window).mean()
        ratio = volumes / ma.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=volumes.index, columns=volumes.columns)
        sig[ratio > self.high] = 1.0
        sig[ratio < self.low] = -1.0
        return sig


class VolumeMACrossFactor(Factor):
    """量能均线交叉因子.

    短期量均线 > 长期量均线 → 量能放大 (活跃); 反之缩量.
    研报: 量能趋势技术分析.

    参数:
        short_window: 短期量均线
        long_window: 长期量均线
    """

    name = "volume_ma_cross"

    def __init__(self, short_window: int = 5, long_window: int = 20):
        self.short_window = short_window
        self.long_window = long_window

    def compute(self, volumes: pd.DataFrame, **kwargs) -> pd.DataFrame:
        short = volumes.rolling(self.short_window).mean()
        long = volumes.rolling(self.long_window).mean()
        sig = pd.DataFrame(0.0, index=volumes.index, columns=volumes.columns)
        sig[short > long] = 1.0
        sig[short < long] = -1.0
        return sig


class EaseOfMovementFactor(Factor):
    """价格变动难易度因子.

    EMV = ((high+low)/2 - (prev_high+prev_low)/2) × (high-low) / volume
    正值大 → 价格易涨 (量小推动); 负值大 → 易跌.
    研报: Arm's Ease of Movement.

    参数:
        window: 平滑窗口
        z_threshold: z-score 阈值
    """

    name = "ease_of_movement"

    def __init__(self, window: int = 14, z_threshold: float = 1.0):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, ohlcv: pd.DataFrame, **kwargs) -> pd.DataFrame:
        req = {"high", "low", "volume"}
        if not req.issubset(ohlcv.columns):
            raise ValueError(f"EaseOfMovementFactor requires columns {req}")
        mid = (ohlcv["high"] + ohlcv["low"]) / 2
        mid_change = mid.diff()
        box_ratio = (ohlcv["high"] - ohlcv["low"]) / ohlcv["volume"].replace(0, np.nan)
        emv = mid_change * box_ratio
        emv_smooth = emv.rolling(self.window).mean()
        mean = emv_smooth.rolling(self.window).mean()
        std = emv_smooth.rolling(self.window).std()
        z = (emv_smooth - mean) / std.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=ohlcv.index, columns=["emv"])
        sig.loc[z > self.z_threshold, "emv"] = 1.0
        sig.loc[z < -self.z_threshold, "emv"] = -1.0
        return sig


class KlingerOscillatorFactor(Factor):
    """克凌格量能振荡器因子.

    KVO = 短EMA(量能趋势) - 长EMA(量能趋势), 量能趋势基于价格方向.
    KVO 上穿 0 → 多头; 下穿 → 空头.
    研报: Klinger (1980).

    参数:
        short_window: 短期 EMA
        long_window: 长期 EMA
    """

    name = "klinger"

    def __init__(self, short_window: int = 34, long_window: int = 55):
        self.short_window = short_window
        self.long_window = long_window

    def compute(self, ohlcv: pd.DataFrame, **kwargs) -> pd.DataFrame:
        req = {"high", "low", "close", "volume"}
        if not req.issubset(ohlcv.columns):
            raise ValueError(f"KlingerOscillatorFactor requires columns {req}")
        tp = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3
        trend = np.sign(tp.diff())
        # 量能趋势: 方向为 0 时取前值
        trend = trend.replace(0, np.nan).ffill().fillna(0)
        dm = ohlcv["high"].diff() - ohlcv["low"].diff()
        cm = dm.rolling(34).sum() + dm.rolling(55).sum().abs() + dm.rolling(13).sum().abs()
        vf = ohlcv["volume"] * trend * (dm.abs() / cm.replace(0, np.nan) * 100)
        kvo = vf.ewm(span=self.short_window).mean() - vf.ewm(span=self.long_window).mean()
        sig = pd.DataFrame(0.0, index=ohlcv.index, columns=["klinger"])
        # 上穿/下穿 0
        prev = kvo.shift(1)
        sig.loc[(kvo > 0) & (prev <= 0), "klinger"] = 1.0
        sig.loc[(kvo < 0) & (prev >= 0), "klinger"] = -1.0
        return sig
