# coding=utf-8
"""量能退潮/波动率锥因子.

    - VolumeDecayFactor: 量能衰减 (放量后缩量)
    - VolatilityConeFactor: 波动率锥 (短/长波动比)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class VolumeDecayFactor(Factor):
    """量能衰减因子.

    近期量能相对前期高峰衰减程度: 缩量 → 热度退潮 → 趋势减弱.
    研报: 中信期货量能分析.

    参数:
        peak_window: 高峰参考窗口
        decay_window: 近期窗口
        threshold: 衰减阈值 (比值)
    """

    name = "volume_decay"

    def __init__(self, peak_window: int = 60, decay_window: int = 10,
                 threshold: float = 0.5):
        self.peak_window = peak_window
        self.decay_window = decay_window
        self.threshold = threshold

    def compute(self, volumes: pd.DataFrame, **kwargs) -> pd.DataFrame:
        peak = volumes.rolling(self.peak_window).max()
        recent = volumes.rolling(self.decay_window).mean()
        decay = recent / peak.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=volumes.index, columns=volumes.columns)
        sig[decay < self.threshold] = 1.0    # 缩量 → 退潮信号
        sig[decay > 1 / self.threshold] = -1.0
        return sig


class VolatilityConeFactor(Factor):
    """波动率锥因子.

    短周期波动率 / 长周期波动率: 比值高 → 短期波动放大.
    研报: 期权波动率锥 (volatility cone).

    参数:
        short_window: 短波动窗口
        long_window: 长波动窗口
        z_threshold: z-score 阈值
    """

    name = "volatility_cone"

    def __init__(self, short_window: int = 10, long_window: int = 120,
                 z_threshold: float = 1.0):
        self.short_window = short_window
        self.long_window = long_window
        self.z_threshold = z_threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = np.log(prices).diff()
        short_vol = ret.rolling(self.short_window).std() * np.sqrt(252)
        long_vol = ret.rolling(self.long_window).std() * np.sqrt(252)
        cone = short_vol / long_vol.replace(0, np.nan)
        mean = cone.rolling(self.long_window).mean()
        std = cone.rolling(self.long_window).std()
        z = (cone - mean) / std.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[z > self.z_threshold] = 1.0    # 短期波动放大
        sig[z < -self.z_threshold] = -1.0
        return sig
