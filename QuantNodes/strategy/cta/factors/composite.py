# coding=utf-8
"""复合增强因子.

研报主流复合方法, 基于已有因子/价格:
    - MultiHorizonMomentumFactor: 多周期动量聚合 (3/6/12月加权)
    - SharpeMomentumFactor: 波动率调整动量 (夏普动量)
    - MomentumCrashFilterFactor: 长短期动量差 (动量崩溃规避)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class MultiHorizonMomentumFactor(Factor):
    """多周期动量聚合因子.

    3/6/12月动量加权平均 (近端权重高).
    研报: AQR 多周期动量, 华泰金工.

    参数:
        horizons: 各周期天数 (默认 [63, 126, 252])
        weights: 权重 (默认 [0.5, 0.3, 0.2])
    """

    name = "multi_horizon_mom"

    def __init__(self, horizons: list[int] | None = None, weights: list[float] | None = None):
        self.horizons = horizons or [63, 126, 252]
        self.weights = weights or [0.5, 0.3, 0.2]
        if len(self.horizons) != len(self.weights):
            raise ValueError("horizons and weights must have equal length")

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        rets = {}
        for h, w in zip(self.horizons, self.weights):
            rets[h] = prices.pct_change(h) * w
        combined = sum(rets.values())
        # 归一化: 除以权重和
        combined = combined / sum(self.weights)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[combined > 0] = 1.0
        sig[combined < 0] = -1.0
        return sig


class SharpeMomentumFactor(Factor):
    """夏普动量因子 (波动率调整动量).

    momentum / vol: 单位风险动量, 高 = 高质量趋势.
    研报: 招商证券 CTA 因子体系.

    参数:
        lookback: 动量窗口
        vol_window: 波动窗口
        threshold: 信号阈值
    """

    name = "sharpe_mom"

    def __init__(self, lookback: int = 63, vol_window: int = 20, threshold: float = 0.5):
        self.lookback = lookback
        self.vol_window = vol_window
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = np.log(prices).diff()
        mom = prices.pct_change(self.lookback)
        vol = ret.rolling(self.vol_window).std() * np.sqrt(252)
        sharpe_mom = mom / vol.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[sharpe_mom > self.threshold] = 1.0
        sig[sharpe_mom < -self.threshold] = -1.0
        return sig


class MomentumCrashFilterFactor(Factor):
    """动量崩溃过滤因子.

    短期动量转负且长期仍正 → 趋势崩溃 → 空头/规避多头.
    研报: 华泰金工动量崩溃研究.

    参数:
        short_lookback: 短期动量窗口
        long_lookback: 长期动量窗口
    """

    name = "momentum_crash"

    def __init__(self, short_lookback: int = 20, long_lookback: int = 252):
        self.short_lookback = short_lookback
        self.long_lookback = long_lookback

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        short_mom = prices.pct_change(self.short_lookback)
        long_mom = prices.pct_change(self.long_lookback)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        # 短期转负 & 长期仍正 → 崩溃
        sig[(short_mom < 0) & (long_mom > 0)] = -1.0
        return sig
