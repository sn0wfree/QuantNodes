# coding=utf-8
"""量价背离因子.

基于价格与成交量的配合/背离:
    - PriceVolumeDivergenceFactor: 价创新高量不增 → 背离
    - VolumeConfirmationFactor: 放量+涨 确认趋势
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class PriceVolumeDivergenceFactor(Factor):
    """量价背离因子.

    价格创新高但成交量未创新高 → 上涨动力衰竭 → 反转信号.
    研报: 华泰金工量价背离系列.

    参数:
        window: 新高/新量窗口
    """

    name = "pv_divergence"

    def __init__(self, window: int = 20):
        self.window = window

    def compute(self, prices: pd.DataFrame, volumes: pd.DataFrame,
                **kwargs) -> pd.DataFrame:
        # 对齐列: 只用共同的列
        common_cols = prices.columns.intersection(volumes.columns)
        prices = prices[common_cols]
        volumes = volumes[common_cols]

        price_high = prices.rolling(self.window).max()
        vol_high = volumes.rolling(self.window).max()
        # 价创新高 (今日=窗口最高) 但量未创新高
        price_new_high = (prices >= price_high) & (prices.shift(1) < price_high.shift(1))
        vol_not_high = (volumes < vol_high.shift(1)).fillna(False).astype(bool)
        # 价创新低但量未创新低 → 下跌衰竭
        price_low = prices.rolling(self.window).min()
        vol_low = volumes.rolling(self.window).min()
        price_new_low = (prices <= price_low) & (prices.shift(1) > price_low.shift(1))
        vol_not_low = (volumes > vol_low.shift(1)).fillna(False).astype(bool)

        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[price_new_high & vol_not_high] = -1.0  # 顶背离 → 空
        sig[price_new_low & vol_not_low] = 1.0     # 底背离 → 多
        return sig


class VolumeConfirmationFactor(Factor):
    """量价确认因子.

    放量上涨 → 趋势确认 (多头); 放量下跌 → 空头确认.
    缩量涨/跌 → 趋势减弱.

    参数:
        window: 均量窗口
        mult: 放量倍数
    """

    name = "volume_confirmation"

    def __init__(self, window: int = 20, mult: float = 1.5):
        self.window = window
        self.mult = mult

    def compute(self, prices: pd.DataFrame, volumes: pd.DataFrame,
                **kwargs) -> pd.DataFrame:
        # 对齐列: 只用共同的列
        common_cols = prices.columns.intersection(volumes.columns)
        prices = prices[common_cols]
        volumes = volumes[common_cols]

        vol_ma = volumes.rolling(self.window).mean()
        vol_expand = (volumes > vol_ma.replace(0, np.nan) * self.mult).fillna(False).astype(bool)
        ret = prices.pct_change()
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[vol_expand & (ret > 0)] = 1.0   # 放量上涨 → 多
        sig[vol_expand & (ret < 0)] = -1.0  # 放量下跌 → 空
        return sig
