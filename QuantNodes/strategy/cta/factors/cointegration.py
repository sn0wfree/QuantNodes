# coding=utf-8
"""价差均值回归/协整类因子.

基于跨品种价差的均值回归:
    - SpreadMeanReversionFactor: 价差 z-score 均值回归
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class SpreadMeanReversionFactor(Factor):
    """价差均值回归因子.

    跨品种价差 (如 RB-I, Y-M) 偏离长期均值 → 回归交易:
    z-score 高 → 价差过高 → 空 (价差回归下行); z-score 低 → 多.
    研报: 配对交易/统计套利, 协整价差.

    参数:
        window: 均值/标准差窗口
        z_threshold: z-score 触发阈值
    """

    name = "spread_mean_reversion"

    def __init__(self, window: int = 60, z_threshold: float = 1.5):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, spread: pd.Series, **kwargs) -> pd.Series:
        """spread: 价差序列 (index=DatetimeIndex).

        Returns:
            pd.Series: +1 价差低估 → 多 (价差回升), -1 高估 → 空
        """
        mean = spread.rolling(self.window).mean()
        std = spread.rolling(self.window).std()
        z = (spread - mean) / std.replace(0, np.nan)
        sig = pd.Series(0.0, index=spread.index)
        sig[z < -self.z_threshold] = 1.0    # 低估 → 做多价差
        sig[z > self.z_threshold] = -1.0    # 高估 → 做空价差
        return sig


class PairRatioMeanReversionFactor(Factor):
    """比价均值回归因子.

    两品种比值 A/B 偏离均值 → 回归.
    与 PriceRatioFactor 类似但直接输出 z-score 信号 (无方向假设).

    参数:
        num_product: 分子
        den_product: 分母
        window: 均值窗口
        z_threshold: 触发阈值
    """

    name = "pair_ratio_mr"

    def __init__(self, num_product: str, den_product: str, window: int = 60,
                 z_threshold: float = 1.5):
        self.num_product = num_product
        self.den_product = den_product
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.num_product not in prices.columns or self.den_product not in prices.columns:
            raise KeyError(
                f"products {self.num_product}/{self.den_product} not in panel: "
                f"{list(prices.columns)}"
            )
        ratio = prices[self.num_product] / prices[self.den_product].replace(0, np.nan)
        mean = ratio.rolling(self.window).mean()
        std = ratio.rolling(self.window).std()
        z = (ratio - mean) / std.replace(0, np.nan)
        col = f"{self.num_product}_{self.den_product}_mr"
        sig = pd.DataFrame(0.0, index=prices.index, columns=[col])
        sig.loc[z < -self.z_threshold, col] = 1.0
        sig.loc[z > self.z_threshold, col] = -1.0
        return sig
