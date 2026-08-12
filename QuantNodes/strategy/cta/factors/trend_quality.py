# coding=utf-8
"""趋势质量类因子.

基于线性拟合/二阶导数的趋势纯度与加速度:
    - TrendQualityFactor: R² 线性拟合 (趋势纯度)
    - MomentumSlopeFactor: 动量变化率 (加速度/减速)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


def _r2_fit(series: pd.Series) -> float:
    """滚动窗口内价格对数 vs 时间的 R² (0-1, 1=完美趋势)."""
    s = series.dropna()
    if len(s) < 5:
        return np.nan
    x = np.arange(len(s), dtype=float)
    y = np.log(s.values)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot <= 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


class TrendQualityFactor(Factor):
    """趋势质量因子.

    窗口内对数价格对时间的 R²: R² 高 → 趋势纯 (单边无回撤).
    研报: AQR 趋势质量, 用于过滤假突破.

    参数:
        window: 拟合窗口
        threshold: R² 阈值
    """

    name = "trend_quality"

    def __init__(self, window: int = 30, threshold: float = 0.6):
        self.window = window
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for col in prices.columns:
            r2 = prices[col].rolling(self.window).apply(_r2_fit, raw=False)
            # R² 高 → 趋势纯 → 顺势; R² 低 → 震荡 → 无信号
            # 方向: 用窗口收益方向
            ret = prices[col].pct_change(self.window)
            sig.loc[(r2 > self.threshold) & (ret > 0), col] = 1.0
            sig.loc[(r2 > self.threshold) & (ret < 0), col] = -1.0
        return sig


class MomentumSlopeFactor(Factor):
    """动量斜率因子.

    动量的变化率 (加速度): 动量加速 → 趋势增强; 减速 → 衰竭.
    研报: 华泰金工动量加速/动量崩溃进阶.

    参数:
        lookback: 动量窗口
        slope_window: 加速度窗口
        z_threshold: z-score 阈值
    """

    name = "momentum_slope"

    def __init__(self, lookback: int = 20, slope_window: int = 10,
                 z_threshold: float = 1.0):
        self.lookback = lookback
        self.slope_window = slope_window
        self.z_threshold = z_threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        mom = prices.pct_change(self.lookback)
        slope = mom.diff(self.slope_window)
        mean = slope.rolling(self.slope_window).mean()
        std = slope.rolling(self.slope_window).std()
        z = (slope - mean) / std.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[z > self.z_threshold] = 1.0    # 动量加速 → 多
        sig[z < -self.z_threshold] = -1.0  # 动量衰竭 → 空
        return sig
