# coding=utf-8
"""统计特征类因子.

基于价格序列的统计特征, 判别市场状态 (趋势/回归/分布):
    - HurstExponentFactor: Hurst 指数 (H>0.5 趋势, H<0.5 均值回归)
    - VarianceRatioFactor: 方差比检验 (VR>1 动量持续)
    - AutocorrelationFactor: 收益自相关 (正=动量, 负=反转)
    - SkewnessFactor: 收益偏度 (负偏=风险溢价)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """计算时间序列的 Hurst 指数 (R/S 分析).

    H > 0.5: 持续性 (趋势), H < 0.5: 反持续性 (均值回归), H = 0.5: 随机游走.

    Args:
        series: 价格序列
        max_lag: 最大滞后

    Returns:
        float Hurst 指数 (0-1)
    """
    s = series.dropna().values
    if len(s) < max_lag * 2:
        return 0.5
    # 对数价格 (R/S 分析标准输入)
    if (s <= 0).any():
        return 0.5
    s = np.log(s)
    lags = range(2, min(max_lag, len(s) // 2))
    tau = []
    for lag in lags:
        n_blocks = len(s) // lag
        if n_blocks < 1:
            continue
        blocks = s[: n_blocks * lag].reshape(n_blocks, lag)
        # 每个区块: 去均值累计偏差
        cumsum = np.cumsum(blocks - blocks.mean(axis=1, keepdims=True), axis=1)
        block_std = blocks.std(axis=1, ddof=1)
        valid = block_std > 1e-12
        if not valid.any():
            continue
        rs = (cumsum.max(axis=1) - cumsum.min(axis=1)) / block_std
        rs = rs[valid]
        rs = rs[~np.isnan(rs) & ~np.isinf(rs)]
        if len(rs) > 0:
            tau.append((lag, np.mean(rs)))
    if len(tau) < 3:
        return 0.5
    lags_arr = np.array([t[0] for t in tau], dtype=float)
    rs_arr = np.array([t[1] for t in tau], dtype=float)
    if (rs_arr <= 0).any():
        return 0.5
    # Hurst = log(RS) / log(lag) 的斜率
    h = np.polyfit(np.log(lags_arr), np.log(rs_arr), 1)[0]
    return float(np.clip(h, 0.0, 1.0))


def variance_ratio(series: pd.Series, k: int = 5) -> float:
    """方差比 VR(k) = Var(k期收益) / (k × Var(1期收益)).

    VR > 1: 正自相关 (动量), VR < 1: 负自相关 (反转).
    """
    ret = series.pct_change(fill_method=None).dropna()
    if len(ret) < k * 2:
        return 1.0
    var_1 = ret.var(ddof=1)
    if var_1 <= 0:
        return 1.0
    ret_k = series.pct_change(k, fill_method=None).dropna()
    var_k = ret_k.var(ddof=1)
    return float(var_k / (k * var_1))


class HurstExponentFactor(Factor):
    """Hurst 指数因子.

    H > 0.5 → 趋势持续 → 多头; H < 0.5 → 均值回归 → 空头.
    研报: Hurst (1951) R/S 分析, 华泰金工用于趋势判别.

    参数:
        window: 滚动窗口
        max_lag: 最大滞后
        threshold: 偏离 0.5 的触发阈值
    """

    name = "hurst"

    def __init__(self, window: int = 120, max_lag: int = 20, threshold: float = 0.05):
        self.window = window
        self.max_lag = max_lag
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for col in prices.columns:
            h = prices[col].rolling(self.window).apply(
                lambda s: hurst_exponent(s, self.max_lag), raw=False,
            )
            sig.loc[h > 0.5 + self.threshold, col] = 1.0
            sig.loc[h < 0.5 - self.threshold, col] = -1.0
        return sig


class VarianceRatioFactor(Factor):
    """方差比因子.

    VR > 1 → 动量持续 → 多头; VR < 1 → 反转 → 空头.
    研报: Lo & MacKinlay (1988).

    参数:
        window: 滚动窗口
        k: 聚合期数
        threshold: 偏离 1.0 的触发阈值
    """

    name = "var_ratio"

    def __init__(self, window: int = 120, k: int = 5, threshold: float = 0.1):
        self.window = window
        self.k = k
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for col in prices.columns:
            vr = prices[col].rolling(self.window).apply(
                lambda s: variance_ratio(s, self.k), raw=False,
            )
            sig.loc[vr > 1 + self.threshold, col] = 1.0
            sig.loc[vr < 1 - self.threshold, col] = -1.0
        return sig


class AutocorrelationFactor(Factor):
    """收益自相关因子.

    corr(ret, ret_{t-1}) > 0 → 动量; < 0 → 反转.
    研报: 华泰金工动量/反转判别.

    参数:
        window: 滚动窗口
        threshold: 相关阈值
    """

    name = "autocorr"

    def __init__(self, window: int = 60, threshold: float = 0.1):
        self.window = window
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = prices.pct_change(fill_method=None)
        ac = ret.rolling(self.window).apply(
            lambda s: s.autocorr() if len(s.dropna()) > 2 else np.nan, raw=False,
        )
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[ac > self.threshold] = 1.0
        sig[ac < -self.threshold] = -1.0
        return sig


class SkewnessFactor(Factor):
    """收益偏度因子.

    负偏度 → 风险溢价 (做多), 正偏度 → 彩票偏好 (做空).
    研报: Harvey et al. (2018), Amaya et al. (2015) 已实现偏度.

    参数:
        window: 滚动窗口
        threshold: 偏度阈值
    """

    name = "skewness"

    def __init__(self, window: int = 60, threshold: float = 0.3):
        self.window = window
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = prices.pct_change(fill_method=None)
        skew = ret.rolling(self.window).skew()
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[skew < -self.threshold] = 1.0   # 负偏 → 多
        sig[skew > self.threshold] = -1.0   # 正偏 → 空
        return sig


def _fractal_dimension(series: pd.Series, max_lag: int = 10) -> float:
    """估计价格序列的分形维度 D (基于 Hurst: D = 2 - H).

    D ≈ 1.5 随机游走; D > 1.5 反持续; D < 1.5 趋势.
    """
    h = hurst_exponent(series, max_lag)
    return float(2.0 - h)


class FractalDimensionFactor(Factor):
    """分形维度因子.

    D = 2 - Hurst: D < 1.5 → 趋势 (H>0.5) → 顺势; D > 1.5 → 反持续 → 反转.
    研报: 分形市场假说, 与 Hurst 互补.

    参数:
        window: 滚动窗口
        max_lag: Hurst 最大滞后
        threshold: 偏离 1.5 阈值
    """

    name = "fractal_dim"

    def __init__(self, window: int = 120, max_lag: int = 20, threshold: float = 0.05):
        self.window = window
        self.max_lag = max_lag
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for col in prices.columns:
            d = prices[col].rolling(self.window).apply(
                lambda s: _fractal_dimension(s, self.max_lag), raw=False,
            )
            sig.loc[d < 1.5 - self.threshold, col] = 1.0   # 趋势 → 多
            sig.loc[d > 1.5 + self.threshold, col] = -1.0  # 反持续 → 空
        return sig
