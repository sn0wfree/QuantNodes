# coding=utf-8
"""相关性结构因子.

基于全品种收益面板:
    - AvgCorrelationFactor: 品种间平均相关性 (系统性风险)
    - DispersionFactor: 横截面收益离散度 (机会广度)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class AvgCorrelationFactor(Factor):
    """平均相关性因子.

    品种间滚动平均相关: 高 → 系统性风险 (同涨同跌).
    研报: 学术相关性研究, 组合风险管理.

    参数:
        window: 相关窗口
        threshold: 高相关阈值
    """

    name = "avg_correlation"

    def __init__(self, window: int = 60, threshold: float = 0.5):
        self.window = window
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = prices.pct_change(fill_method=None)
        # 逐品种对计算滚动相关, 取均值
        cols = list(prices.columns)
        corr_sum = None
        n_pairs = 0
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c = ret[cols[i]].rolling(self.window).corr(ret[cols[j]])
                corr_sum = c if corr_sum is None else corr_sum.add(c, fill_value=0)
                n_pairs += 1
        avg_corr = corr_sum / n_pairs if corr_sum is not None else \
            pd.Series(0.0, index=prices.index)
        sig = pd.DataFrame(0.0, index=prices.index, columns=["avg_corr"])
        sig.loc[avg_corr > self.threshold, "avg_corr"] = 1.0
        sig.loc[avg_corr < -self.threshold, "avg_corr"] = -1.0
        return sig


class DispersionFactor(Factor):
    """横截面离散度因子.

    品种间日收益的标准差: 高 → 品种分化 (机会多).
    研报: 分散度与市场状态.

    参数:
        window: 平滑窗口
        z_threshold: z-score 阈值
    """

    name = "dispersion"

    def __init__(self, window: int = 20, z_threshold: float = 1.0):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = prices.pct_change(fill_method=None)
        dispersion = ret.std(axis=1)  # 横截面日收益标准差
        smooth = dispersion.rolling(self.window).mean()
        z = (smooth - smooth.rolling(self.window * 3).mean()) / \
            smooth.rolling(self.window * 3).std().replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=prices.index, columns=["dispersion"])
        sig.loc[z > self.z_threshold, "dispersion"] = 1.0
        sig.loc[z < -self.z_threshold, "dispersion"] = -1.0
        return sig
