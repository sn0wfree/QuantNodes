# coding=utf-8
"""市场结构类因子.

    - NewHighLowFactor: 新高-新低计数 (NH-NL)
    - VolumeReturnResidualFactor: 量价回归残差 (量不能解释的价格变动)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class NewHighLowFactor(Factor):
    """新高-新低因子.

    NH-NL: N 日内创新高天数 - 创新低天数.
    正值 → 上涨结构 (强); 负值 → 下跌结构.
    研报: 经典市场宽度指标 (NH-NL).

    参数:
        window: 新高/新低窗口
        z_threshold: z-score 阈值
    """

    name = "new_high_low"

    def __init__(self, window: int = 60, z_threshold: float = 1.0):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # 当日是否创新高/新低 (过去 window 日, 不含当日)
        rolling_high = prices.rolling(self.window).max().shift(1)
        rolling_low = prices.rolling(self.window).min().shift(1)
        is_high = prices >= rolling_high
        is_low = prices <= rolling_low
        nh = is_high.rolling(self.window).sum()
        nl = is_low.rolling(self.window).sum()
        nhnl = nh - nl
        mean = nhnl.rolling(self.window).mean()
        std = nhnl.rolling(self.window).std()
        z = (nhnl - mean) / std.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[z > self.z_threshold] = 1.0
        sig[z < -self.z_threshold] = -1.0
        return sig


class VolumeReturnResidualFactor(Factor):
    """量价回归残差因子.

    用成交量解释收益的回归残差: 残差正 → 价涨无量 (独立上涨);
    残差负 → 价跌无量 (独立下跌).
    研报: 量价关系深化 (华泰).

    参数:
        window: 回归窗口
        z_threshold: z-score 阈值
    """

    name = "vol_ret_residual"

    def __init__(self, window: int = 30, z_threshold: float = 1.0):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, prices: pd.DataFrame, volumes: pd.DataFrame,
                **kwargs) -> pd.DataFrame:
        ret = prices.pct_change()
        vol = volumes.pct_change()
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for col in prices.columns:
            r = ret[col]
            v = vol[col]
            # 滚动回归: ret = a + b*v + e, 残差 e
            cov = r.rolling(self.window).cov(v)
            var = v.rolling(self.window).var().replace(0, np.nan)
            beta = cov / var
            alpha = r.rolling(self.window).mean() - beta * v.rolling(self.window).mean()
            resid = r - (alpha + beta * v)
            resid_std = resid.rolling(self.window).std()
            z = resid / resid_std.replace(0, np.nan)
            sig.loc[z > self.z_threshold, col] = 1.0
            sig.loc[z < -self.z_threshold, col] = -1.0
        return sig
