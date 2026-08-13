# coding=utf-8
"""横截面因子 (Cross-sectional).

截面因子在同一时点比较所有品种, 做多强品种/做空弱品种.
用分位数 (quantile) 作为多空分组阈值, 比 rank 阈值更稳健 (避免边界冲突).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class MomentumXSFactor(Factor):
    """横截面动量因子.

    同一时点所有品种 N 期收益率:
    signal = +1 收益 > q(1-top_pct) (强), -1 收益 < q(top_pct) (弱), 否则 0.

    参数:
        lookback: 收益回看窗口
        top_pct: 多空分组分位 (默认 0.2)
    """

    name = "momentum_xs"

    def __init__(self, lookback: int = 60, top_pct: float = 0.2):
        self.lookback = lookback
        self.top_pct = top_pct

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = prices.pct_change(self.lookback)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for date in ret.index:
            row = ret.loc[date].dropna()
            if len(row) < 3:
                continue
            lo = row.quantile(self.top_pct)
            hi = row.quantile(1 - self.top_pct)
            sig.loc[date, row[row > hi].index] = 1.0
            sig.loc[date, row[row < lo].index] = -1.0
        return sig


class VolatilityXSFactor(Factor):
    """横截面波动率因子.

    低波动品种表现优于高波动 (低波异象):
    signal = +1 波动率 < q(top_pct) (低波), -1 > q(1-top_pct) (高波).

    参数:
        window: 波动窗口
        top_pct: 多空分组分位
    """

    name = "vol_xs"

    def __init__(self, window: int = 20, top_pct: float = 0.2):
        self.window = window
        self.top_pct = top_pct

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = np.log(prices).diff()
        vol = ret.rolling(self.window).std() * np.sqrt(252)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for date in vol.index:
            row = vol.loc[date].dropna()
            if len(row) < 3:
                continue
            lo = row.quantile(self.top_pct)
            hi = row.quantile(1 - self.top_pct)
            sig.loc[date, row[row < lo].index] = 1.0       # 低波 → 多
            sig.loc[date, row[row > hi].index] = -1.0      # 高波 → 空
        return sig


class BasisXSFactor(Factor):
    """横截面期限结构因子.

    backwardation (贴水) 品种收益优于 contango (升水):
    signal = +1 贴水深 (basis > q(1-top_pct)), -1 升水深 (< q(top_pct)).

    参数:
        top_pct: 多空分组分位
    """

    name = "basis_xs"

    def __init__(self, top_pct: float = 0.2):
        self.top_pct = top_pct

    def compute(self, basis: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """basis: (T, N) 每品种的 spread_1_2 面板 (近月-次月)."""
        sig = pd.DataFrame(0.0, index=basis.index, columns=basis.columns)
        for date in basis.index:
            row = basis.loc[date].dropna()
            if len(row) < 3:
                continue
            lo = row.quantile(self.top_pct)
            hi = row.quantile(1 - self.top_pct)
            sig.loc[date, row[row > hi].index] = 1.0      # 贴水深 → 多
            sig.loc[date, row[row < lo].index] = -1.0     # 升水深 → 空
        return sig


class AmihudXSFactor(Factor):
    """横截面非流动性因子.

    非流动性低 (流动性好) 的品种相对表现更优:
    signal = +1 非流动性排名低 (流动性好) → 多; 高 → 空.
    研报: Amihud (2002) 横截面应用.

    参数:
        window: Amihud 计算窗口
        top_pct: 多空分组分位
    """

    name = "amihud_xs"

    def __init__(self, window: int = 20, top_pct: float = 0.2):
        self.window = window
        self.top_pct = top_pct

    def compute(self, prices: pd.DataFrame, volumes: pd.DataFrame,
                **kwargs) -> pd.DataFrame:
        ret = prices.pct_change().abs()
        illiq = ret / volumes.replace(0, np.nan)
        illiq_smooth = illiq.rolling(self.window).mean()
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for date in illiq_smooth.index:
            row = illiq_smooth.loc[date].dropna()
            if len(row) < 3:
                continue
            lo = row.quantile(self.top_pct)
            hi = row.quantile(1 - self.top_pct)
            sig.loc[date, row[row < lo].index] = 1.0      # 流动性好 → 多
            sig.loc[date, row[row > hi].index] = -1.0     # 流动性差 → 空
        return sig
