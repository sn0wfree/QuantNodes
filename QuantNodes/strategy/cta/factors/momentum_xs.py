# coding=utf-8
"""横截面因子 (Cross-sectional).

截面因子在同一时点比较所有品种, 做多强品种/做空弱品种.
用分位数 (quantile) 作为多空分组阈值, 比 rank 阈值更稳健 (避免边界冲突).

实现说明: 全部向量化 (逐行 quantile(axis=1) + mask), 无 per-row Python 循环.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


def _quantile_signals(
    metric: pd.DataFrame, top_pct: float,
    long_when: str, min_cross: int = 3,
) -> pd.DataFrame:
    """通用横截面分位数多空信号 (向量化).

    Args:
        metric: (T, N) 截面指标面板
        top_pct: 多空分组分位
        long_when: 'high' (指标高做多) | 'low' (指标低做多)
        min_cross: 截面最小品种数 (少于则该日全0)

    Returns:
        信号面板 (+1/-1/0)
    """
    n = metric.notna().sum(axis=1)
    valid = n.ge(min_cross).to_numpy()[:, None]
    lo = metric.quantile(top_pct, axis=1)
    hi = metric.quantile(1 - top_pct, axis=1)
    if long_when == "high":
        long_mask = metric.gt(hi, axis=0).to_numpy() & valid
        short_mask = metric.lt(lo, axis=0).to_numpy() & valid
    else:
        long_mask = metric.lt(lo, axis=0).to_numpy() & valid
        short_mask = metric.gt(hi, axis=0).to_numpy() & valid
    values = np.where(long_mask, 1.0, np.where(short_mask, -1.0, 0.0))
    return pd.DataFrame(values, index=metric.index, columns=metric.columns)


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
        ret = prices.pct_change(self.lookback, fill_method=None)
        return _quantile_signals(ret, self.top_pct, long_when="high")


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
        return _quantile_signals(vol, self.top_pct, long_when="low")


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
        return _quantile_signals(basis, self.top_pct, long_when="high")


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
        ret = prices.pct_change(fill_method=None).abs()
        illiq = ret / volumes.replace(0, np.nan)
        illiq_smooth = illiq.rolling(self.window).mean()
        return _quantile_signals(illiq_smooth, self.top_pct, long_when="low")
