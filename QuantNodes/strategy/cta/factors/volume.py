# coding=utf-8
"""成交量/持仓量因子.

量价因子 (基于主力连续面板的 volume / open_interest):
    - VolumeTrendFactor: 量能趋势 (量升价涨/量缩价跌)
    - VolumeBreakoutFactor: 放量突破 (量 > N 日均量 × 倍数)
    - OpenInterestTrendFactor: 持仓量变化方向
    - OBVFactor: 能量潮 (OBV) 趋势
    - VolOIRatioFactor: 量仓比 (换手活跃度)
    - AmihudFactor: 非流动性 (|ret| / volume)
    - VolumePriceCorrFactor: 量价相关性
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class VolumeTrendFactor(Factor):
    """量能趋势因子.

    价格方向 × 量能方向: 价涨量增 → +1, 价跌量增 → -1.
    signal = sign(价格变化) × sign(量能变化).

    参数:
        window: 量能均线窗口
        price_window: 价格变化窗口
    """

    name = "volume_trend"

    def __init__(self, window: int = 20, price_window: int = 5):
        self.window = window
        self.price_window = price_window

    def compute(self, prices: pd.DataFrame, volumes: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """需要价格面板 + 成交量面板 (同 index/columns)."""
        vol_ma = volumes.rolling(self.window).mean()
        vol_growth = volumes / vol_ma.replace(0, np.nan) - 1
        price_dir = np.sign(prices.pct_change(self.price_window))
        return price_dir * np.sign(vol_growth)


class VolumeBreakoutFactor(Factor):
    """放量突破因子.

    signal = +1 量 > N日均量×倍数 (放量), 否则 0.

    参数:
        window: 量均窗口
        multiplier: 放量倍数 (默认 2)
    """

    name = "volume_breakout"

    def __init__(self, window: int = 20, multiplier: float = 2.0):
        self.window = window
        self.multiplier = multiplier

    def compute(self, volumes: pd.DataFrame, **kwargs) -> pd.DataFrame:
        vol_ma = volumes.rolling(self.window).mean()
        ratio = volumes / vol_ma.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=volumes.index, columns=volumes.columns)
        sig[ratio > self.multiplier] = 1.0
        return sig


class OpenInterestTrendFactor(Factor):
    """持仓量趋势因子.

    持仓量增长方向: 增仓 → +1 (资金流入), 减仓 → -1.

    参数:
        window: OI 变化窗口
    """

    name = "oi_trend"

    def __init__(self, window: int = 5):
        self.window = window

    def compute(self, open_interest: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return np.sign(open_interest.diff(self.window))


class OBVFactor(Factor):
    """能量潮 OBV 趋势因子.

    OBV 累计: 价涨加成交量, 价跌减成交量.
    signal = sign(OBV 的 N 期变化).

    参数:
        window: OBV 变化窗口
    """

    name = "obv"

    def __init__(self, window: int = 20):
        self.window = window

    def compute(self, prices: pd.DataFrame, volumes: pd.DataFrame, **kwargs) -> pd.DataFrame:
        direction = np.sign(prices.diff()).fillna(0)
        obv = (direction * volumes).cumsum()
        return np.sign(obv.diff(self.window))


class VolOIRatioFactor(Factor):
    """量仓比因子 (换手活跃度).

    成交量 / 持仓量: 高比值 = 投机活跃 (短线), 低比值 = 持仓蓄势 (趋势).
    signal = +1 比值高 (活跃), -1 比值低.

    参数:
        window: 平滑窗口
        z_threshold: z-score 阈值
    """

    name = "vol_oi_ratio"

    def __init__(self, window: int = 20, z_threshold: float = 1.0):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, volumes: pd.DataFrame, open_interest: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ratio = volumes / open_interest.replace(0, np.nan)
        mean = ratio.rolling(self.window).mean()
        std = ratio.rolling(self.window).std()
        z = (ratio - mean) / std.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=ratio.index, columns=ratio.columns)
        sig[z > self.z_threshold] = 1.0
        sig[z < -self.z_threshold] = -1.0
        return sig


class AmihudFactor(Factor):
    """Amihud 非流动性因子.

    illiq = |ret| / volume: 高值 = 非流动性 (冲击成本高).
    研报: Amihud (2002). signal = +1 高非流动性, -1 低.

    参数:
        window: 平滑窗口
        z_threshold: z-score 阈值
    """

    name = "amihud"

    def __init__(self, window: int = 20, z_threshold: float = 1.0):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, prices: pd.DataFrame, volumes: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = prices.pct_change().abs()
        illiq = ret / volumes.replace(0, np.nan)
        mean = illiq.rolling(self.window).mean()
        std = illiq.rolling(self.window).std()
        z = (illiq - mean) / std.replace(0, np.nan)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[z > self.z_threshold] = 1.0
        sig[z < -self.z_threshold] = -1.0
        return sig


class VolumePriceCorrFactor(Factor):
    """量价相关性因子.

    滚动 corr(ret, volume): 正相关 = 放量上涨 (趋势确认).
    signal = +1 正相关, -1 负相关.

    参数:
        window: 相关窗口
        threshold: 相关阈值
    """

    name = "vol_price_corr"

    def __init__(self, window: int = 20, threshold: float = 0.3):
        self.window = window
        self.threshold = threshold

    def compute(self, prices: pd.DataFrame, volumes: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ret = prices.pct_change()
        corr = ret.rolling(self.window).corr(volumes)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[corr > self.threshold] = 1.0
        sig[corr < -self.threshold] = -1.0
        return sig


class OIPriceDivergenceFactor(Factor):
    """持仓量-价格背离因子.

    价涨仓减 (OI 降) → 多头获利了结 → 反转预警 (顶背离);
    价跌仓增 (OI 升) → 空头加仓 → 趋势延续 (顺势空).
    研报: 中信期货资金面/持仓结构研究.

    参数:
        window: OI 变化窗口
        price_window: 价格变化窗口
    """

    name = "oi_price_divergence"

    def __init__(self, window: int = 5, price_window: int = 5):
        self.window = window
        self.price_window = price_window

    def compute(self, prices: pd.DataFrame, open_interest: pd.DataFrame,
                **kwargs) -> pd.DataFrame:
        price_ret = prices.pct_change(self.price_window)
        oi_change = open_interest.diff(self.window)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        # 价涨仓减 → 顶背离 → 空
        sig[(price_ret > 0) & (oi_change < 0)] = -1.0
        # 价跌仓增 → 空头加仓 → 顺势空
        sig[(price_ret < 0) & (oi_change > 0)] = -1.0
        # 价涨仓增 → 多头加仓 → 顺势多
        sig[(price_ret > 0) & (oi_change > 0)] = 1.0
        # 价跌仓减 → 空头回补 → 底背离 → 多
        sig[(price_ret < 0) & (oi_change < 0)] = 1.0
        return sig
