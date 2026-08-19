"""量价共振拐点因子.

原理: 价格突破 + 成交量确认 = 真实拐点.

条件:
1. 价格突破: close > 过去N日最高 + K*ATR (上轨)
   或 close < 过去N日最低 - K*ATR (下轨)
2. 成交量确认: volume > 过去M日均量 * 倍数
3. 同时满足 → 拐点信号 (+1 或 -1)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class TurnPointBreakoutFactor(Factor):
    """量价共振拐点因子.

    Args:
        lookback: 通道回看天数 (默认20)
        atr_mult: ATR倍数 (默认0.5)
        vol_mult: 放量倍数 (默认1.5)
        vol_lookback: 成交量均线天数 (默认10)
    """

    name = "turn_point_breakout"

    def __init__(
        self,
        lookback: int = 20,
        atr_mult: float = 0.5,
        vol_mult: float = 1.5,
        vol_lookback: int = 10,
    ):
        self.lookback = lookback
        self.atr_mult = atr_mult
        self.vol_mult = vol_mult
        self.vol_lookback = vol_lookback

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """计算量价共振拐点信号.

        Args:
            ohlc: (T, 4) OHLC面板，列名 ['open','high','low','close','volume']

        Returns:
            pd.DataFrame: (T, N) 信号 {-1, 0, 1}
        """
        if ohlc is None or ohlc.empty:
            return pd.DataFrame()

        close = ohlc['close']
        high = ohlc['high']
        low = ohlc['low']
        volume = ohlc['volume'] if 'volume' in ohlc.columns else None

        # ATR (使用简化版: high-low 均值)
        atr = (high - low).rolling(self.lookback).mean()

        # 通道上轨/下轨 (使用前一日数据)
        upper = close.rolling(self.lookback).max() + self.atr_mult * atr
        lower = close.rolling(self.lookback).min() - self.atr_mult * atr
        upper_prev = upper.shift(1)
        lower_prev = lower.shift(1)

        # 价格突破
        price_break_up = close > upper_prev
        price_break_dn = close < lower_prev

        # 放量确认
        if volume is not None:
            vol_ma = volume.rolling(self.vol_lookback).mean()
            vol_surge = volume > vol_ma * self.vol_mult
        else:
            # 如果没有成交量数据，只用价格突破
            vol_surge = pd.Series(True, index=close.index)

        # 量价共振
        sig = pd.DataFrame(0, index=ohlc.index, columns=ohlc.columns)
        sig[price_break_up & vol_surge] = 1
        sig[price_break_dn & vol_surge] = -1
        return sig


class FastTurnPointFactor(Factor):
    """快速拐点因子 (短窗口, 高敏感).

    用于快速捕捉外部冲击后的趋势启动.
    条件: 突破10日通道 + 放量1.3倍
    """

    name = "fast_turn_point"

    def __init__(
        self,
        lookback: int = 10,
        atr_mult: float = 0.3,
        vol_mult: float = 1.3,
        vol_lookback: int = 5,
    ):
        self.lookback = lookback
        self.atr_mult = atr_mult
        self.vol_mult = vol_mult
        self.vol_lookback = vol_lookback

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return TurnPointBreakoutFactor(
            lookback=self.lookback,
            atr_mult=self.atr_mult,
            vol_mult=self.vol_mult,
            vol_lookback=self.vol_lookback,
        ).compute(ohlc)
