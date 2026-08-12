# coding=utf-8
"""均值回归速度与波动收缩因子.

    - HalfLifeFactor: OU 过程半衰期 (均值回归速度)
    - SqueezeFactor: 波动收缩 (布林带内于 Keltner → 突破前兆)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor
from .indicators import atr


class HalfLifeFactor(Factor):
    """OU 半衰期因子.

    用 Ornstein-Uhlenbeck 回归估计均值回归半衰期:
    Δy = a + b×y_{t-1} + ε, 半衰期 = -ln(2)/ln(1+b).
    半衰期短 → 快速均值回归 → 反转交易; 长 → 趋势.
    研报: 统计套利均值回归速度.

    参数:
        window: 回归窗口
        short_threshold: 短半衰期阈值 (天)
        long_threshold: 长半衰期阈值 (天)
    """

    name = "half_life"

    def __init__(self, window: int = 60, short_threshold: float = 10.0,
                 long_threshold: float = 30.0):
        self.window = window
        self.short = short_threshold
        self.long = long_threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        def _half_life(s):
            s = s.dropna()
            if len(s) < 10:
                return np.nan
            y = s.values
            dy = np.diff(y)
            y_lag = y[:-1]
            # 回归 dy = a + b*y_lag
            n = len(y_lag)
            x = np.column_stack([np.ones(n), y_lag])
            try:
                beta = np.linalg.lstsq(x, dy, rcond=None)[0]
            except np.linalg.LinAlgError:
                return np.nan
            b = beta[1]
            if b >= 0 or b <= -1:
                return np.inf
            hl = -np.log(2) / np.log(1 + b)
            return float(hl)

        hl = prices.rolling(self.window).apply(_half_life, raw=False)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[hl < self.short] = 1.0     # 快速回归 → 反转
        sig[hl > self.long] = -1.0     # 慢回归 → 趋势
        return sig


class SqueezeFactor(Factor):
    """波动收缩因子.

    布林带宽度 < Keltner 通道宽度 → 波动收缩 (squeeze) → 突破前兆.
    收缩后释放 → 突破方向.
    研报: John Carter 波动收缩策略.

    参数:
        window: 布林/Keltner 窗口
        mult_bb: 布林带倍数
        mult_kt: Keltner 倍数
    """

    name = "squeeze"

    def __init__(self, window: int = 20, mult_bb: float = 2.0, mult_kt: float = 1.5):
        self.window = window
        self.mult_bb = mult_bb
        self.mult_kt = mult_kt

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        req = {"high", "low", "close"}
        if not req.issubset(ohlc.columns):
            raise ValueError(f"SqueezeFactor requires columns {req}")
        close = ohlc["close"]
        mid = close.rolling(self.window).mean()
        std = close.rolling(self.window).std()
        bb_width = 2 * self.mult_bb * std
        kt_width = 2 * self.mult_kt * atr(ohlc, self.window)
        squeeze = bb_width < kt_width
        sig = pd.DataFrame(0.0, index=ohlc.index, columns=["squeeze"])
        # 收缩状态标记 + 突破方向
        prev = squeeze.shift(1)
        # 收缩中
        sig.loc[squeeze, "squeeze"] = 1.0
        # 收缩释放且突破
        release = squeeze & ~prev.fillna(False)
        sig.loc[release & (close > mid), "squeeze"] = 1.0
        sig.loc[release & (close < mid), "squeeze"] = -1.0
        return sig
