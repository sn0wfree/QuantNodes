# coding=utf-8
"""趋势进阶因子.

基于 OHLCV 的趋势强度/方向进阶指标:
    - AroonFactor: 阿里翁指标
    - VortexFactor: 涡旋指标
    - LinearSlopeFactor: 线性回归斜率
    - TRIXFactor: 三重指数平滑
    - ChandelierExitFactor: 吊灯止损 (ATR 回落)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor
from .indicators import atr


class AroonFactor(Factor):
    """阿里翁趋势因子.

    Aroon_Up = 100×(n - 距创新高天数)/n; Aroon_Down 同理.
    差值 > 0 → 上涨趋势; 上穿 0 → 多.
    研报: 经典趋势强度指标 (Chande).

    参数:
        window: 窗口
    """

    name = "aroon"

    def __init__(self, window: int = 25):
        self.window = window

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        # 距创新高/新低天数: 窗口内最高/最低位置
        def _days_since_high(s):
            return self.window - 1 - int(np.argmax(s)) if len(s) == self.window else np.nan

        def _days_since_low(s):
            return self.window - 1 - int(np.argmin(s)) if len(s) == self.window else np.nan

        up = 100 * (self.window - prices.rolling(self.window).apply(
            _days_since_high, raw=True,
        )) / self.window
        down = 100 * (self.window - prices.rolling(self.window).apply(
            _days_since_low, raw=True,
        )) / self.window
        diff = up - down
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        # Aroon 差值: >50 上趋势主导 → 多; < -50 → 空
        sig[diff > 50] = 1.0
        sig[diff < -50] = -1.0
        return sig


class VortexFactor(Factor):
    """涡旋指标因子.

    VI+ = Σ|high - prev_low| / ΣTR; VI- = Σ|low - prev_high| / ΣTR.
    VI+ > VI- → 上涨趋势 → 多.
    研报: 涡旋指标 (Etienne & Betsy).

    参数:
        window: 窗口
    """

    name = "vortex"

    def __init__(self, window: int = 14):
        self.window = window

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        req = {"high", "low", "close"}
        if not req.issubset(ohlc.columns):
            raise ValueError(f"VortexFactor requires columns {req}")
        prev_high = ohlc["high"].shift(1)
        prev_low = ohlc["low"].shift(1)
        prev_close = ohlc["close"].shift(1)
        vm_plus = (ohlc["high"] - prev_low).abs().rolling(self.window).sum()
        vm_minus = (ohlc["low"] - prev_high).abs().rolling(self.window).sum()
        tr = pd.concat([
            ohlc["high"] - ohlc["low"],
            (ohlc["high"] - prev_close).abs(),
            (ohlc["low"] - prev_close).abs(),
        ], axis=1).max(axis=1).rolling(self.window).sum().replace(0, np.nan)
        vi_plus = vm_plus / tr
        vi_minus = vm_minus / tr
        sig = pd.DataFrame(0.0, index=ohlc.index, columns=["vortex"])
        prev_p, prev_m = vi_plus.shift(1), vi_minus.shift(1)
        sig.loc[(vi_plus > vi_minus) & (prev_p <= prev_m), "vortex"] = 1.0
        sig.loc[(vi_plus < vi_minus) & (prev_p >= prev_m), "vortex"] = -1.0
        return sig


class LinearSlopeFactor(Factor):
    """线性回归斜率因子.

    窗口内对数价格对时间的回归斜率 (方向 + 强度).
    斜率正 → 上行趋势; 斜率负 → 下行.
    研报: 线性回归趋势分析.

    参数:
        window: 回归窗口
        z_threshold: z-score 阈值
    """

    name = "linear_slope"

    def __init__(self, window: int = 20, z_threshold: float = 1.0):
        self.window = window
        self.z_threshold = z_threshold

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        def _slope(s):
            s = s.dropna()
            if len(s) < 5:
                return np.nan
            x = np.arange(len(s), dtype=float)
            y = np.log(s.values)
            return float(np.polyfit(x, y, 1)[0])

        slope = prices.rolling(self.window).apply(_slope, raw=False)
        # 斜率方向直接作信号: 恒定斜率时 z-score 无意义, 用符号判别
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[slope > 0] = 1.0
        sig[slope < 0] = -1.0
        return sig


class TRIXFactor(Factor):
    """TRIX 三重指数平滑因子.

    TRIX = EMA(EMA(EMA(price))) 的 1 期变化率.
    TRIX 上穿 0 → 多; 下穿 → 空.
    研报: 经典三重指数平滑 (Hutson).

    参数:
        window: EMA 窗口
    """

    name = "trix"

    def __init__(self, window: int = 15):
        self.window = window

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        ema1 = prices.ewm(span=self.window, adjust=False).mean()
        ema2 = ema1.ewm(span=self.window, adjust=False).mean()
        ema3 = ema2.ewm(span=self.window, adjust=False).mean()
        trix = ema3.pct_change() * 100
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        prev = trix.shift(1)
        sig[(trix > 0) & (prev <= 0)] = 1.0
        sig[(trix < 0) & (prev >= 0)] = -1.0
        return sig


class ChandelierExitFactor(Factor):
    """吊灯止损因子.

    多头止损 = 窗口最高 - N×ATR; 空头止损 = 窗口最低 + N×ATR.
    价格跌破吊灯 → 多头离场 (空信号).
    研报: 吊灯止损 (LeBeau).

    参数:
        window: 最高/最低窗口
        atr_window: ATR 窗口
        mult: ATR 倍数
    """

    name = "chandelier"

    def __init__(self, window: int = 22, atr_window: int = 14, mult: float = 3.0):
        self.window = window
        self.atr_window = atr_window
        self.mult = mult

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        req = {"high", "low", "close"}
        if not req.issubset(ohlc.columns):
            raise ValueError(f"ChandelierExitFactor requires columns {req}")
        roll_high = ohlc["high"].rolling(self.window).max()
        roll_low = ohlc["low"].rolling(self.window).min()
        a = atr(ohlc, self.atr_window)
        long_exit = roll_high - self.mult * a
        short_exit = roll_low + self.mult * a
        sig = pd.DataFrame(0.0, index=ohlc.index, columns=["chandelier"])
        sig.loc[ohlc["close"] < long_exit, "chandelier"] = -1.0  # 破多止损
        sig.loc[ohlc["close"] > short_exit, "chandelier"] = 1.0  # 破空止损
        return sig
