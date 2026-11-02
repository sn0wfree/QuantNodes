# coding=utf-8
"""经典振荡指标因子.

基于 OHLCV 的经典摆动/振荡指标:
    - MACDFactor: MACD 金叉/死叉
    - StochasticFactor: 随机指标 KDJ
    - CCIFactor: 商品通道指数
    - WilliamsRFactor: 威廉 %R
    - TSIFactor: 真实强度指数 (双 EMA 平滑动量)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Factor


class MACDFactor(Factor):
    """MACD 因子.

    MACD = EMA12 - EMA26, 信号线 = EMA9(MACD).
    金叉 (MACD 上穿信号线) → 多; 死叉 → 空.
    研报: 经典趋势指标 (Appel).

    参数:
        fast: 快线 EMA (默认 12)
        slow: 慢线 EMA (默认 26)
        signal: 信号线 EMA (默认 9)
    """

    name = "macd"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def raw_value(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """连续值: MACD 柱 (macd - 信号线), 柱为正 → 多头."""
        ema_fast = prices.ewm(span=self.fast, adjust=False).mean()
        ema_slow = prices.ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        return macd - macd.ewm(span=self.signal, adjust=False).mean()

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        macd = self.raw_value(prices)
        signal_line = macd.ewm(span=self.signal, adjust=False).mean()
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        # 金叉/死叉
        cross_up = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
        cross_down = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))
        sig[cross_up] = 1.0
        sig[cross_down] = -1.0
        return sig


class StochasticFactor(Factor):
    """随机指标 KDJ 因子.

    K = SMA(%K, 3), %K = (close - low_n)/(high_n - low_n)×100.
    K 上穿 D → 多; 下穿 → 空.
    研报: 经典随机指标 (Lane).

    参数:
        window: %K 窗口
        smooth: K 平滑窗口
    """

    name = "stochastic"

    def __init__(self, window: int = 14, smooth: int = 3):
        self.window = window
        self.smooth = smooth

    def raw_value(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """连续值: k − d, 上穿为正 → 多头."""
        req = {"high", "low", "close"}
        if not req.issubset(ohlc.columns):
            raise ValueError(f"StochasticFactor requires columns {req}")
        k, d = self._kd(ohlc)
        return pd.DataFrame({"stochastic": k - d}, index=ohlc.index)

    def _kd(self, ohlc: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        high_n = ohlc["high"].rolling(self.window).max()
        low_n = ohlc["low"].rolling(self.window).min()
        pct_k = (ohlc["close"] - low_n) / (high_n - low_n).replace(0, np.nan) * 100
        k = pct_k.rolling(self.smooth).mean()
        d = k.rolling(self.smooth).mean()
        return k, d

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        k, d = self._kd(ohlc)
        sig = pd.DataFrame(0.0, index=ohlc.index, columns=["stochastic"])
        cross_up = (k > d) & (k.shift(1) <= d.shift(1))
        cross_down = (k < d) & (k.shift(1) >= d.shift(1))
        sig.loc[cross_up, "stochastic"] = 1.0
        sig.loc[cross_down, "stochastic"] = -1.0
        return sig


class CCIFactor(Factor):
    """商品通道指数因子.

    CCI = (TP - SMA(TP,n)) / (0.015 × 平均偏差).
    CCI > 100 → 超买; < -100 → 超卖.
    研报: 经典商品通道指数 (Lambert).

    参数:
        window: CCI 窗口
        high_threshold: 超买线 (默认 100)
        low_threshold: 超卖线 (默认 -100)
    """

    name = "cci"

    def __init__(self, window: int = 20, high_threshold: float = 100.0,
                 low_threshold: float = -100.0):
        self.window = window
        self.high = high_threshold
        self.low = low_threshold

    def raw_value(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """连续值: cci 值, >0 强势 / <0 弱势."""
        req = {"high", "low", "close"}
        if not req.issubset(ohlc.columns):
            raise ValueError(f"CCIFactor requires columns {req}")
        tp = (ohlc["high"] + ohlc["low"] + ohlc["close"]) / 3
        tp_sma = tp.rolling(self.window).mean()
        mean_dev = tp.rolling(self.window).apply(
            lambda s: np.mean(np.abs(s - s.mean())), raw=True,
        )
        cci = (tp - tp_sma) / (0.015 * mean_dev.replace(0, np.nan))
        return pd.DataFrame({"cci": cci}, index=ohlc.index)

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        cci = self.raw_value(ohlc)["cci"]
        sig = pd.DataFrame(0.0, index=ohlc.index, columns=["cci"])
        sig.loc[cci < self.low, "cci"] = 1.0    # 超卖 → 多
        sig.loc[cci > self.high, "cci"] = -1.0  # 超买 → 空
        return sig


class WilliamsRFactor(Factor):
    """威廉 %R 因子.

    %R = (high_n - close) / (high_n - low_n) × -100.
    %R < -80 超卖 → 多; > -20 超买 → 空.
    研报: 经典威廉指标 (Williams).

    参数:
        window: %R 窗口
        low_threshold: 超卖线 (默认 -80)
        high_threshold: 超买线 (默认 -20)
    """

    name = "williams_r"

    def __init__(self, window: int = 14, low_threshold: float = -80.0,
                 high_threshold: float = -20.0):
        self.window = window
        self.low = low_threshold
        self.high = high_threshold

    def raw_value(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """连续值: %R 值, 越高越接近超买 (空头侧)."""
        req = {"high", "low", "close"}
        if not req.issubset(ohlc.columns):
            raise ValueError(f"WilliamsRFactor requires columns {req}")
        high_n = ohlc["high"].rolling(self.window).max()
        low_n = ohlc["low"].rolling(self.window).min()
        wr = (high_n - ohlc["close"]) / (high_n - low_n).replace(0, np.nan) * -100
        return pd.DataFrame({"williams_r": wr}, index=ohlc.index)

    def compute(self, ohlc: pd.DataFrame, **kwargs) -> pd.DataFrame:
        wr = self.raw_value(ohlc)["williams_r"]
        sig = pd.DataFrame(0.0, index=ohlc.index, columns=["williams_r"])
        sig.loc[wr < self.low, "williams_r"] = 1.0    # 超卖 → 多
        sig.loc[wr > self.high, "williams_r"] = -1.0  # 超买 → 空
        return sig


class TSIFactor(Factor):
    """真实强度指数 (True Strength Index) 因子.

    双 EMA 平滑动量, 输出 -100 ~ +100:
        PC  = EMA(EMA(Δclose, r), s)
        APC = EMA(EMA(|Δclose|, r), s)
        TSI = 100 · PC / APC

    信号:
        TSI >  high_threshold → 多 (趋势向上)
        TSI <  low_threshold  → 空 (趋势向下)
        区间 [-high, high]     → 0 (中性)

    研报: 华泰金工《时序CTA方法论综述》(2026-06-12), Levine & Pedersen (2016).
    与 RSI 类似但先双 EMA 平滑, 对噪声更鲁棒; 用于构建 TSI 置信区间判别趋势可交易性.

    参数:
        r: 第一层 EMA 窗口 (默认 25)
        s: 第二层 EMA 窗口 (默认 13)
        high_threshold: 多头触发 (默认 25)
        low_threshold: 空头触发 (默认 -25)
    """

    name = "tsi"

    def __init__(self, r: int = 25, s: int = 13,
                 high_threshold: float = 25.0, low_threshold: float = -25.0):
        self.r = r
        self.s = s
        self.high = high_threshold
        self.low = low_threshold

    def raw_value(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """连续值: TSI 值, 范围约 [-100, +100]."""
        delta = prices.diff()
        ema_r = lambda x: x.ewm(span=self.r, adjust=False).mean()
        ema_s = lambda x: x.ewm(span=self.s, adjust=False).mean()
        pc = ema_s(ema_r(delta))
        apc = ema_s(ema_r(delta.abs()))
        tsi = 100.0 * pc / apc.replace(0, np.nan)
        return tsi

    def compute(self, prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
        tsi = self.raw_value(prices)
        sig = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        sig[tsi > self.high] = 1.0
        sig[tsi < self.low] = -1.0
        return sig
