# coding=utf-8
"""CTA 技术指标基础函数 (向量化).

提供不依赖 TA-Lib 的纯 pandas 指标实现, 输入输出均为 DataFrame/Series:
    - RSI               相对强弱
    - ATR               平均真实波幅
    - ADX              平均趋向指数 (趋势强度)
    - BollingerBands    布林带
    - EfficiencyRatio   考夫曼效率比
    - Donchian          唐奇安通道
    - ZScore            滚动 z-score
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """相对强弱指标 RSI (Wilder 平滑)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    # avg_loss=0 → 全部上涨 → RSI=100; avg_gain=0 → 全部下跌 → RSI=0
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        out = 100 - 100 / (1 + rs)
    out = out.replace([np.inf, -np.inf], 100.0)
    return out.fillna(50)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """平均真实波幅 ATR. df 需含 high/low/close 列."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window).mean()


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """平均趋向指数 ADX (趋势强度, 0-100). df 需含 high/low/close 列."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr_w = tr.ewm(alpha=1 / window, min_periods=window).mean().replace(0, np.nan)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / window, min_periods=window).mean() / atr_w
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / window, min_periods=window).mean() / atr_w

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, min_periods=window).mean().fillna(0)


def bollinger_bands(
    close: pd.Series,
    window: int = 20,
    n_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """布林带: (中轨, 上轨, 下轨)."""
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    return mid, mid + n_std * std, mid - n_std * std


def bollinger_percent_b(
    close: pd.Series,
    window: int = 20,
    n_std: float = 2.0,
) -> pd.Series:
    """布林带 %B: 价格在通道中的位置 (0=下轨, 1=上轨)."""
    mid, upper, lower = bollinger_bands(close, window, n_std)
    return (close - lower) / (upper - lower).replace(0, np.nan)


def efficiency_ratio(close: pd.Series, window: int = 10) -> pd.Series:
    """考夫曼效率比 = |净变动| / 路径总长 (0-1, 越高趋势越强)."""
    change = (close - close.shift(window)).abs()
    volatility = close.diff().abs().rolling(window).sum()
    return change / volatility.replace(0, np.nan)


def donchian(close: pd.Series, window: int = 20) -> tuple[pd.Series, pd.Series]:
    """唐奇安通道 (不含当日): (上轨, 下轨)."""
    return close.rolling(window).max().shift(1), close.rolling(window).min().shift(1)


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """滚动 z-score."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)
