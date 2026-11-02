# coding=utf-8
"""技术性止损: ATR 移动止损 (吊灯).

文章 §"技术性止损" 给出三种形式: 固定百分比、ATR 静态、ATR 移动 (吊灯).
吊灯是最常用的趋势策略止损: 止损位只向浮盈最大化方向移动,
既能在趋势反转初期锁定大部分利润, 又能避免"被震出"。

本模块对已有的 ChandelierExitFactor 做薄包装, 暴露**止损价位**(Series)
而非触发事件, 便于在回测引擎中按价位判定平仓.

研报: 华泰金工《时序CTA方法论综述》(2026-06-12); 吊灯止损 (LeBeau).
"""
from __future__ import annotations

import pandas as pd

from ..factors.indicators import atr
from ..factors.trend_advanced import ChandelierExitFactor


def _true_range(ohlc: pd.DataFrame) -> pd.Series:
    """真实波幅 TR: max(H-L, |H-prevC|, |L-prevC|)."""
    prev_close = ohlc["close"].shift(1)
    return pd.concat([
        ohlc["high"] - ohlc["low"],
        (ohlc["high"] - prev_close).abs(),
        (ohlc["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def _atr_clipped(
    ohlc: pd.DataFrame,
    window: int = 14,
    clip_low: float = 0.05,
    clip_high: float = 0.95,
) -> pd.Series:
    """分位截断版 ATR (研报原文): 对 TR 序列执行 5%-95% 动态分位数截断.

    "动态"指截断边界由滚动窗口内的分位数实时确定, 而非全样本固定值;
    用于抑制单日极端波动 (涨跌停/跳空) 对 ATR 的冲击.
    """
    tr = _true_range(ohlc)
    # 分位估计窗口: 2×ATR窗口, 保证 5%/95% 分位有足够样本
    clip_win = window * 2
    tr_lo = tr.rolling(clip_win, min_periods=window).quantile(clip_low)
    tr_hi = tr.rolling(clip_win, min_periods=window).quantile(clip_high)
    tr_clipped = tr.clip(lower=tr_lo, upper=tr_hi)
    return tr_clipped.ewm(alpha=1 / window, min_periods=window).mean()


def chandelier_stop_prices(
    ohlc: pd.DataFrame,
    window: int = 22,
    atr_window: int = 14,
    mult: float = 3.0,
    tr_clip: bool = True,
    clip_low: float = 0.05,
    clip_high: float = 0.95,
) -> pd.DataFrame:
    """吊灯止损**价位** (而非触发事件).

    Args:
        ohlc: 含 high / low / close 的 OHLC 面板.
        window: 最高/最低滚动窗口 (默认 22)
        atr_window: ATR 窗口 (默认 14)
        mult: ATR 倍数 (默认 3.0, 文章推荐 k≤5)
        tr_clip: 是否对 TR 做 5%-95% 分位截断 (研报原文要求, 默认开启)
        clip_low / clip_high: 截断分位数 (默认 5%/95%)

    Returns:
        pd.DataFrame, 列 [long_stop, short_stop]:
            long_stop  = rolling(window) high - mult × ATR  ← 多头跌破即离场
            short_stop = rolling(window) low  + mult × ATR  ← 空头升破即离场
    """
    req = {"high", "low", "close"}
    if not req.issubset(ohlc.columns):
        raise ValueError(f"chandelier_stop_prices requires columns {req}")
    roll_high = ohlc["high"].rolling(window).max()
    roll_low = ohlc["low"].rolling(window).min()
    if tr_clip:
        a = _atr_clipped(ohlc, atr_window, clip_low, clip_high)
    else:
        a = atr(ohlc, atr_window)
    return pd.DataFrame({
        "long_stop": roll_high - mult * a,
        "short_stop": roll_low + mult * a,
    }, index=ohlc.index)


def chandelier_exit_signal(
    ohlc: pd.DataFrame,
    position: pd.Series,
    window: int = 22,
    atr_window: int = 14,
    mult: float = 3.0,
    tr_clip: bool = True,
    clip_low: float = 0.05,
    clip_high: float = 0.95,
) -> pd.Series:
    """逐日判定: 当前持仓在吊灯止损下是否应离场.

    Args:
        ohlc: OHLC 面板
        position: 持仓方向 Series, 取值 {-1, 0, +1}
        window / atr_window / mult / tr_clip: 同 chandelier_stop_prices

    Returns:
        pd.Series[bool], True 表示该日应清仓 (从有仓位变为 0).
    """
    stops = chandelier_stop_prices(
        ohlc, window, atr_window, mult, tr_clip, clip_low, clip_high,
    )
    close = ohlc["close"]
    out = pd.Series(False, index=ohlc.index)
    long_open = position > 0
    short_open = position < 0
    out[long_open & (close < stops["long_stop"])] = True
    out[short_open & (close > stops["short_stop"])] = True
    return out


def make_chandelier_factor(
    window: int = 22,
    atr_window: int = 14,
    mult: float = 3.0,
) -> ChandelierExitFactor:
    """构造已配置好的 ChandelierExitFactor 实例 (供调用方复用).

    该因子返回事件型信号 ({0, ±1}), 与本文的价位接口互补.
    """
    return ChandelierExitFactor(window=window, atr_window=atr_window, mult=mult)
