# coding=utf-8
"""commodity_breakout.py — 通用商品突破策略 + 追踪止损.

数据源无关: 接收价格 Series 作为参数, 调用方自行加载.

核心逻辑:
    - 价格接近N日高点时做多 (趋势跟踪)
    - 追踪止损: 从最高价回撤X%时出场

参数:
    - lookback: 突破窗口 (天, 默认20)
    - threshold: 突破阈值 (接近高点的比例, 默认0.95)
    - trailing_stop: 追踪止损比例 (默认0.03)

参考实现:
    - QuantNodes/strategy/commodity_fundamental/gold/gold_breakout.py
    - QuantNodes/strategy/commodity_fundamental/copper/copper_breakout.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_breakout_signal(
    close: pd.Series,
    lookback: int = 20,
    threshold: float = 0.95,
    trailing_stop: float = 0.03,
) -> pd.DataFrame:
    """计算突破策略信号.

    Args:
        close: 价格序列 (index为日期)
        lookback: 突破窗口 (天)
        threshold: 突破阈值 (价格 > N日高点 * threshold → 做多)
        trailing_stop: 追踪止损比例 (从峰值回撤X%时出场)

    Returns:
        DataFrame with columns:
            - close: 收盘价
            - high_n: N日最高价
            - signal: 原始信号 (1=做多, 0=空仓)
            - position: 带止损的仓位
            - peak_price: 追踪止损用的峰值
    """
    high_n = close.rolling(lookback).max()
    signal = (close > high_n * threshold).astype(float)

    position = pd.Series(0.0, index=close.index)
    current_pos = 0.0
    peak_price = 0.0

    for i in range(1, len(close)):
        if signal.iloc[i] > 0 and current_pos == 0:
            current_pos = 1.0
            peak_price = close.iloc[i]

        if current_pos > 0:
            peak_price = max(peak_price, close.iloc[i])
            if close.iloc[i] < peak_price * (1 - trailing_stop):
                current_pos = 0.0

        if signal.iloc[i] == 0 and current_pos > 0:
            current_pos = 0.0

        position.iloc[i] = current_pos

    return pd.DataFrame({
        "close": close,
        "high_n": high_n,
        "signal": signal,
        "position": position,
        "peak_price": pd.Series(peak_price, index=close.index),
    })


def backtest_breakout(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
) -> dict:
    """回测突破策略 (日频).

    Args:
        signal: compute_breakout_signal() 输出
        cost_bps: 交易成本 (基点)

    Returns:
        dict with keys: n_days, ann_ret, ann_vol, sharpe, max_dd,
                        win_rate, avg_pos, nav
    """
    close = signal["close"]
    pos = signal["position"]
    ret = close.pct_change()

    strat_ret = pos.shift(1) * ret
    cost = pos.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()
    nav = (1 + strat_ret).cumprod()

    if len(strat_ret) == 0:
        return {
            "n_days": 0, "ann_ret": 0, "ann_vol": 0,
            "sharpe": 0, "max_dd": 0, "win_rate": 0,
            "avg_pos": 0, "nav": pd.Series(dtype=float),
        }

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 252
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol = strat_ret.std() * np.sqrt(252)
    max_dd = (nav / nav.cummax() - 1).min()

    return {
        "n_days": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "avg_pos": pos.mean(),
        "nav": nav,
    }


__all__ = [
    "compute_breakout_signal",
    "backtest_breakout",
]
