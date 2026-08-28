# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_daily_stop.py

日度ATR追踪止损机制

独立于信号生成，只关注持仓状态。

用法:
    # 1. 获取月度持仓信号 (任意来源)
    monthly_positions = pd.Series([1, 1, 0, -1, ...], index=month_ends)
    
    # 2. 应用日度止损
    daily_positions = apply_daily_atr_stop(
        monthly_positions=monthly_positions,
        daily_ohlc=daily_ohlc,
        atr_multiple=1.5,
        atr_window=14,
    )
    
    # 3. daily_positions 是日频仓位，已包含止损逻辑
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                window: int = 14) -> pd.Series:
    """计算ATR."""
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1)),
    }).max(axis=1)
    return tr.rolling(window).mean()


def apply_daily_atr_stop(
    weekly_positions: pd.Series,
    daily_ohlc: pd.DataFrame,
    atr_multiple: float = 1.5,
    atr_window: int = 14,
) -> pd.Series:
    """应用日度ATR追踪止损.

    参数:
        weekly_positions: 周频持仓信号 (index=周频日期, values=仓位)
        daily_ohlc: 日频OHLC数据 (columns: open, high, low, close)
        atr_multiple: ATR倍数 (默认1.5)
        atr_window: ATR窗口 (默认14)

    返回:
        pd.Series: 日频仓位 (止损触发后为0)
    """
    high = daily_ohlc['high']
    low = daily_ohlc['low']
    close = daily_ohlc['close']

    # 计算ATR
    atr = compute_atr(high, low, close, atr_window)

    # 将周频信号映射到日频 (向前填充)
    daily_signal = weekly_positions.reindex(daily_ohlc.index, method='ffill').fillna(0)

    # 应用ATR止损
    daily_position = pd.Series(0.0, index=daily_ohlc.index)

    current_pos = 0.0
    current_stop = np.nan

    for i in range(1, len(daily_ohlc)):
        # 获取当前信号
        new_signal = daily_signal.iloc[i]

        # 信号变化时重置止损
        if new_signal != current_pos:
            current_pos = new_signal
            if current_pos > 0:
                current_stop = close.iloc[i] - atr_multiple * atr.iloc[i]
            elif current_pos < 0:
                current_stop = close.iloc[i] + atr_multiple * atr.iloc[i]
            else:
                current_stop = np.nan

        # 更新止损线
        if current_pos > 0 and not np.isnan(atr.iloc[i]):
            new_stop = close.iloc[i] - atr_multiple * atr.iloc[i]
            current_stop = max(current_stop, new_stop)
        elif current_pos < 0 and not np.isnan(atr.iloc[i]):
            new_stop = close.iloc[i] + atr_multiple * atr.iloc[i]
            current_stop = min(current_stop, new_stop)

        # 检查触发
        triggered = False
        if current_pos > 0 and not np.isnan(current_stop):
            if low.iloc[i] <= current_stop:
                triggered = True
        elif current_pos < 0 and not np.isnan(current_stop):
            if high.iloc[i] >= current_stop:
                triggered = True

        if triggered:
            current_pos = 0.0
            current_stop = np.nan

        daily_position.iloc[i] = current_pos

    return daily_position


def backtest_daily_stop(
    daily_position: pd.Series,
    daily_close: pd.Series,
    cost_bps: float = 5.0,
) -> dict:
    """回测日度止损策略.

    参数:
        daily_position: 日频仓位
        daily_close: 日频收盘价
        cost_bps: 交易成本

    返回: 回测结果字典
    """
    ret = daily_close.pct_change()

    strat_ret = daily_position.shift(1) * ret
    cost = daily_position.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()

    if len(strat_ret) == 0:
        return {"n_days": 0, "ann_ret": 0, "ann_vol": 0, "sharpe": 0,
                "max_dd": 0, "win_rate": 0, "nav": pd.Series(dtype=float)}

    nav = (1 + strat_ret).cumprod()
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 252
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(252)
    max_dd = (nav / nav.cummax() - 1).min()

    return {
        "n_days": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "nav": nav,
    }


__all__ = [
    "compute_atr",
    "apply_daily_atr_stop",
    "backtest_daily_stop",
]
