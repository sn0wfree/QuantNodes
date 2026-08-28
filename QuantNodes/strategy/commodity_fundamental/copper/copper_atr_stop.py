# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/copper/copper_atr_stop.py

铜ATR追踪止损策略

月度信号生成 + 日度ATR追踪止损

核心逻辑:
    1. 月末生成月度信号 (sig = +1/-1/0)
    2. 持仓期间每日监控止损线
    3. 做多: 止损线 = max(前止损线, 收盘价 - 1.5×ATR14)
    4. 做空: 止损线 = min(前止损线, 收盘价 + 1.5×ATR14)
    5. 触发: 以止损价平仓

参数:
    atr_multiple: 1.5 (止损距离)
    atr_window: 14 (ATR计算窗口)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB = Path("data/cmf/fundamental.duckdb")


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                window: int = 14) -> pd.Series:
    """计算ATR (Average True Range)."""
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1)),
    }).max(axis=1)
    return tr.rolling(window).mean()


def apply_atr_stop(
    signal_weekly: pd.Series,
    daily_data: pd.DataFrame,
    atr_multiple: float = 1.5,
    atr_window: int = 14,
) -> pd.Series:
    """应用ATR追踪止损.

    参数:
        signal_weekly: 周频信号 (月末信号, index=周频日期)
        daily_data: 日频数据 (columns: open, high, low, close)
        atr_multiple: ATR倍数 (默认1.5)
        atr_window: ATR窗口 (默认14)

    返回: 日频仓位 (止损触发后为0)
    """
    high = daily_data['high']
    low = daily_data['low']
    close = daily_data['close']

    # 计算ATR
    atr = compute_atr(high, low, close, atr_window)

    # 生成月度信号 (取每月最后一个交易日的信号)
    monthly_signal = signal_weekly.resample('M').last()

    # 日频仓位
    position = pd.Series(0.0, index=daily_data.index)
    stop_line = pd.Series(np.nan, index=daily_data.index)

    current_pos = 0.0
    current_stop = np.nan
    entry_price = np.nan

    for i in range(1, len(daily_data)):
        date = daily_data.index[i]
        prev_date = daily_data.index[i-1]

        # 检查是否是新的月份 (需要更新信号)
        if date.month != prev_date.month:
            # 月末信号
            month_end = prev_date
            if month_end in monthly_signal.index:
                new_signal = monthly_signal.loc[month_end]
                if new_signal != current_pos:
                    current_pos = new_signal
                    entry_price = close.loc[date]
                    if current_pos > 0:
                        current_stop = entry_price - atr_multiple * atr.loc[date]
                    elif current_pos < 0:
                        current_stop = entry_price + atr_multiple * atr.loc[date]
                    else:
                        current_stop = np.nan

        # 更新止损线
        if current_pos > 0:
            # 做多: 止损线向上追踪
            new_stop = close.loc[date] - atr_multiple * atr.loc[date]
            current_stop = max(current_stop, new_stop)
        elif current_pos < 0:
            # 做空: 止损线向下追踪
            new_stop = close.loc[date] + atr_multiple * atr.loc[date]
            current_stop = min(current_stop, new_stop)

        # 检查触发
        triggered = False
        if current_pos > 0 and not np.isnan(current_stop):
            if low.loc[date] <= current_stop:
                triggered = True
        elif current_pos < 0 and not np.isnan(current_stop):
            if high.loc[date] >= current_stop:
                triggered = True

        if triggered:
            current_pos = 0.0
            current_stop = np.nan

        position.iloc[i] = current_pos
        stop_line.iloc[i] = current_stop

    return position


def build_signal_with_atr_stop(
    con=None,
    symbol: str = "CU0",
    atr_multiple: float = 1.5,
    atr_window: int = 14,
) -> pd.DataFrame:
    """构建带ATR止损的信号.

    使用v6.2作为基础信号, 加入ATR追踪止损.
    """
    from .copper_model_v6 import build_signal_v6, apply_bull_enhance

    if con is None:
        con = duckdb.connect(str(DB), read_only=True)

    # 加载日频数据
    df_daily = con.execute(
        "SELECT trade_date, open, high, low, close "
        "FROM futures_main_close_cu ORDER BY trade_date"
    ).df()
    df_daily['trade_date'] = pd.to_datetime(df_daily['trade_date'])
    df_daily = df_daily.set_index('trade_date')

    # 构建v6.2信号 (周频)
    sig = build_signal_v6("CU0", con)
    sig = apply_bull_enhance(sig, con, threshold=3, bull_mom_weeks=13)
    sig['position'] = sig['position_sized']

    # 转为周频信号
    weekly_signal = sig['position'].resample('W-FRI').last()

    # 应用ATR止损
    daily_position = apply_atr_stop(
        weekly_signal, df_daily,
        atr_multiple=atr_multiple, atr_window=atr_window,
    )

    # 对齐到周频
    weekly_position = daily_position.resample('W-FRI').last()
    weekly_position = weekly_position.reindex(sig.index).fillna(method='ffill')

    out = pd.DataFrame({
        'M': sig['M'],
        'position': weekly_position,
        'position_raw': sig['position'],
    })

    return out


def backtest_atr_stop(signal: pd.DataFrame, cost_bps: float = 5.0) -> dict:
    """回测ATR止损策略."""
    close = signal['M']
    common = signal.index.intersection(close.index)
    pos = signal.loc[common, "position"].fillna(0)
    ret = close.pct_change().reindex(common)

    strat_ret = pos.shift(1) * ret
    cost = pos.diff().abs() * cost_bps / 10000
    strat_ret = (strat_ret - cost).dropna()

    if len(strat_ret) == 0:
        return {"n_weeks": 0, "ann_ret": 0, "ann_vol": 0, "sharpe": 0,
                "max_dd": 0, "win_rate": 0, "nav": pd.Series(dtype=float)}

    nav = (1 + strat_ret).cumprod()
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = strat_ret.std() * np.sqrt(52)
    max_dd = (nav / nav.cummax() - 1).min()

    return {
        "n_weeks": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "nav": nav,
    }


__all__ = [
    "compute_atr",
    "apply_atr_stop",
    "build_signal_with_atr_stop",
    "backtest_atr_stop",
]
