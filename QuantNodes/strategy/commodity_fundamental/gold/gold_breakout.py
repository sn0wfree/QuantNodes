# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/gold/gold_breakout.py

黄金突破策略 + 追踪止损

核心逻辑:
    - 金价接近N日高点时做多 (趋势跟踪)
    - 追踪止损: 从最高价回撤X%时出场

参考铜突破策略 (copper_breakout.py), 直接复用逻辑, 仅替换数据源.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB = Path("data/cmf/fundamental.duckdb")


def load_gold_price() -> pd.Series:
    """加载SHFE黄金主力日度收盘价."""
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"]


def compute_breakout_signal(
    lookback: int = 20,
    threshold: float = 0.95,
    trailing_stop: float = 0.03,
) -> pd.DataFrame:
    """计算黄金突破策略信号.

    参数:
        lookback: 突破窗口 (天)
        threshold: 突破阈值 (接近高点的比例)
        trailing_stop: 追踪止损比例

    返回: DataFrame with columns:
        - close: 金价
        - high_n: N日最高价
        - signal: 原始信号 (1=做多, 0=空仓)
        - position: 带止损的仓位
        - peak_price: 追踪止损用的峰值
    """
    au = load_gold_price()

    high_n = au.rolling(lookback).max()
    signal = (au > high_n * threshold).astype(float)

    position = pd.Series(0.0, index=au.index)
    current_pos = 0.0
    peak_price = 0.0

    for i in range(1, len(au)):
        if signal.iloc[i] > 0 and current_pos == 0:
            current_pos = 1.0
            peak_price = au.iloc[i]

        if current_pos > 0:
            peak_price = max(peak_price, au.iloc[i])
            if au.iloc[i] < peak_price * (1 - trailing_stop):
                current_pos = 0.0

        if signal.iloc[i] == 0 and current_pos > 0:
            current_pos = 0.0

        position.iloc[i] = current_pos

    return pd.DataFrame({
        "close": au,
        "high_n": high_n,
        "signal": signal,
        "position": position,
        "peak_price": pd.Series(peak_price, index=au.index),
    })


def backtest_breakout(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
    periods_per_year: float = 252.0,
) -> dict:
    """回测黄金突破策略 (日频口径).

    与周频 backtest_weekly 的差异: 年化因子默认 252 (日频),
    可通过 periods_per_year 显式指定以对齐其他策略口径.
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
            "win_rate_active": 0, "avg_pos": 0, "pct_active": 0,
            "turnover": 0, "nav": pd.Series(dtype=float),
        }

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / periods_per_year
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol = strat_ret.std() * np.sqrt(periods_per_year)
    max_dd = (nav / nav.cummax() - 1).min()
    active = strat_ret[pos.shift(1).reindex(strat_ret.index).fillna(0) > 0]
    n_turnover = pos.diff().abs().sum() / 2
    turnover = n_turnover / (len(strat_ret) / periods_per_year)

    return {
        "n_days": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "win_rate_active": (active > 0).mean() if len(active) > 0 else 0,
        "avg_pos": pos.mean(),
        "pct_active": (pos > 0).mean(),
        "turnover": turnover,
        "nav": nav,
    }


def get_weekly_returns(start="2018-01-01", end="2025-12-31"):
    """获取周频收益用于组合优化."""
    signal = compute_breakout_signal()
    ret = signal["close"].pct_change()
    pos = signal["position"]
    strat_ret = pos.shift(1) * ret
    weekly_ret = strat_ret.resample("W-FRI").sum()
    weekly_ret = weekly_ret[(weekly_ret.index >= start) & (weekly_ret.index <= end)]
    return weekly_ret


__all__ = [
    "compute_breakout_signal",
    "backtest_breakout",
    "get_weekly_returns",
]