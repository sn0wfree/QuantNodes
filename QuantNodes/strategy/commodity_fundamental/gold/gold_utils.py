# coding=utf-8
"""gold_utils.py — 黄金策略公共函数模块.

消除6个单因子策略文件中的重复代码:
    - load_gold_price_d/w() — 日频/周频金价加载
    - load_indicator_d/w() — 日频/周频指标加载
    - backtest_weekly() — 通用周频回测
    - get_weekly_returns() — 通用周频收益
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB_PATH = Path("data/cmf/fundamental.duckdb")


def load_gold_price_d() -> pd.Series:
    """加载黄金日度收盘价."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"]


def load_gold_price_w() -> pd.Series:
    """加载黄金周度收盘价 (W-FRI)."""
    return load_gold_price_d().resample("W-FRI").last().dropna()


def load_indicator_d(value_col: str, source: str = "gold_fred") -> pd.Series:
    """加载日度指标."""
    con = duckdb.connect(str(DB_PATH), read_only=True)
    df = con.execute(
        f"SELECT trade_date, value FROM fundamentals_daily "
        f"WHERE source='{source}' AND value_col='{value_col}' ORDER BY trade_date"
    ).df()
    con.close()
    if df.empty:
        return pd.Series(dtype=float)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["value"]


def load_indicator_w(value_col: str, source: str = "gold_fred") -> pd.Series:
    """加载月度/日度指标, 前向填充到周度 (W-FRI)."""
    return load_indicator_d(value_col, source).resample("W-FRI").last().ffill().dropna()


def backtest_weekly(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
) -> dict:
    """通用周频回测.

    Args:
        signal: DataFrame with columns ['close', 'position']
        cost_bps: 交易成本 (bps)

    Returns:
        dict with keys: n_weeks, ann_ret, ann_vol, sharpe, max_dd,
                        win_rate, win_rate_active, avg_pos, pct_active, nav
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
            "n_weeks": 0, "ann_ret": 0, "ann_vol": 0,
            "sharpe": 0, "max_dd": 0, "win_rate": 0,
            "win_rate_active": 0, "avg_pos": 0, "pct_active": 0,
            "nav": pd.Series(dtype=float),
        }

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = len(strat_ret) / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1
    ann_vol = strat_ret.std() * np.sqrt(52)
    max_dd = (nav / nav.cummax() - 1).min()
    active = strat_ret[pos.shift(1).reindex(strat_ret.index).fillna(0) > 0]

    return {
        "n_weeks": len(strat_ret),
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0,
        "max_dd": max_dd,
        "win_rate": (strat_ret > 0).mean(),
        "win_rate_active": (active > 0).mean() if len(active) > 0 else 0,
        "avg_pos": pos.mean(),
        "pct_active": (pos > 0).mean(),
        "nav": nav,
    }


def get_weekly_returns(
    signal: pd.DataFrame,
    start: str = "2018-01-01",
    end: str = "2025-12-31",
) -> pd.Series:
    """获取周频收益用于组合优化.

    Args:
        signal: DataFrame with columns ['close', 'position']
        start: 开始日期
        end: 结束日期

    Returns:
        周频收益序列
    """
    close = signal["close"]
    pos = signal["position"]
    ret = close.pct_change()
    strat_ret = pos.shift(1) * ret
    weekly_ret = strat_ret.resample("W-FRI").sum()
    return weekly_ret[(weekly_ret.index >= start) & (weekly_ret.index <= end)]


__all__ = [
    "DB_PATH",
    "load_gold_price_d",
    "load_gold_price_w",
    "load_indicator_d",
    "load_indicator_w",
    "backtest_weekly",
    "get_weekly_returns",
]
