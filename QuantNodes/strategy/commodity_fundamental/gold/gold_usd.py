# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/gold/gold_usd.py

黄金美元趋势策略

核心逻辑:
    美元走弱 → 黄金上涨 (强负相关)
    - DXY (ICE美元指数) 下降时做多黄金
    - DXY变化率/趋势信号

策略变体:
    - dxy_ema: DXY < EMA(DXY, N周) → 做多
    - dxy_momentum: DXY 20周收益率 < 0 → 做多
    - dxy_zscore: DXY z-score > 阈值 (高位) → 做空美元→做多黄金
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB = Path("data/cmf/fundamental.duckdb")


def load_dxy_w() -> pd.Series:
    """加载美元指数周度 (W-FRI)."""
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "SELECT trade_date, value FROM fundamentals_daily "
        "WHERE source='gold_macro' AND value_col='dxy' "
        "ORDER BY trade_date"
    ).df()
    con.close()
    if df.empty:
        return pd.Series(dtype=float)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    s = df.set_index("trade_date")["value"].resample("W-FRI").last().ffill().dropna()
    return s


def load_gold_price_w() -> pd.Series:
    """加载黄金周度收盘价 (W-FRI)."""
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"].resample("W-FRI").last().dropna()


def compute_usd_signal(
    ema_window: int = 20,
    mom_window: int = 20,
    z_window: int = 52,
    z_thresh: float = 0.5,
    variant: str = "dxy_ema",
) -> pd.DataFrame:
    """计算美元趋势策略信号.

    参数:
        ema_window: DXY均线窗口 (周)
        mom_window: DXY动量窗口 (周)
        z_window: DXY z-score窗口 (周)
        z_thresh: DXY z-score阈值 (高位做空)
        variant: 策略变体
            - "dxy_ema": DXY < EMA → 做多黄金
            - "dxy_momentum": DXY变化率<0 → 做多黄金
            - "dxy_zscore": DXY z-score>阈值 → 做多黄金

    返回: DataFrame with columns:
        - close: 金价
        - dxy: 美元指数
        - dxy_ema: DXY均线
        - dxy_mom: DXY动量
        - dxy_zscore: DXY z-score
        - signal: 原始信号 (0/1)
        - position: 仓位 (0/1)
    """
    close = load_gold_price_w()
    dxy = load_dxy_w()

    common = close.index.intersection(dxy.index)
    close = close.reindex(common)
    dxy = dxy.reindex(common)

    dxy_ema = dxy.ewm(span=ema_window, adjust=False).mean()
    dxy_mom = dxy.pct_change(mom_window)
    dxy_mean = dxy.rolling(z_window).mean()
    dxy_std = dxy.rolling(z_window).std()
    dxy_zscore = (dxy - dxy_mean) / dxy_std

    if variant == "dxy_ema":
        signal = (dxy < dxy_ema).astype(float)
    elif variant == "dxy_momentum":
        signal = (dxy_mom < 0).astype(float)
    elif variant == "dxy_zscore":
        signal = (dxy_zscore > z_thresh).astype(float)
    elif variant == "combined":
        s1 = (dxy < dxy_ema).astype(float)
        s2 = (dxy_mom < 0).astype(float)
        signal = ((s1 + s2) >= 1).astype(float)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    position = signal.fillna(0).clip(0, 1)

    return pd.DataFrame({
        "close": close,
        "dxy": dxy,
        "dxy_ema": dxy_ema,
        "dxy_mom": dxy_mom,
        "dxy_zscore": dxy_zscore,
        "signal": signal,
        "position": position,
    })


def backtest_usd(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
) -> dict:
    """回测美元趋势策略."""
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
            "avg_pos": 0, "pct_active": 0, "nav": pd.Series(dtype=float),
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
    start="2020-01-01",
    end="2025-12-31",
    ema_window: int = 20,
    variant: str = "dxy_ema",
):
    """获取周频收益用于组合优化 (DXY从2020开始, 回测从2020起)."""
    signal = compute_usd_signal(ema_window=ema_window, variant=variant)
    close = signal["close"]
    pos = signal["position"]
    ret = close.pct_change()
    strat_ret = pos.shift(1) * ret
    weekly_ret = strat_ret.resample("W-FRI").sum()
    weekly_ret = weekly_ret[(weekly_ret.index >= start) & (weekly_ret.index <= end)]
    return weekly_ret


__all__ = [
    "load_dxy_w",
    "load_gold_price_w",
    "compute_usd_signal",
    "backtest_usd",
    "get_weekly_returns",
]