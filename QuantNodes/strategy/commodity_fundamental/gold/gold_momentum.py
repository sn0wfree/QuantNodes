# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/gold/gold_momentum.py

黄金长期动量策略

核心逻辑:
    黄金长期上涨趋势中, 52周动量为正 → 持续做多
    进攻型因子, 用于牛市中放大仓位暴露

策略变体:
    - long_only: 52周动量 > 0 → 做多 (主要变体)
    - dual_window: 26周 + 52周动量联合
    - momentum_breakout: 动量>阈值 (如30%) → 做多
"""
from __future__ import annotations

import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DB = Path("data/cmf/fundamental.duckdb")


def load_gold_price_w() -> pd.Series:
    """加载黄金周度收盘价 (W-FRI)."""
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df.set_index("trade_date")["close"].resample("W-FRI").last().dropna()


def compute_momentum_signal(
    mom_window: int = 52,
    mom_threshold: float = 0.0,
    short_window: int = 26,
    variant: str = "long_only",
) -> pd.DataFrame:
    """计算动量策略信号.

    参数:
        mom_window: 长期动量窗口 (周)
        mom_threshold: 动量阈值 (>阈值做多)
        short_window: 短期动量窗口 (dual_window变体使用)
        variant: 策略变体
            - "long_only": 52周动量>0 → 做多
            - "dual_window": 26周和52周都>0 → 做多
            - "momentum_breakout": 52周动量>阈值 → 做多

    返回: DataFrame with columns:
        - close: 金价
        - mom_long: 长期动量 (52周)
        - mom_short: 短期动量 (26周)
        - signal: 原始信号 (0/1)
        - position: 仓位 (0/1)
    """
    close = load_gold_price_w()

    mom_long = close.pct_change(mom_window)
    mom_short = close.pct_change(short_window)

    if variant == "long_only":
        signal = (mom_long > mom_threshold).astype(float)
    elif variant == "dual_window":
        s1 = (mom_long > 0).astype(float)
        s2 = (mom_short > 0).astype(float)
        signal = ((s1 + s2) >= 1).astype(float)
    elif variant == "momentum_breakout":
        signal = (mom_long > mom_threshold).astype(float)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    position = signal.fillna(0).clip(0, 1)

    return pd.DataFrame({
        "close": close,
        "mom_long": mom_long,
        "mom_short": mom_short,
        "signal": signal,
        "position": position,
    })


def backtest_momentum(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
) -> dict:
    """回测动量策略."""
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
    start="2018-01-01",
    end="2025-12-31",
    mom_window: int = 52,
    variant: str = "long_only",
):
    """获取周频收益用于组合优化."""
    signal = compute_momentum_signal(mom_window=mom_window, variant=variant)
    close = signal["close"]
    pos = signal["position"]
    ret = close.pct_change()
    strat_ret = pos.shift(1) * ret
    weekly_ret = strat_ret.resample("W-FRI").sum()
    weekly_ret = weekly_ret[(weekly_ret.index >= start) & (weekly_ret.index <= end)]
    return weekly_ret


__all__ = [
    "load_gold_price_w",
    "compute_momentum_signal",
    "backtest_momentum",
    "get_weekly_returns",
]