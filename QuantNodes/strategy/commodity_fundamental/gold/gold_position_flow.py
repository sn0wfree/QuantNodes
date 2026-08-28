# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/gold/gold_position_flow.py

黄金仓位流策略

核心逻辑:
    央行持续购金 + ETF资金流入 → 实体需求强劲 → 做多黄金
    进攻型因子, 反映官方和零售需求变化

策略变体:
    - cb_only: 央行黄金储备6个月变化率 > 阈值
    - etf_only: ETF持仓20日变化率 > 0
    - combined: 央行 + ETF联合信号
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


def load_cb_w() -> pd.Series:
    """央行黄金储备周度 (W-FRI, 月度数据前向填充)."""
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "SELECT trade_date, value FROM fundamentals_daily "
        "WHERE source='gold_position' AND value_col='cb_reserves' "
        "ORDER BY trade_date"
    ).df()
    con.close()
    if df.empty:
        return pd.Series(dtype=float)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    s = df.set_index("trade_date")["value"].resample("W-FRI").last().ffill().dropna()
    return s


def load_etf_w() -> pd.Series:
    """ETF持仓周度 (W-FRI)."""
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        "SELECT trade_date, value FROM fundamentals_daily "
        "WHERE source='gold_position' AND value_col='etf_tonnes' "
        "ORDER BY trade_date"
    ).df()
    con.close()
    if df.empty:
        return pd.Series(dtype=float)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    s = df.set_index("trade_date")["value"].resample("W-FRI").last().ffill().dropna()
    return s


def compute_position_flow_signal(
    cb_change_window: int = 26,
    etf_change_window: int = 4,
    variant: str = "combined",
) -> pd.DataFrame:
    """计算仓位流策略信号.

    参数:
        cb_change_window: 央行储备变化窗口 (周)
        etf_change_window: ETF变化窗口 (周)
        variant: 策略变体
            - "cb_only": 央行储备持续增持
            - "etf_only": ETF持续流入
            - "combined": 央行+ETF联合 (任一)

    返回: DataFrame with columns:
        - close: 金价
        - cb_reserves: 央行储备
        - cb_chg: 央行储备变化率
        - etf_tonnes: ETF持仓
        - etf_chg: ETF变化率
        - signal: 原始信号 (0/1)
        - position: 仓位 (0/1)
    """
    close = load_gold_price_w()
    cb = load_cb_w()
    etf = load_etf_w()

    common = close.index
    out = pd.DataFrame(index=common)
    out["close"] = close.reindex(common)
    out["cb_reserves"] = cb.reindex(common)
    out["etf_tonnes"] = etf.reindex(common)

    cb_chg = cb.pct_change(cb_change_window)
    etf_chg = etf.pct_change(etf_change_window)
    out["cb_chg"] = cb_chg.reindex(common)
    out["etf_chg"] = etf_chg.reindex(common)

    if variant == "cb_only":
        signal = (cb_chg > 0).astype(float)
    elif variant == "etf_only":
        signal = (etf_chg > 0).astype(float)
    elif variant == "combined":
        s_cb = (cb_chg > 0).astype(float)
        s_etf = (etf_chg > 0).astype(float)
        signal = ((s_cb.fillna(0) + s_etf.fillna(0)) >= 1).astype(float)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    position = signal.fillna(0).clip(0, 1)
    out["signal"] = signal
    out["position"] = position

    return out


def backtest_position_flow(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
) -> dict:
    """回测仓位流策略."""
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
    variant: str = "combined",
):
    """获取周频收益用于组合优化."""
    signal = compute_position_flow_signal(variant=variant)
    close = signal["close"]
    pos = signal["position"]
    ret = close.pct_change()
    strat_ret = pos.shift(1) * ret
    weekly_ret = strat_ret.resample("W-FRI").sum()
    weekly_ret = weekly_ret[(weekly_ret.index >= start) & (weekly_ret.index <= end)]
    return weekly_ret


__all__ = [
    "load_gold_price_w",
    "load_cb_w",
    "load_etf_w",
    "compute_position_flow_signal",
    "backtest_position_flow",
    "get_weekly_returns",
]