# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/gold/gold_risk.py

黄金避险情绪策略

核心逻辑:
    市场恐慌 → 避险需求 → 做多黄金
    - VIX: 标普500波动率指数 (市场恐慌)
    - GPRD: 地缘政治风险指数 (地缘冲突)
    - GVZ: 黄金ETF波动率指数 (黄金自身波动)

策略变体:
    - vix_pct: VIX > N周百分位阈值 → 做多
    - gprd_pct: GPRD > N周百分位阈值 → 做多
    - combined: VIX和GPRD联合信号
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


def load_indicator_w(value_col: str, source: str = "gold_fred") -> pd.Series:
    """加载指标, 重采样到周度 (W-FRI)."""
    con = duckdb.connect(str(DB), read_only=True)
    df = con.execute(
        f"SELECT trade_date, value FROM fundamentals_daily "
        f"WHERE source='{source}' AND value_col='{value_col}' ORDER BY trade_date"
    ).df()
    con.close()
    if df.empty:
        return pd.Series(dtype=float)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    s = df.set_index("trade_date")["value"].resample("W-FRI").last().ffill().dropna()
    return s


def _rolling_pct(s: pd.Series, window: int) -> pd.Series:
    """计算滚动百分位排名."""
    return s.rolling(window).rank(pct=True)


def compute_risk_signal(
    vix_pct_window: int = 52,
    vix_threshold: float = 0.7,
    gprd_pct_window: int = 52,
    gprd_threshold: float = 0.7,
    variant: str = "combined",
) -> pd.DataFrame:
    """计算避险情绪策略信号.

    参数:
        vix_pct_window: VIX百分位计算窗口
        vix_threshold: VIX百分位阈值 (>阈值做多)
        gprd_pct_window: GPRD百分位计算窗口
        gprd_threshold: GPRD百分位阈值
        variant: 策略变体
            - "vix_pct": VIX百分位
            - "gprd_pct": GPRD百分位
            - "combined": VIX+GPRD联合
            - "weighted": VIX*0.5 + GPRD*0.5

    返回: DataFrame with columns:
        - close: 金价
        - vix: VIX值
        - gprd: GPRD值
        - vix_pct: VIX百分位
        - gprd_pct: GPRD百分位
        - signal: 原始信号 (0/1)
        - position: 仓位 (0/1)
    """
    close = load_gold_price_w()
    vix = load_indicator_w("vix")
    gprd = load_indicator_w("gprd", source="gold_risk")

    common = close.index.intersection(vix.index).intersection(gprd.index)
    close = close.reindex(common)
    vix = vix.reindex(common)
    gprd = gprd.reindex(common)

    vix_pct = _rolling_pct(vix, vix_pct_window)
    gprd_pct = _rolling_pct(gprd, gprd_pct_window)

    if variant == "vix_pct":
        signal = (vix_pct > vix_threshold).astype(float)
    elif variant == "gprd_pct":
        signal = (gprd_pct > gprd_threshold).astype(float)
    elif variant == "combined":
        s1 = (vix_pct > vix_threshold).astype(float)
        s2 = (gprd_pct > gprd_threshold).astype(float)
        signal = ((s1 + s2) >= 1).astype(float)
    elif variant == "weighted":
        score = (vix_pct.fillna(0) + gprd_pct.fillna(0)) / 2
        signal = (score > 0.7).astype(float)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    position = signal.fillna(0).clip(0, 1)

    return pd.DataFrame({
        "close": close,
        "vix": vix,
        "gprd": gprd,
        "vix_pct": vix_pct,
        "gprd_pct": gprd_pct,
        "signal": signal,
        "position": position,
    })


def backtest_risk(
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
) -> dict:
    """回测避险情绪策略."""
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
    vix_threshold: float = 0.7,
    variant: str = "vix_pct",
):
    """获取周频收益用于组合优化."""
    signal = compute_risk_signal(vix_threshold=vix_threshold, variant=variant)
    close = signal["close"]
    pos = signal["position"]
    ret = close.pct_change()
    strat_ret = pos.shift(1) * ret
    weekly_ret = strat_ret.resample("W-FRI").sum()
    weekly_ret = weekly_ret[(weekly_ret.index >= start) & (weekly_ret.index <= end)]
    return weekly_ret


__all__ = [
    "load_gold_price_w",
    "load_indicator_w",
    "compute_risk_signal",
    "backtest_risk",
    "get_weekly_returns",
]