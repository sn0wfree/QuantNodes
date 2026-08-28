# coding=utf-8
"""gold_position_flow.py — 黄金仓位流策略.

核心逻辑:
    央行持续购金 + ETF资金流入 → 实体需求强劲 → 做多黄金
    进攻型因子, 反映官方和零售需求变化

策略变体:
    - cb_only: 央行黄金储备6个月变化率 > 0
    - etf_only: ETF持仓20日变化率 > 0
    - combined: 央行 + ETF联合信号
"""
from __future__ import annotations

import pandas as pd

from .gold_utils import (
    load_gold_price_w,
    load_indicator_w,
    backtest_weekly,
    get_weekly_returns as _get_weekly_returns,
)


def load_cb_w() -> pd.Series:
    """央行黄金储备周度 (W-FRI, 月度数据前向填充)."""
    return load_indicator_w("cb_reserves", source="gold_position")


def load_etf_w() -> pd.Series:
    """ETF持仓周度 (W-FRI)."""
    return load_indicator_w("etf_tonnes", source="gold_position")


def compute_position_flow_signal(
    cb_change_window: int = 26,
    etf_change_window: int = 4,
    variant: str = "combined",
) -> pd.DataFrame:
    """计算仓位流策略信号.

    参数:
        cb_change_window: 央行储备变化窗口 (周)
        etf_change_window: ETF变化窗口 (周)
        variant: 策略变体 (cb_only/etf_only/combined)

    返回: DataFrame with columns:
        - close, cb_reserves, cb_chg, etf_tonnes, etf_chg, signal, position
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


def backtest_position_flow(signal: pd.DataFrame, cost_bps: float = 5.0) -> dict:
    """回测仓位流策略."""
    return backtest_weekly(signal, cost_bps)


def get_weekly_returns(
    start="2020-01-01",
    end="2025-12-31",
    variant: str = "combined",
):
    """获取周频收益用于组合优化."""
    signal = compute_position_flow_signal(variant=variant)
    return _get_weekly_returns(signal, start, end)


__all__ = [
    "load_cb_w",
    "load_etf_w",
    "compute_position_flow_signal",
    "backtest_position_flow",
    "get_weekly_returns",
]
