# coding=utf-8
"""gold_risk.py — 黄金避险情绪策略.

核心逻辑:
    市场恐慌 → 避险需求 → 做多黄金
    - VIX: 标普500波动率指数
    - GPRD: 地缘政治风险指数

策略变体:
    - vix_pct: VIX > N周百分位阈值 → 做多
    - gprd_pct: GPRD > N周百分位阈值 → 做多
    - combined: VIX和GPRD联合信号
    - weighted: VIX*0.5 + GPRD*0.5
"""
from __future__ import annotations

import pandas as pd

from .gold_utils import (
    load_gold_price_w,
    load_indicator_w,
    backtest_weekly,
    get_weekly_returns as _get_weekly_returns,
)


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
        vix_threshold: VIX百分位阈值
        gprd_pct_window: GPRD百分位计算窗口
        gprd_threshold: GPRD百分位阈值
        variant: 策略变体 (vix_pct/gprd_pct/combined/weighted)

    返回: DataFrame with columns:
        - close, vix, gprd, vix_pct, gprd_pct, signal, position
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
        weighted_threshold = (vix_threshold + gprd_threshold) / 2
        signal = (score > weighted_threshold).astype(float)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    position = signal.fillna(0).clip(0, 1)

    return pd.DataFrame({
        "close": close, "vix": vix, "gprd": gprd,
        "vix_pct": vix_pct, "gprd_pct": gprd_pct,
        "signal": signal, "position": position,
    })


def backtest_risk(signal: pd.DataFrame, cost_bps: float = 5.0) -> dict:
    """回测避险情绪策略."""
    return backtest_weekly(signal, cost_bps)


def get_weekly_returns(
    start="2018-01-01",
    end="2025-12-31",
    vix_threshold: float = 0.7,
    variant: str = "vix_pct",
):
    """获取周频收益用于组合优化."""
    signal = compute_risk_signal(vix_threshold=vix_threshold, variant=variant)
    return _get_weekly_returns(signal, start, end)


__all__ = [
    "compute_risk_signal",
    "backtest_risk",
    "get_weekly_returns",
]
