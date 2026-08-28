# coding=utf-8
"""gold_usd.py — 黄金美元趋势策略.

核心逻辑:
    美元走弱 → 黄金上涨 (强负相关)
    - DXY (ICE美元指数) 下降时做多黄金

策略变体:
    - dxy_ema: DXY < EMA(DXY, N周) → 做多
    - dxy_momentum: DXY 20周收益率 < 0 → 做多
    - dxy_zscore: DXY z-score > 阈值 → 做多
"""
from __future__ import annotations

import pandas as pd

from .gold_utils import (
    load_gold_price_w,
    load_indicator_w,
    backtest_weekly,
    get_weekly_returns as _get_weekly_returns,
)


def load_dxy_w() -> pd.Series:
    """加载美元指数周度 (W-FRI)."""
    return load_indicator_w("dxy", source="gold_macro")


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
        z_thresh: DXY z-score阈值
        variant: 策略变体 (dxy_ema/dxy_momentum/dxy_zscore/combined)

    返回: DataFrame with columns:
        - close, dxy, dxy_ema, dxy_mom, dxy_zscore, signal, position
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
        "close": close, "dxy": dxy, "dxy_ema": dxy_ema,
        "dxy_mom": dxy_mom, "dxy_zscore": dxy_zscore,
        "signal": signal, "position": position,
    })


def backtest_usd(signal: pd.DataFrame, cost_bps: float = 5.0) -> dict:
    """回测美元趋势策略."""
    return backtest_weekly(signal, cost_bps)


def get_weekly_returns(
    start="2020-01-01",
    end="2025-12-31",
    ema_window: int = 20,
    variant: str = "dxy_ema",
):
    """获取周频收益用于组合优化."""
    signal = compute_usd_signal(ema_window=ema_window, variant=variant)
    return _get_weekly_returns(signal, start, end)


__all__ = [
    "load_dxy_w",
    "compute_usd_signal",
    "backtest_usd",
    "get_weekly_returns",
]
