# coding=utf-8
"""期货趋势信号."""
from __future__ import annotations

import pandas as pd

from ..factors.trend import DonchianChannelFactor, MAFactor, MomentumFactor


def donchian_signal(
    prices: pd.DataFrame,
    entry_window: int = 20,
    exit_window: int = 10,
) -> pd.DataFrame:
    """唐奇安通道趋势信号 (-1/0/+1)."""
    return DonchianChannelFactor(entry_window, exit_window).compute(prices)


def ma_signal(
    prices: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
) -> pd.DataFrame:
    """双均线趋势信号."""
    return MAFactor(short_window, long_window).compute(prices)


def momentum_signal(
    prices: pd.DataFrame,
    lookback: int = 20,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """动量趋势信号."""
    return MomentumFactor(lookback, threshold).compute(prices)


def position_from_signal(
    signals: pd.DataFrame,
    max_qty: float = 1.0,
) -> pd.DataFrame:
    """信号 → 目标手数 (简单缩放)."""
    return signals.clip(-1, 1) * max_qty
