# coding=utf-8
"""gold_momentum.py — 黄金长期动量策略.

核心逻辑:
    黄金长期上涨趋势中, 52周动量为正 → 持续做多
    进攻型因子, 用于牛市中放大仓位暴露

策略变体:
    - long_only: 52周动量 > 0 → 做多
    - dual_window: 26周 + 52周动量联合
    - momentum_breakout: 动量>阈值 → 做多
"""
from __future__ import annotations

import pandas as pd

from .gold_utils import (
    load_gold_price_w,
    backtest_weekly,
    get_weekly_returns as _get_weekly_returns,
)


def compute_momentum_signal(
    mom_window: int = 52,
    mom_threshold: float = 0.0,
    short_window: int = 26,
    variant: str = "long_only",
) -> pd.DataFrame:
    """计算动量策略信号.

    参数:
        mom_window: 长期动量窗口 (周)
        mom_threshold: 动量阈值
        short_window: 短期动量窗口
        variant: 策略变体 (long_only/dual_window/momentum_breakout)

    返回: DataFrame with columns:
        - close, mom_long, mom_short, signal, position
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
        "close": close, "mom_long": mom_long, "mom_short": mom_short,
        "signal": signal, "position": position,
    })


def backtest_momentum(signal: pd.DataFrame, cost_bps: float = 5.0) -> dict:
    """回测动量策略."""
    return backtest_weekly(signal, cost_bps)


def get_weekly_returns(
    start="2018-01-01",
    end="2025-12-31",
    mom_window: int = 52,
    variant: str = "long_only",
):
    """获取周频收益用于组合优化."""
    signal = compute_momentum_signal(mom_window=mom_window, variant=variant)
    return _get_weekly_returns(signal, start, end)


__all__ = [
    "compute_momentum_signal",
    "backtest_momentum",
    "get_weekly_returns",
]
