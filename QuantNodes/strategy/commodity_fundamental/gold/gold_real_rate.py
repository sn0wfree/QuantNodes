# coding=utf-8
"""gold_real_rate.py — 黄金实际利率择时策略.

核心逻辑:
    实际利率下降 → 黄金持有成本降低 → 做多黄金
    - TIPS收益率: 10年期通胀保值债券收益率 (实际利率代理)
    - RIR: 美联储模型实际利率

策略变体:
    - tips_ma: TIPS < MA(TIPS, N周) → 做多
    - tips_level: TIPS < 阈值 → 做多
    - rir_negative: RIR < 0 → 做多
    - combined: 多个信号投票
"""
from __future__ import annotations

import pandas as pd

from .gold_utils import (
    load_gold_price_w,
    load_indicator_w,
    backtest_weekly,
    get_weekly_returns as _get_weekly_returns,
)


def compute_real_rate_signal(
    tips_window: int = 26,
    use_rir_filter: bool = False,
    use_level_filter: bool = False,
    tips_level_thresh: float = 0.5,
    variant: str = "tips_ma",
    tips_daily: pd.Series = None,
) -> pd.DataFrame:
    """计算实际利率策略信号.

    参数:
        tips_window: TIPS均线窗口 (周)
        use_rir_filter: 是否使用RIR负值过滤
        use_level_filter: 是否使用TIPS水平过滤
        tips_level_thresh: TIPS水平阈值 (%)
        variant: 策略变体 (tips_ma/tips_level/rir_negative/combined)
        tips_daily: 日频tips10y序列 (可选, 统一入口)

    返回: DataFrame with columns:
        - close, tips, rir, tips_ma, signal, position
    """
    close = load_gold_price_w()

    if tips_daily is not None:
        tips = tips_daily.reindex(close.index).ffill().resample("W-FRI").last().ffill()
    else:
        tips = load_indicator_w("tips")
    rir = load_indicator_w("rir")

    common = close.index.intersection(tips.index)
    close = close.reindex(common)
    tips = tips.reindex(common)
    rir = rir.reindex(common)

    tips_ma = tips.rolling(tips_window).mean()

    if variant == "tips_ma":
        signal = (tips < tips_ma).astype(float)
    elif variant == "tips_level":
        signal = (tips < tips_level_thresh).astype(float)
    elif variant == "rir_negative":
        signal = (rir < 0).astype(float)
    elif "variant" == "combined":
        s1 = (tips < tips_ma).astype(float)
        s2 = (tips < tips_level_thresh).astype(float)
        s3 = (rir < 0).astype(float)
        signal = ((s1 + s2 + s3) / 3 >= 0.5).astype(float)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if use_rir_filter and variant != "rir_negative":
        signal = signal * (rir < 1.0).astype(float)

    position = signal.fillna(0).clip(0, 1)

    return pd.DataFrame({
        "close": close, "tips": tips, "rir": rir,
        "tips_ma": tips_ma, "signal": signal, "position": position,
    })


def backtest_real_rate(signal: pd.DataFrame, cost_bps: float = 5.0) -> dict:
    """回测实际利率策略."""
    return backtest_weekly(signal, cost_bps)


def get_weekly_returns(
    start="2018-01-01",
    end="2025-12-31",
    tips_window: int = 26,
    variant: str = "tips_ma",
):
    """获取周频收益用于组合优化."""
    signal = compute_real_rate_signal(tips_window=tips_window, variant=variant)
    return _get_weekly_returns(signal, start, end)


__all__ = [
    "compute_real_rate_signal",
    "backtest_real_rate",
    "get_weekly_returns",
]
