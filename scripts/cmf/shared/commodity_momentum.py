# coding=utf-8
"""commodity_momentum.py — 通用商品动量策略.

数据源无关: 接收周频收盘价 Series 作为参数.

核心逻辑:
    - 动量: 过去N周收益率 > 0 → 做多
    - 波动率目标化: 高波动时降低仓位

策略变体:
    - long_only: 长期动量 > 0 → 做多
    - dual_window: 短期+长期动量联合
    - with_vol_target: 波动率目标化

参考实现:
    - QuantNodes/strategy/commodity_fundamental/gold/gold_momentum.py (默认52周)
    - QuantNodes/strategy/commodity_fundamental/copper/copper_momentum.py (默认26周)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_momentum_signal(
    close_w: pd.Series,
    lookback: int = 26,
    vol_window: int = 13,
    vol_target: float = 0.12,
    use_vol_target: bool = True,
    variant: str = "long_only",
    short_window: int = 13,
) -> pd.DataFrame:
    """计算动量策略信号.

    Args:
        close_w: 周频收盘价 (index为日期)
        lookback: 长期动量窗口 (周)
        vol_window: 波动率窗口 (周)
        vol_target: 目标波动率 (年化)
        use_vol_target: 是否使用波动率目标化
        variant: 策略变体
            - long_only: 仅长期动量 > 0 → 做多
            - dual_window: 短期+长期动量联合 (双确认)
        short_window: 短期动量窗口 (周, dual_window变体使用)

    Returns:
        DataFrame with columns:
            - close_w: 周频收盘价
            - momentum: 长期动量 (收益率)
            - volatility: 波动率
            - signal: 原始信号
            - position: 仓位 [0, 1]
    """
    mom_long = close_w.pct_change(lookback)

    if variant == "long_only":
        signal = (mom_long > 0).astype(float)
        position = signal.copy()
    elif variant == "dual_window":
        mom_short = close_w.pct_change(short_window)
        s_long = (mom_long > 0).astype(float)
        s_short = (mom_short > 0).astype(float)
        position = pd.Series(0.0, index=close_w.index)
        position[(s_long > 0) & (s_short > 0)] = 1.0
        position[((s_long > 0) ^ (s_short > 0))] = 0.5
    else:
        raise ValueError(f"Unknown variant: {variant}")

    volatility = close_w.pct_change().rolling(vol_window).std() * np.sqrt(52)

    if use_vol_target:
        scale = (vol_target / volatility).fillna(1.0).clip(upper=1.0)
        position = position * scale

    return pd.DataFrame({
        "close_w": close_w,
        "momentum": mom_long,
        "volatility": volatility,
        "signal": signal if variant == "long_only" else position,
        "position": position.fillna(0).clip(0, 1),
    })


__all__ = [
    "compute_momentum_signal",
]