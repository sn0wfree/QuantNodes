# coding=utf-8
"""commodity_vol_target.py — 通用商品波动率目标化.

数据源无关: 接收仓位 + 波动率序列作为参数.

核心逻辑:
    - position * clip(target_vol / realized_vol, upper=1.0)
    - 高波动时降低仓位, 低波动时满仓

参考实现:
    - QuantNodes/strategy/commodity_fundamental/copper/copper_momentum.py (vol_target inline)
    - QuantNodes/strategy/cta/factors/utils/vol_target.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vol_target_positions(
    position: pd.Series,
    volatility: pd.Series,
    target_vol: float = 0.12,
    upper_cap: float = 1.0,
) -> pd.Series:
    """波动率目标化仓位.

    Args:
        position: 原始仓位 [0, 1]
        volatility: 波动率序列 (年化)
        target_vol: 目标波动率 (年化)
        upper_cap: 缩放上限 (默认1.0, 不放大)

    Returns:
        波动率目标化后的仓位 [0, 1]
    """
    scale = (target_vol / volatility).fillna(1.0).clip(upper=upper_cap)
    return (position * scale).clip(0, upper_cap)


def realized_volatility(
    close: pd.Series,
    window: int = 13,
    periods_per_year: int = 52,
) -> pd.Series:
    """计算滚动年化波动率.

    Args:
        close: 价格序列
        window: 滚动窗口
        periods_per_year: 年化周期数 (周频52, 日频252)

    Returns:
        年化波动率序列
    """
    return close.pct_change().rolling(window).std() * np.sqrt(periods_per_year)


__all__ = [
    "vol_target_positions",
    "realized_volatility",
]
