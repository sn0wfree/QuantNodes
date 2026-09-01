# coding=utf-8
"""commodity_timing.py — 通用商品因子开仓时机过滤.

数据源无关: 接收仓位序列 + 时机指标作为参数.

核心模式:
    momentum > 0 OR vol < threshold → 开仓, 否则空仓

应用场景:
    - 铜: momentum > 0 OR vol < 70th percentile → 开仓
    - 黄金: 类似模式用于综合信号

参考实现:
    - QuantNodes/strategy/commodity_fundamental/copper/copper_combined.py
"""
from __future__ import annotations

import pandas as pd


def momentum_or_vol_filter(
    position: pd.Series,
    momentum: pd.Series,
    volatility: pd.Series,
    vol_percentile: float = 0.7,
    vol_window: int = 52,
) -> pd.Series:
    """因子开仓时机过滤: 动量为正 或 波动率低于阈值 → 允许开仓.

    Args:
        position: 原始仓位序列
        momentum: 动量信号 (如收益率)
        volatility: 波动率序列
        vol_percentile: 波动率分位数阈值 (低于此分位数 → 允许开仓)
        vol_window: 波动率分位数计算窗口

    Returns:
        过滤后的仓位序列 (时机不对时清仓)
    """
    vol_threshold = volatility.rolling(vol_window).quantile(vol_percentile)
    timing = (momentum > 0) | (volatility < vol_threshold)
    return position.where(timing, 0.0)


def momentum_filter(
    position: pd.Series,
    momentum: pd.Series,
) -> pd.Series:
    """简单动量过滤: 动量 > 0 → 允许开仓.

    Args:
        position: 原始仓位序列
        momentum: 动量信号

    Returns:
        过滤后的仓位序列
    """
    return position.where(momentum > 0, 0.0)


__all__ = [
    "momentum_or_vol_filter",
    "momentum_filter",
]
