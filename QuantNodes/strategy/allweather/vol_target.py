# coding=utf-8
"""AllWeather — 波动率目标仓位 (PLAN Step 4).

规则: 组合已实现波动 (lookback 日) → 杠杆 = target_vol / realized_vol,
cap 到 max_leverage → 基础权重整体缩放.

无前视约定: 杠杆[t] 使用截至 t-1 日的组合收益 (shift 1 天).
杠杆估计不足 (warmup) → 杠杆 = 1 (不缩放).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = 252.0


def apply_vol_target(
    base_weights: pd.DataFrame,
    returns: pd.DataFrame,
    target_vol: float = 0.10,
    lookback: int = 20,
    max_leverage: float = 3.0,
) -> pd.DataFrame:
    """波动率目标缩放.

    Args:
        base_weights: (T x N) 基础权重 (调仓日更新, 持有期不变)
        returns: (T x N) 日收益率
        target_vol: 目标年化波动
        lookback: 已实现波动回看窗口
        max_leverage: 杠杆上限

    Returns:
        (T x N) 缩放后权重 = base_weights × leverage
    """
    # 组合历史收益: 以各日权重 × 各日收益近似 (仅用于波动估计)
    port_ret = (base_weights * returns).sum(axis=1)
    realized = port_ret.rolling(lookback, min_periods=lookback).std() * np.sqrt(ANN)
    realized = realized.shift(1)  # 无前视: t 日杠杆用截至 t-1 的已实现波动

    leverage = (target_vol / realized).clip(upper=max_leverage)
    leverage = leverage.fillna(1.0).clip(lower=0.0)

    return base_weights.mul(leverage, axis=0)
