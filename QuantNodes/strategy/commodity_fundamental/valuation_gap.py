# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/valuation_gap.py

§3.2 估值缺口 + 行业景气度叠加 (仅在去库阶段生效).

规则:
    被动去库 (基础1.0): 行业景气调头下行 且 估值高估 → 下调至 0.7
    主动去库 (基础0.4): 行业景气转多   或 估值低估   → 上修至 0.7

行业景气度 (各品种代理):
    铜   → 钻机数 4 周变化 z-score 的 4 周变化
    原油 → 钻机数 4 周变化 z-score 的 4 周变化
    豆粕 → 猪价指数 4 周变化 z-score 的 4 周变化
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sign_constrained_lasso import rolling_zscore


def industry_momentum(
    industry_series: pd.Series,
    z_window: int = 104,
    delta_window: int = 4,
) -> pd.Series:
    """行业景气度代理: z-score 的 4 周变化.

    返回值: > 0 转好, < 0 转差.
    """
    z = rolling_zscore(industry_series, window=z_window)
    return z.diff(delta_window).rename("industry_dz")


def apply_valuation_layer(
    quadrant_pos: pd.Series,
    gap: pd.Series,
    industry_dz: pd.Series,
    gap_low_threshold: float = 0.05,
    gap_high_threshold: float = -0.05,
) -> pd.Series:
    """§3.2 调仓规则.

    参数:
        gap_low_threshold: 估值缺口 > 此值视为低估
        gap_high_threshold: 估值缺口 < 此值视为高估
    """
    pos = quadrant_pos.copy()

    is_passive_destk = (quadrant_pos == 1.0)
    is_active_destk = (quadrant_pos == 0.4)
    industry_down = (industry_dz < 0)
    industry_up = (industry_dz > 0)
    undervalued = (gap > gap_low_threshold)
    overvalued = (gap < gap_high_threshold)

    pos[is_passive_destk & industry_down & overvalued] = 0.7
    pos[is_active_destk & (industry_up | undervalued)] = 0.7

    return pos


__all__ = ["industry_momentum", "apply_valuation_layer"]