# coding=utf-8
"""AllWeather V6 子因子周频重采样 + z-score 标准化 (PLAN §13.3.2).

将日度因子分数 (T × N) 转为周频分数 (W × N):
    1. resample_weekly: W-FRI 重采样 (last/mean)
    2. zscore_cross_section: 横截面滚动 z-score (52 周窗口)
    3. combine_subfactors: 三子因子加权求和 → 综合分数

参照黄金 gold_combined.py: 6 因子等权 Sharpe 1.37, OOS/IS=4.47.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd


WeeklyAgg = Literal["last", "mean"]


def resample_weekly(
    factor_daily: pd.DataFrame,
    agg: WeeklyAgg = "last",
) -> pd.DataFrame:
    """日度因子 → W-FRI 重采样.

    Args:
        factor_daily: T × N 日度因子分数.
        agg: 'last' (取周五最后值, 默认, 对齐 W-FRI 调仓)
             'mean' (取周内均值, 更稳健).

    Returns:
        W × N 周频因子分数 (index 为 W-FRI).
    """
    if agg == "last":
        return factor_daily.resample("W-FRI").last()
    elif agg == "mean":
        return factor_daily.resample("W-FRI").mean()
    else:
        raise ValueError(f"agg={agg}, 可选 last/mean")


def zscore_cross_section(
    factor_panel: pd.DataFrame,
    lookback_weeks: int = 52,
    min_periods: int = 12,
) -> pd.DataFrame:
    """横截面滚动 z-score 标准化 (时间序列滚动, 每个时点横截面标准化).

    factor_panel: W × N (周频).
    输出: 每个时点, 对 N 个资产做滚动 z-score (rolling lookback_weeks).
          消除资产间量纲差异, 使因子分数可比较.

    窗口: 默认 52 周 (1 年), 与黄金多因子 OOS/IS=4.47 基准一致.
    """
    if factor_panel.empty:
        return factor_panel.copy()

    # 时间序列 rolling mean/std (沿 axis=0, 对每列单独)
    rolling_mean = factor_panel.rolling(
        lookback_weeks, min_periods=min_periods
    ).mean()
    rolling_std = factor_panel.rolling(
        lookback_weeks, min_periods=min_periods
    ).std()

    z = (factor_panel - rolling_mean) / rolling_std.replace(0, np.nan)
    return z


def combine_subfactors(
    factors: dict[str, pd.DataFrame],
    weights: Optional[dict[str, float]] = None,
    lookback_weeks: int = 52,
    min_periods: int = 12,
) -> pd.DataFrame:
    """组合多子因子 → 单一综合分数 (W × N).

    步骤:
      1. 每个子因子横截面 z-score 标准化
      2. 加权求和 (默认等权 {a:0.33, b:0.33, c:0.34})
      3. 输出 W × N 综合分数

    Args:
        factors: {"a": W×N, "b": W×N, "c": W×N}.
        weights: {"a": 0.33, "b": 0.33, "c": 0.34}, 默认等权.

    Returns:
        W × N 综合因子分数 (z-score 加权和).
    """
    if not factors:
        raise ValueError("factors 不能为空")

    if weights is None:
        n = len(factors)
        weights = {k: 1.0 / n for k in factors}

    weight_sum = sum(weights.get(k, 0.0) for k in factors)
    if weight_sum <= 0:
        raise ValueError("weights 权重总和必须 > 0")

    combined = None
    for k, factor in factors.items():
        if factor.empty:
            continue
        z = zscore_cross_section(factor, lookback_weeks, min_periods)
        contrib = z * (weights.get(k, 0.0) / weight_sum)
        combined = contrib if combined is None else combined + contrib

    if combined is None:
        raise ValueError("所有 factors 都为空")

    return combined
