"""动量倾斜 alpha 层 (D.2, PLAN §17.14): 双向 t-stat 倾斜因子.

与 D.1 负趋势因子同源 (风险调整动量 mom/vol = t-stat), 方向双向化:
- D.1 (单向风控): t-stat 为负 → 减仓 (penalty)
- D.2 (双向 alpha): t-stat 为正 → 超配 (补仓确认), 为负 → 低配

倾斜因子: t_i = 1 + β·score_i, score = tanh(0.5·t-stat) (与 D.1 修复后
sub_trend 同源同分布, 健康范围 [−0.61, +0.67]).
高位减仓低位补仓闭环: D.1 管减仓 (趋势破坏), D.2 管补仓 (动量拐正超配).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_score(panel: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """t-stat 风险调整动量 → tanh(0.5·t-stat) ∈ [−1, 1].

    与 neg_trend_daily._slope_vol_adj 同源: win 日累计收益 / (vol·√win).
    """
    ret = panel.pct_change(fill_method=None)
    vol = ret.rolling(lookback, min_periods=max(lookback // 2, 10)).std()
    mom = ret.rolling(lookback).sum()
    tstat = mom / (vol.replace(0, np.nan) * np.sqrt(lookback))
    return np.tanh(0.5 * tstat).fillna(0.0)


def momentum_tilt_factors(panel: pd.DataFrame, lookback: int = 60,
                          beta: float = 0.20) -> pd.DataFrame:
    """双向倾斜因子 t_i ∈ [1−β, 1+β] (T×N, 周频调仓时取行).

    Args:
        panel: 价格面板
        lookback: 动量窗口 (t-stat)
        beta: 倾斜幅度上限 (score=±1 时达到; score 分布实际 ±0.67 → ±0.67β)

    Returns:
        tilt DataFrame, 每行 t_i; 晚上市资产 score=0 → t=1 (中性)
    """
    score = momentum_score(panel, lookback)
    tilt = 1.0 + beta * score
    return tilt.clip(1.0 - beta, 1.0 + beta)


def apply_momentum_tilt(weights: pd.Series, tilt_row: pd.Series,
                        min_obs: int = 60) -> pd.Series:
    """单次调仓: w_tilt = w_base × t_i (归一化; 量不足资产中性)."""
    if weights.abs().sum() <= 0:
        return weights
    t = tilt_row.reindex(weights.index)
    # 量不足 (动量 NaN) 的资产 → 中性 1.0
    t = t.where(t.notna(), 1.0)
    # 从未上市资产 (权重=0) 保持 0
    t = t.where(weights > 0, 1.0)
    w_tilt = weights * t
    total = w_tilt.sum()
    if total > 0:
        w_tilt = w_tilt / total
    return w_tilt
