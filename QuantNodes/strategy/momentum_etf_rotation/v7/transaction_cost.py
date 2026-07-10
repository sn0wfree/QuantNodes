"""v7.0 交易成本模块 (Stage 30.5 Phase A1).

[动机] 单次回测无交易成本, 26% 年化可能虚高 1-2%. 加入双边 0.1% × 月度 12 次
       = 1.2%/年 drag, 模拟真实交易.

[核心]
    1. apply_turnover_cost(weights_old, weights_new, fee_bps)
       返回调整后的 new_weights, 已扣手续费
    2. portfolio_drag(weights, fee_bps, rebal_freq_per_year)
       年化成本估算

[业界惯例]
    - A 股 ETF 双边: 买入 0.05% + 卖出 0.05% = 0.10% (无印花税, ETF 免)
    - 月度调仓: 12 次/年, 累计 1.2% 拖损
    - 高频 (周度) 调仓: 52 次/年, 5.2% 拖损 (显著)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_turnover(weights_new: dict[str, float], weights_old: Optional[dict[str, float]] = None) -> float:
    """调仓换手率 = Σ|w_new - w_old| / 2.

    Args:
        weights_new: 目标权重 dict.
        weights_old: 当前权重 dict (默认等权 = 第一次建仓).

    Returns:
        float, e.g., 0.5 表示 50% 仓位换手.
    """
    if weights_old is None:
        return sum(weights_new.values()) / 2
    keys = set(weights_new) | set(weights_old)
    return sum(abs(weights_new.get(k, 0) - weights_old.get(k, 0)) for k in keys) / 2


def apply_turnover_cost(
    weights_new: dict[str, float],
    weights_old: Optional[dict[str, float]],
    fee_bps: float = 10.0,
) -> dict[str, float]:
    """在调仓时扣手续费. 返回归一化后的 weights (sum=1).

    Args:
        weights_new: 目标权重.
        weights_old: 当前权重 (None = 第一次建仓, 视作等权 → 0).
        fee_bps: 双边手续费 (bps), 默认 10 = 0.10%.

    Returns:
        dict[etf_code] -> adjusted_weight, sum=1.
    """
    if not weights_new:
        return {}
    turnover = compute_turnover(weights_new, weights_old)
    cost_rate = turnover * fee_bps / 10000
    adj = {c: w * (1 - cost_rate) for c, w in weights_new.items()}
    s = sum(adj.values())
    if s < 1e-9:
        n = len(adj)
        return {c: 1.0 / n for c in adj}
    return {c: v / s for c, v in adj.items()}


def portfolio_drag(fee_bps: float = 10.0, rebal_freq_per_year: int = 12) -> float:
    """年化成本估算 (假设权重完全换手).

    Args:
        fee_bps: 双边手续费 (bps).
        rebal_freq_per_year: 调仓频率 (年化次数).

    Returns:
        float, e.g., 0.012 = 1.2%.
    """
    return fee_bps * rebal_freq_per_year / 10000
