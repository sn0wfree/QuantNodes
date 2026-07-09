# coding=utf-8
"""子策略权重分配 (Stage 16A, v3.0).

多策略组合的关键问题: 如何在子策略之间分配权重?

支持 3 种方法:
    1. equal_sub_weights: 等权 (1/N 简单 baseline)
    2. risk_parity_sub_weights: 风险平价 (用 Ledoit-Wolf 子策略协方差)
    3. signal_weighted_sub_weights: 信号加权 (按 signal_strength 加权)

参考: reports/momentum_etf_rotation/v2/stage16a_plan.md §2.2
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .sub_strategy_v3 import SubStrategyResult


def equal_sub_weights(
    strategy_names: Sequence[str] | int,
) -> dict[str, float]:
    """等权分配 (简单 baseline).

    Args:
        strategy_names: 子策略名列表, 或数量 (int)
                       传 int 时用 strategy_0, strategy_1, ... 占位

    Returns:
        dict[str, float]: name -> 权重 (1/N)
    """
    if isinstance(strategy_names, int):
        if strategy_names <= 0:
            return {}
        names = [f"strategy_{i}" for i in range(strategy_names)]
    else:
        names = list(strategy_names)
        if not names:
            return {}

    w = 1.0 / len(names)
    return {name: w for name in names}


def sub_weights_from_results(
    sub_results: Sequence[SubStrategyResult],
    method: str = "equal",
) -> dict[str, float]:
    """从子策略结果直接构造子策略权重.

    Args:
        sub_results: 子策略结果列表
        method: "equal" / "signal" (signal 加权)

    Returns:
        dict[str, float]: 子策略 name -> 权重
    """
    names = [r.meta.get("strategy", f"strategy_{i}") for i, r in enumerate(sub_results)]
    if not names:
        return {}
    if method == "signal":
        return signal_weighted_sub_weights(sub_results)
    return equal_sub_weights(names)


def risk_parity_sub_weights(
    sub_navs: pd.DataFrame,
    method: str = "sample",
    halflife: int = 60,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> dict[str, float]:
    """风险平价子策略权重.

    用子策略 NAV 序列计算协方差, 求子策略权重使风险贡献相等.

    Args:
        sub_navs: 子策略 NAV 序列, columns=name, index=date
        method: "sample" / "ledoit_wolf" / "ewma"
        halflife: EWMA halflife (仅 method="ewma")
        max_iter: SLSQP 最大迭代
        tol: SLSQP 容差

    Returns:
        dict[str, float]: name -> 子策略权重
    """
    from ..common.covariance import estimate_covariance
    from ..common.risk_parity import solve_risk_parity

    if sub_navs.empty or sub_navs.shape[1] == 0:
        return equal_sub_weights(0)

    # 1. 算子策略日收益
    rets = sub_navs.pct_change().dropna()
    if rets.empty or len(rets) < 5:
        return equal_sub_weights(sub_navs.shape[1])

    # 2. 协方差矩阵
    try:
        cov = estimate_covariance(rets, method=method, halflife=halflife)
    except Exception:
        # 协方差失败 → 等权
        return equal_sub_weights(sub_navs.shape[1])

    # 3. 风险平价求解
    try:
        weights = solve_risk_parity(cov, max_iter=max_iter, tol=tol)
    except Exception:
        return equal_sub_weights(sub_navs.shape[1])

    return dict(zip(sub_navs.columns, weights))


def signal_weighted_sub_weights(
    sub_results: Sequence[SubStrategyResult],
    signal_clip: float = 1.0,
) -> dict[str, float]:
    """信号加权子策略权重.

    按 signal_strength 加权, signal_strength 越强权重越大.
    signal_strength 范围约 [-1, 1+], 用 signal_clip 限制.

    Args:
        sub_results: 子策略结果列表
        signal_clip: signal_strength clip 范围 [-signal_clip, signal_clip]

    Returns:
        dict[str, float]: name -> 子策略权重
    """
    if not sub_results:
        return {}

    # 用 meta.strategy 区分
    signals = {}
    for r in sub_results:
        name = r.meta.get("strategy", "unknown")
        s = float(np.clip(r.signal_strength, -signal_clip, signal_clip))
        # 移到非负区间: 1 + s (signal=-1 → 0, signal=0 → 0.5, signal=+1 → 1)
        signals[name] = max(0.0, 1.0 + s)

    total = sum(signals.values())
    if total <= 0:
        return {name: 1.0 / len(signals) for name in signals}

    return {name: w / total for name, w in signals.items()}


def combine_sub_results(
    sub_results: Sequence[SubStrategyResult],
    sub_weights: dict[str, float],
    pool_codes: set[str] | None = None,
) -> dict[str, float]:
    """合并多个子策略结果为最终权重.

    算法:
        1. 用 sub_weights[name] 加权
        2. 同一 ETF 被多个子策略选中 → 权重累加
        3. 归一化到 1
        4. 可选: 应用 max_weight 约束 (通过 pool_codes 过滤非法)

    Args:
        sub_results: 子策略结果列表
        sub_weights: name -> 权重
        pool_codes: 池子 code 集合 (None = 不过滤)

    Returns:
        dict[str, float]: code -> 合并后权重
    """
    combined: dict[str, float] = {}
    for r in sub_results:
        name = r.meta.get("strategy", "unknown")
        sub_w = sub_weights.get(name, 0.0)
        for code, w in r.weights.items():
            if pool_codes is not None and code not in pool_codes:
                continue
            combined[code] = combined.get(code, 0.0) + sub_w * w

    # 归一化
    total = sum(combined.values())
    if total > 0:
        combined = {c: w / total for c, w in combined.items()}

    return combined


__all__ = [
    "equal_sub_weights",
    "sub_weights_from_results",
    "risk_parity_sub_weights",
    "signal_weighted_sub_weights",
    "combine_sub_results",
]
