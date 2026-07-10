"""v7.0 流动性 cap 模块 (Stage 30.5 Phase A2).

[动机] 单 ETF 集中度过高, 极端行情可能 100% 集中到小盘 ETF, 冲击成本不可忽略.
       加入流动性 cap: 单 ETF 30% 权重上限 + 单 ETF 月度换手率 30% ADV 上限.

[核心]
    1. apply_max_weight_cap(weights, max_weight=0.30)  ← 已在 IV 方案中, 抽出共用
    2. apply_turnover_cap(weights_new, weights_old, max_turnover=0.30)
       单 ETF 月度最大换手 30% (相对自身权重)

[业界惯例]
    - 大型 ETF (510300, 510500) 日均成交 > 5 亿, 流动性充裕, 30% cap 不限
    - 中型 ETF (512760, 510880) 日均成交 1-3 亿, 30% cap 适中
    - 小型 ETF (< 1 亿) 应有更严 cap (e.g., 10%)
"""
from __future__ import annotations

from typing import Optional


def _iterative_cap(weights: dict[str, float], max_weight: float, max_iter: int = 50) -> dict[str, float]:
    """迭代 cap+redistribute, 直到所有权重 ≤ max_weight.

    算法:
        1. 任何 w > max_weight → cap 到 max_weight, 多余 excess 累积
        2. excess 按比例分配给 w < max_weight 的 ETF (按 (max-w) 加权)
        3. 重复直到 excess ≈ 0
    """
    if not weights:
        return {}
    w = {c: float(v) for c, v in weights.items()}
    for _ in range(max_iter):
        excess = 0.0
        for c in w:
            if w[c] > max_weight:
                excess += w[c] - max_weight
                w[c] = max_weight
        if excess < 1e-12:
            break
        below = {c: max_weight - w[c] for c in w if w[c] < max_weight}
        free = sum(below.values())
        if free < 1e-12:
            break
        for c, space in below.items():
            w[c] += space / free * excess
    s = sum(w.values())
    if s < 1e-9:
        n = len(w)
        return {c: 1.0 / n for c in w}
    return {c: v / s for c, v in w.items()}


def apply_max_weight_cap(weights: dict[str, float], max_weight: float = 0.30) -> dict[str, float]:
    """单 ETF 权重上限 cap, 重新归一化 (迭代 cap).

    Args:
        weights: 目标权重.
        max_weight: 单 ETF 最大权重, 默认 0.30.

    Returns:
        dict[etf_code] -> capped_weight, sum=1, all ≤ max_weight.
    """
    return _iterative_cap(weights, max_weight)


def apply_turnover_cap(
    weights_new: dict[str, float],
    weights_old: Optional[dict[str, float]],
    max_turnover: float = 0.30,
) -> dict[str, float]:
    """单 ETF 月度换手率 cap (相对自身权重), 迭代 cap.

    Args:
        weights_new: 目标权重.
        weights_old: 当前权重 (None = 第一次建仓).
        max_turnover: 单 ETF 最大换手率, 默认 0.30.

    Returns:
        dict[etf_code] -> capped_weight, sum=1.
    """
    if not weights_new or weights_old is None:
        return weights_new
    capped = {}
    for c, w_new in weights_new.items():
        w_old = weights_old.get(c, 0.0)
        delta = w_new - w_old
        max_delta = max_turnover * max(w_old, 0.01)
        if abs(delta) > max_delta:
            delta = max_delta * (1 if delta > 0 else -1)
        capped[c] = w_old + delta
    s = sum(capped.values())
    if s < 1e-9:
        n = len(capped)
        return {c: 1.0 / n for c in capped}
    return _iterative_cap(capped, max_weight=1.0)
