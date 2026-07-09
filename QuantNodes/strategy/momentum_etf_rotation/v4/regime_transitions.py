# coding=utf-8
"""状态转移矩阵 — 距离驱动的金融常识先验 (Stage 17, v4.0).

核心思想: 状态空间 1D 排列
    bear(0) ─── transition(1) ─── bull(2)

距离 d(i, j) = |i - j| 反映转换难度.

转移率:
    rate[i, j] = exp(-α × d_eff(i, j))
    d_eff = d + γ × (potential[i] - potential[j])  # 势能差微调 (从高到低需要能量)

矩阵归一化后, 用作 HMM 的 transmat_prior 或固定转移矩阵.

参考: reports/momentum_etf_rotation/v4/HMM_DISTANCE_PLAN.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


# 状态 1D 势能
POTENTIAL: dict[int, float] = {
    0: 0.0,   # bear
    1: 0.5,   # transition
    2: 1.0,   # bull
}


@dataclass
class DistanceTransitionConfig:
    """距离驱动转移矩阵配置."""
    alpha: float = 1.5           # 距离衰减率 (越大越粘)
    gamma: float = 0.3           # 势能差权重 (0=纯距离, 1=强势能)
    sticky_bonus: float = 0.0    # 自循环额外加成 (0=靠距离自然)
    n_states: int = 3            # 状态数 (默认 3)


def distance_between(state_i: int, state_j: int) -> int:
    """1D 距离 d(i, j) = |i - j|."""
    return abs(state_i - state_j)


def effective_distance(
    state_i: int,
    state_j: int,
    gamma: float = 0.3,
) -> float:
    """有效距离 = 几何距离 + 势能差微调.

    势能差 (pot[i] - pot[j]) 反映"向下"难度:
    - bull → bear: pot 1-0=+1, 加上 γ 距离更远 (崩溃需要能量)
    - bear → bull: pot 0-1=-1, 加上 γ 距离更近 (反弹相对容易)
    """
    d = distance_between(state_i, state_j)
    pot_i = POTENTIAL.get(state_i, 0.5)
    pot_j = POTENTIAL.get(state_j, 0.5)
    pot_diff = pot_i - pot_j  # 修正: i - j (从高到低需要能量)
    return d + gamma * pot_diff


def distance_rate(
    state_i: int,
    state_j: int,
    alpha: float = 1.5,
    gamma: float = 0.3,
    sticky_bonus: float = 0.0,
) -> float:
    """距离驱动的转移率.

    Args:
        state_i: 源状态
        state_j: 目标状态
        alpha: 距离衰减率
        gamma: 势能权重
        sticky_bonus: 自循环额外加成

    Returns:
        rate: 未归一化的转移率 (用于构建矩阵)
    """
    eff_d = effective_distance(state_i, state_j, gamma)
    rate = np.exp(-alpha * eff_d)
    if state_i == state_j:
        rate *= (1 + sticky_bonus)
    return float(rate)


def build_distance_transmat(
    alpha: float = 1.5,
    gamma: float = 0.3,
    sticky_bonus: float = 0.0,
    n_states: int = 3,
) -> np.ndarray:
    """构建 n_states × n_states 距离驱动转移矩阵 (row=from, col=to).

    Args:
        alpha: 距离衰减率 (推荐 1.5: 中等)
        gamma: 势能差权重 (推荐 0.3: 轻度)
        sticky_bonus: 自循环额外加成
        n_states: 状态数

    Returns:
        np.ndarray, 形状 (n_states, n_states), 每行 sum=1
    """
    out = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            out[i, j] = distance_rate(i, j, alpha, gamma, sticky_bonus)

    # 归一化
    row_sums = out.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    out = out / row_sums
    return out


def soft_constrain(
    learned: np.ndarray,
    prior: np.ndarray,
    lam: float = 0.3,
) -> np.ndarray:
    """软约束: 混合 learned 和 prior.

    transmat = (1-λ) × learned + λ × prior
    重新归一化每行.

    Args:
        learned: HMM 学到的转移矩阵
        prior: 距离先验矩阵
        lam: 先验权重 (0=纯 learned, 1=纯 prior)

    Returns:
        混合后的转移矩阵
    """
    if not (0 <= lam <= 1):
        raise ValueError(f"lam 必须在 [0, 1], 当前 {lam}")

    mixed = (1 - lam) * learned + lam * prior
    row_sums = mixed.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    return mixed / row_sums


def enforce_minimum_duration(
    labels: np.ndarray,
    min_duration: int = 30,
) -> np.ndarray:
    """强制最小持续期.

    短于 min_duration 的状态序列被合并到相邻状态 (取较长一侧).

    Args:
        labels: 状态序列 (1D array)
        min_duration: 最小持续期 (天数)

    Returns:
        合并后的状态序列
    """
    if len(labels) == 0 or min_duration <= 1:
        return labels.copy()

    out = labels.copy()
    n = len(out)
    i = 0
    while i < n:
        # 找当前状态的结束位置
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        run_length = j - i
        if run_length < min_duration:
            # 短状态: 合并到左侧 (i-1) 或 右侧 (j)
            if i == 0 and j < n:
                # 在开头, 合并到右侧
                target = out[j]
            elif j == n and i > 0:
                # 在结尾, 合并到左侧
                target = out[i - 1]
            else:
                # 中间, 取相邻中较长的
                left_count = 0
                k = i - 1
                while k >= 0 and out[k] == out[k + 1]:
                    left_count += 1
                    k -= 1
                right_count = 0
                k = j
                while k < n and out[k] == out[k - 1]:
                    right_count += 1
                    k += 1
                target = out[i - 1] if left_count >= right_count else out[j]
            out[i:j] = target
        i = j

    return out


def validate_transmat(transmat: np.ndarray) -> dict:
    """验证转移矩阵的合理性.

    Returns:
        dict, 包含特性 (sticky, asymmetric, valid)
    """
    n = transmat.shape[0]
    if transmat.shape != (n, n):
        return {"valid": False, "reason": "non-square"}

    # 归一化
    row_sums = transmat.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        return {"valid": False, "reason": f"rows don't sum to 1: {row_sums}"}

    # 非负
    if (transmat < 0).any():
        return {"valid": False, "reason": "negative entries"}

    # 特性
    diag = np.diag(transmat)
    off_diag_max = transmat - np.diag(np.diag(transmat))
    off_diag_max = off_diag_max.max()

    # bull ↔ bear (两端跳跃)
    if n == 3:
        bull_bear = transmat[2, 0] + transmat[0, 2]  # 双向
        trans_bull = transmat[1, 2] + transmat[1, 0]  # transition 出去
    else:
        bull_bear = 0.0
        trans_bull = 0.0

    return {
        "valid": True,
        "n": n,
        "diag_mean": float(diag.mean()),
        "off_diag_max": float(off_diag_max),
        "bull_bear_direct": float(bull_bear),
        "trans_to_extremes": float(trans_bull),
    }


__all__ = [
    "DistanceTransitionConfig",
    "POTENTIAL",
    "distance_between",
    "effective_distance",
    "distance_rate",
    "build_distance_transmat",
    "soft_constrain",
    "enforce_minimum_duration",
    "validate_transmat",
]
