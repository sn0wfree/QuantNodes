# coding=utf-8
"""自适应因子筛选器 — 统一 IC 计算 + 自适应 L1 权重 [ARCHIVED: 未集成到 TV-PR 主线].

功能:
  1. 统一 IC 计算: 宏观因子用面板 IC，量价因子用截面 IC
  2. 自适应权重: 基于 IC / 历史 β / 混合方法
  3. 因子选择: top-K 或 threshold 方法

用于 TV-PR 模型的自适应 L1 罚项:
  - 因子表现好 → w_k 小 → β_k 允许非零 → 因子被选中
  - 因子表现差 → w_k 大 → β_k 被压缩到零 → 因子被淘汰

数学形式:
  原: λ_l1 Σ_t Σ_k |β_{t,k}|          (均匀惩罚)
  新: Σ_t Σ_k w_k(t) |β_{t,k}|        (自适应惩罚)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# 配置
# ============================================================
@dataclass
class AdaptiveFactorConfig:
    """自适应因子筛选配置."""

    # IC 计算
    ic_window: int = 52  # IC 滚动窗口 (周)
    ic_method: Literal["panel", "cross_sectional", "unified"] = "unified"
    min_obs: int = 20  # 最少观测数

    # 自适应权重
    adaptive: bool = True  # 是否启用自适应
    weight_method: Literal["ic", "beta", "hybrid"] = "ic"
    alpha: float = 0.5  # hybrid 方法的混合系数 (IC 权重)
    epsilon: float = 0.01  # 避免除零
    gamma: float = 0.1  # IC 指数平滑系数

    # 权重范围
    weight_min_ratio: float = 0.5  # 最小权重 = lambda_l1 * weight_min_ratio (重要因子)
    weight_max_ratio: float = 1.0  # 最大权重 = lambda_l1 * weight_max_ratio (不重要因子)

    # 因子选择
    min_active_factors: int = 5  # 最少活跃因子数

    # 宏观因子掩码 (前 K_macro 个因子是宏观因子)
    K_macro: int = 12  # 宏观因子数量


# ============================================================
# IC 计算
# ============================================================
def compute_panel_ic(
    X_panel: np.ndarray,
    Y: np.ndarray,
    factor_idx: int,
    window: int,
    min_obs: int = 20,
) -> float:
    """面板 IC: 因子_t vs 市场平均收益_{t+1}.

    用于宏观因子 (所有资产相同).
    """
    T, N, K = X_panel.shape

    # 宏观因子所有资产相同，取第一个
    x = X_panel[-window:, 0, factor_idx]  # (W,)
    # 市场平均收益 (下期)
    market_r = np.nanmean(Y[-window+1:], axis=1)  # (W-1,)
    # 对齐: x[:-1] vs market_r
    x_aligned = x[:-1]

    valid = ~np.isnan(x_aligned) & ~np.isnan(market_r)
    if valid.sum() < min_obs:
        return 0.0

    ic, _ = spearmanr(x_aligned[valid], market_r[valid])
    return ic if not np.isnan(ic) else 0.0


def compute_cross_sectional_ic(
    X_panel: np.ndarray,
    Y: np.ndarray,
    factor_idx: int,
    window: int,
    min_obs: int = 20,
    min_assets: int = 10,
) -> float:
    """截面 IC: 因子_{i,t} vs 资产 i 收益_{t+1} (均值).

    用于量价因子 (资产特异).
    """
    T, N, K = X_panel.shape

    ic_list = []
    for t in range(max(0, T - window), T - 1):
        x = X_panel[t, :, factor_idx]  # (N,)
        r = Y[t + 1]  # (N,)

        valid = ~np.isnan(x) & ~np.isnan(r)
        if valid.sum() < min_assets:
            continue

        ic, _ = spearmanr(x[valid], r[valid])
        if not np.isnan(ic):
            ic_list.append(ic)

    return np.mean(ic_list) if ic_list else 0.0


def compute_unified_ic(
    X_panel: np.ndarray,
    Y: np.ndarray,
    factor_idx: int,
    is_macro: bool,
    window: int,
    min_obs: int = 20,
) -> float:
    """统一 IC: 宏观用面板 IC，量价用截面 IC."""
    if is_macro:
        return compute_panel_ic(X_panel, Y, factor_idx, window, min_obs)
    else:
        return compute_cross_sectional_ic(X_panel, Y, factor_idx, window, min_obs)


def compute_rolling_ic(
    X_panel: np.ndarray,
    Y: np.ndarray,
    K_macro: int,
    window: int = 52,
    ic_method: str = "unified",
    min_obs: int = 20,
    gamma: float = 0.1,
) -> np.ndarray:
    """计算所有因子的滚动 IC (带指数平滑).

    Parameters:
        X_panel: (T, N, K) 因子值面板
        Y: (T, N) 资产收益
        K_macro: 宏观因子数量
        window: IC 滚动窗口
        ic_method: "panel", "cross_sectional", "unified"
        min_obs: 最少观测数
        gamma: 指数平滑系数

    Returns:
        ic_path: (T, K) 滚动 IC 路径 (已平滑)
    """
    T, N, K = X_panel.shape
    ic_raw = np.zeros((T, K))

    # 预计算市场平均收益
    market_r = np.nanmean(Y, axis=1)  # (T,)

    for t in range(window, T):
        start = max(0, t - window)

        for k in range(K):
            is_macro = k < K_macro

            if is_macro:
                # 面板 IC: 因子_t vs 市场收益_{t+1}
                x = X_panel[start:t, 0, k]  # (W,)
                r = market_r[start+1:t+1]     # (W,)
                valid = ~np.isnan(x) & ~np.isnan(r)
                if valid.sum() >= min_obs:
                    ic, _ = spearmanr(x[valid], r[valid])
                    ic_raw[t, k] = ic if not np.isnan(ic) else 0.0
            else:
                # 截面 IC: 因子_{i,t} vs 资产 i 收益_{t+1} (均值)
                ic_list = []
                for s in range(start, t):
                    x = X_panel[s, :, k]  # (N,)
                    r = Y[s + 1]           # (N,)
                    valid = ~np.isnan(x) & ~np.isnan(r)
                    if valid.sum() >= 10:
                        ic, _ = spearmanr(x[valid], r[valid])
                        if not np.isnan(ic):
                            ic_list.append(ic)
                ic_raw[t, k] = np.mean(ic_list) if ic_list else 0.0

    # 指数平滑 (减少噪声)
    ic_smoothed = np.zeros_like(ic_raw)
    ic_smoothed[window] = ic_raw[window]
    for t in range(window + 1, T):
        ic_smoothed[t] = gamma * ic_raw[t] + (1 - gamma) * ic_smoothed[t - 1]

    return ic_smoothed


# ============================================================
# 自适应权重
# ============================================================
def compute_adaptive_weights(
    ic_path: np.ndarray,
    lambda_l1: float,
    method: str = "ic",
    beta_prev: np.ndarray | None = None,
    alpha: float = 0.5,
    epsilon: float = 0.01,
    weight_min_ratio: float = 0.1,
    weight_max_ratio: float = 10.0,
) -> np.ndarray:
    """计算自适应惩罚权重 w_k(t).

    Parameters:
        ic_path: (T, K) 滚动 IC 路径
        lambda_l1: 基础 L1 惩罚系数
        method: "ic", "beta", "hybrid"
        beta_prev: (T, K) 上一轮估计的 β (用于 beta/hybrid 方法)
        alpha: hybrid 方法的混合系数 (IC 权重)
        epsilon: 避免除零
        weight_min_ratio: 最小权重比率
        weight_max_ratio: 最大权重比率

    Returns:
        weights: (T, K) 自适应权重 (已 clip 到合理范围)
    """
    T, K = ic_path.shape

    if method == "ic":
        # w_k(t) = λ_l1 / (|IC_k(t)| + ε)
        importance = np.abs(ic_path) + epsilon
        weights = lambda_l1 / importance

    elif method == "beta":
        # w_k(t) = λ_l1 / (|β_{t-1,k}| + ε)
        if beta_prev is None:
            raise ValueError("beta_prev is required for 'beta' method")
        importance = np.abs(beta_prev) + epsilon
        weights = lambda_l1 / importance

    elif method == "hybrid":
        # w_k(t) = λ_l1 / (α*|IC_k(t)| + (1-α)*|β_{t-1,k}| + ε)
        if beta_prev is None:
            raise ValueError("beta_prev is required for 'hybrid' method")
        ic_importance = np.abs(ic_path)
        beta_importance = np.abs(beta_prev)
        importance = alpha * ic_importance + (1 - alpha) * beta_importance + epsilon
        weights = lambda_l1 / importance

    else:
        raise ValueError(f"Unknown method: {method}")

    # 限制权重范围
    weight_min = lambda_l1 * weight_min_ratio
    weight_max = lambda_l1 * weight_max_ratio
    weights = np.clip(weights, weight_min, weight_max)

    return weights


def compute_adaptive_weights_with_topk(
    ic_path: np.ndarray,
    lambda_l1: float,
    beta_path: np.ndarray | None = None,
    min_active: int = 5,
    epsilon: float = 0.01,
    weight_min_ratio: float = 0.5,
    weight_max_ratio: float = 1.0,
) -> np.ndarray:
    """计算自适应权重 (带 top-K 因子选择).

    策略: 降低重要因子的惩罚，而不是增加不重要因子的惩罚.
    这样可以避免过度稀疏化.

    Parameters:
        ic_path: (T, K) 滚动 IC 路径
        lambda_l1: 基础 L1 惩罚系数
        beta_path: (T, K) 估计的 β (可选)
        min_active: 最少活跃因子数
        epsilon: 避免除零
        weight_min_ratio: 最小权重比率 (重要因子)
        weight_max_ratio: 最大权重比率 (不重要因子)

    Returns:
        weights: (T, K) 自适应权重
    """
    T, K = ic_path.shape

    # 计算综合重要性 (IC + beta)
    if beta_path is not None:
        importance = 0.5 * np.abs(ic_path) + 0.5 * np.abs(beta_path)
    else:
        importance = np.abs(ic_path)

    # 归一化重要性到 [0, 1] 范围
    imp_min = np.nanmin(importance)
    imp_max = np.nanmax(importance)
    if imp_max > imp_min:
        importance_norm = (importance - imp_min) / (imp_max - imp_min)
    else:
        importance_norm = np.ones_like(importance) * 0.5

    # 权重范围
    weight_min = lambda_l1 * weight_min_ratio  # 重要因子的权重 (降低惩罚)
    weight_max = lambda_l1 * weight_max_ratio  # 不重要因子的权重 (保持原惩罚)

    # 计算权重: 重要性越高，权重越低
    # w_k(t) = weight_max - (weight_max - weight_min) * importance_norm_k(t)
    weights = weight_max - (weight_max - weight_min) * importance_norm

    # Top-K 因子选择: 确保至少 min_active 个因子权重最低
    for t in range(T):
        sorted_idx = np.argsort(importance[t])[::-1]  # 从大到小
        top_k_idx = sorted_idx[:min_active]
        weights[t, top_k_idx] = weight_min

    return weights


# ============================================================
# 因子选择分析
# ============================================================
def analyze_factor_selection(
    weights: np.ndarray,
    factor_names: list[str],
    threshold: float | None = None,
) -> pd.DataFrame:
    """分析因子选择结果.

    Parameters:
        weights: (T, K) 自适应权重
        factor_names: 因子名称列表
        threshold: 权重阈值 (低于此值视为被选中)

    Returns:
        df: 因子选择统计 DataFrame
    """
    T, K = weights.shape

    if threshold is None:
        threshold = np.median(weights) * 0.5

    results = []
    for k in range(K):
        # 被选中的时间比例 (权重 < 阈值)
        selected_ratio = np.mean(weights[:, k] < threshold)
        # 平均权重
        mean_weight = np.mean(weights[:, k])
        # 权重标准差
        std_weight = np.std(weights[:, k])

        results.append({
            'factor': factor_names[k],
            'selected_ratio': round(selected_ratio, 4),
            'mean_weight': round(mean_weight, 6),
            'std_weight': round(std_weight, 6),
            'rank': 0,  # 下面填充
        })

    df = pd.DataFrame(results)
    # 按选中比例排序
    df = df.sort_values('selected_ratio', ascending=False).reset_index(drop=True)
    df['rank'] = range(1, len(df) + 1)

    return df


# ============================================================
# 主接口
# ============================================================
def compute_adaptive_l1_weights(
    X_panel: np.ndarray,
    Y: np.ndarray,
    cfg: AdaptiveFactorConfig,
    lambda_l1: float,
    beta_path: np.ndarray | None = None,
) -> np.ndarray:
    """计算自适应 L1 权重 (主接口).

    Parameters:
        X_panel: (T, N, K) 因子值面板
        Y: (T, N) 资产收益
        cfg: 自适应因子配置
        lambda_l1: 基础 L1 惩罚系数
        beta_path: (T, K) 估计的 β (可选, 用于 beta/hybrid 方法)

    Returns:
        weights: (T, K) 自适应权重
    """
    if not cfg.adaptive:
        # 不启用自适应，返回均匀权重
        T, N, K = X_panel.shape
        return np.ones((T, K)) * lambda_l1

    # 1. 计算滚动 IC
    ic_path = compute_rolling_ic(
        X_panel, Y,
        K_macro=cfg.K_macro,
        window=cfg.ic_window,
        ic_method=cfg.ic_method,
        min_obs=cfg.min_obs,
        gamma=cfg.gamma,
    )

    # 2. 计算自适应权重
    if cfg.weight_method in ("beta", "hybrid") and beta_path is not None:
        weights = compute_adaptive_weights(
            ic_path, lambda_l1,
            method=cfg.weight_method,
            beta_prev=beta_path,
            alpha=cfg.alpha,
            epsilon=cfg.epsilon,
            weight_min_ratio=cfg.weight_min_ratio,
            weight_max_ratio=cfg.weight_max_ratio,
        )
    else:
        # 默认用 top-K 方法确保最少活跃因子数
        weights = compute_adaptive_weights_with_topk(
            ic_path, lambda_l1,
            beta_path=beta_path,
            min_active=cfg.min_active_factors,
            epsilon=cfg.epsilon,
            weight_min_ratio=cfg.weight_min_ratio,
            weight_max_ratio=cfg.weight_max_ratio,
        )

    return weights


__all__ = [
    "AdaptiveFactorConfig",
    "compute_panel_ic",
    "compute_cross_sectional_ic",
    "compute_unified_ic",
    "compute_rolling_ic",
    "compute_adaptive_weights",
    "compute_adaptive_weights_with_topk",
    "analyze_factor_selection",
    "compute_adaptive_l1_weights",
]
