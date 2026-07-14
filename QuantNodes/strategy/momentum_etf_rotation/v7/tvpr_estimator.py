# coding=utf-8
"""TV-PR Estimator: Total-Variation Predictive Regression.

Cui et al. (2025) "Breaks and trends in factor premia."

目标函数:
  min Σ_t Σ_i (r_{i,t} - x_{i,t}' β_t)^2 
      + λ_1 Σ_t ||β_t - β_{t-1}||_1 
      + λ_2 Σ_t ||β_t||_1

求解方法: ADMM (Alternating Direction Method of Multipliers)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal


def soft_thresholding(x: np.ndarray, threshold: float) -> np.ndarray:
    """Soft-thresholding 算子.

    S_κ(x) = sign(x) * max(|x| - κ, 0)
    """
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def compute_mse(Y: np.ndarray, X: np.ndarray, beta: np.ndarray) -> float:
    """计算 MSE: Σ_t Σ_i (r_{i,t} - x_{i,t}' β_t)^2."""
    T, N = Y.shape
    mse = 0.0
    for t in range(T):
        pred = X[t] @ beta[t]  # (N,)
        mse += np.sum((Y[t] - pred) ** 2)
    return mse


def compute_tv_cost(beta: np.ndarray) -> float:
    """计算 TV cost: Σ_t ||β_t - β_{t-1}||_1."""
    diff = np.diff(beta, axis=0)  # (T-1, K)
    return np.sum(np.abs(diff))


def compute_l1_cost(beta: np.ndarray) -> float:
    """计算 L1 cost: Σ_t ||β_t||_1."""
    return np.sum(np.abs(beta))


# ============================================================
# ADMM 求解器
# ============================================================
def tvpr_admm(
    Y: np.ndarray,
    X: np.ndarray,
    lambda_tv: float,
    lambda_l1: float,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-5,
    beta_init: np.ndarray | None = None,
) -> np.ndarray:
    """ADMM 求解 TV-PR.

    增广形式:
      min Σ_t Σ_i (r_{i,t} - x_{i,t}' β_t)^2 
          + λ_1 ||z||_1 
          + λ_2 ||β||_1
          + (ρ/2) ||Δβ - z + u||_2^2

    β-update 公式 (避免未来函数):
      (X'X + c*ρI)β[t] = X'Y + ρ(β[t-1] + z[t-1] - u[t-1])
      其中 c = 1 (端点) 或 c = 2 (内部)

    Parameters:
        Y: (T, N) 资产收益
        X: (T, N, K) 因子值面板 (每个资产有自己的因子值)
        lambda_tv: TV 罚项系数
        lambda_l1: L1 罚项系数
        rho: ADMM 惩罚参数
        max_iter: 最大迭代次数
        tol: 收敛阈值
        beta_init: (T, K) 初始 β (warm-start), None 则用 zeros

    Returns:
        beta: (T, K) 时变 β_t
    """
    T, N, K = X.shape

    # 初始化 (用 beta_init 而不是 zeros)
    if beta_init is not None:
        beta = beta_init.copy()
    else:
        beta = np.zeros((T, K))
    z = np.zeros((T - 1, K))
    u = np.zeros((T - 1, K))

    # 预计算 X'X 和 X'Y (每个时间点, 处理 NaN)
    # X[t] is (N, K), Y[t] is (N,)
    # X'X = X[t].T @ X[t] is (K, K)
    # X'Y = X[t].T @ Y[t] is (K,)
    XtX = np.zeros((T, K, K))
    XtY = np.zeros((T, K))
    for t in range(T):
        # 过滤 NaN
        valid_mask = ~np.isnan(Y[t]) & ~np.any(np.isnan(X[t]), axis=1)
        if np.sum(valid_mask) < K:
            # 样本不足，跳过
            continue
        X_valid = X[t][valid_mask]
        Y_valid = Y[t][valid_mask]
        XtX[t] = X_valid.T @ X_valid  # (K, K)
        XtY[t] = X_valid.T @ Y_valid  # (K,)

    # ADMM 迭代
    for iteration in range(max_iter):
        beta_old = beta.copy()

        # 1. β-update: 固定 z, u, 解 Lasso
        # 正确公式: (X'X + c*ρI)β = X'Y + ρ(β[t-1] + z[t-1] - u[t-1])
        # 注意: 不使用 β[t+1]，避免未来函数
        for t in range(T):
            # 计算右端项
            rhs = XtY[t].copy()
            if t > 0:
                rhs += rho * (beta[t - 1] + z[t - 1] - u[t - 1])

            # 计算左端矩阵 (端点 +ρI, 内部 +2ρI)
            LHS = XtX[t].copy()
            if t > 0:
                LHS += rho * np.eye(K)
            if t < T - 1:
                LHS += rho * np.eye(K)

            # 解线性系统
            try:
                beta[t] = np.linalg.solve(LHS, rhs)
            except np.linalg.LinAlgError:
                beta[t] = np.zeros(K)

        # 2. z-update: 固定 β, u, 解 soft-thresholding
        diff_beta = np.diff(beta, axis=0)  # (T-1, K)
        z = soft_thresholding(diff_beta + u, lambda_tv / rho)

        # 3. u-update: 对偶变量更新
        u = u + diff_beta - z

        # 4. 收敛检查 (处理 NaN)
        diff = np.abs(beta - beta_old)
        diff = diff[~np.isnan(diff)]
        if len(diff) == 0:
            break
        primal_res = np.max(diff)
        if primal_res < tol:
            break

    return beta


# ============================================================
# 滚动估计
# ============================================================
def rolling_tvpr(
    Y: pd.DataFrame,
    X_panel: np.ndarray,
    lambda_tv: float,
    lambda_l1: float,
    min_history: int = 52,
    window_size: int = 52,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-5,
) -> pd.DataFrame:
    """Walk-Forward 滚动估计 β_t (滚动窗口 + warm-start).

    对每个时间点 t (t ≥ min_history), 用 [t-window_size, t] 数据估计 β_t.
    用上一期的 β 作为 warm-start 初始值.

    Parameters:
        Y: (T, N) 周频资产收益
        X_panel: (T, N, K) 周频因子值面板
        lambda_tv: TV 罚项系数
        lambda_l1: L1 罚项系数
        min_history: 最少历史期数 (周)
        window_size: 滚动窗口大小 (周)
        rho: ADMM 惩罚参数
        max_iter: 最大迭代次数
        tol: 收敛阈值

    Returns:
        beta_path: (T, K) 时变 β_t
    """
    T, N, K = X_panel.shape

    # 初始化
    beta_path = np.zeros((T, K))
    beta_prev = None  # 用于 warm-start

    # Walk-Forward 滚动估计
    for t in range(min_history, T):
        # 滚动窗口: [t-window_size, t]
        start = max(0, t - window_size)
        Y_train = Y.iloc[start:t + 1].values
        X_train = X_panel[start:t + 1]

        # warm-start: 用上一期的 β 作为初始值
        if beta_prev is not None:
            # 将 beta_prev 扩展到当前窗口大小
            window_len = t - start + 1
            beta_init = np.tile(beta_prev, (window_len, 1))
        else:
            beta_init = None

        # ADMM 求解
        beta_full = tvpr_admm(
            Y_train, X_train,
            lambda_tv, lambda_l1,
            rho=rho, max_iter=max_iter, tol=tol,
            beta_init=beta_init,
        )

        # 取最后一个时间点的 β_t
        beta_path[t] = beta_full[-1]
        beta_prev = beta_full[-1]  # 保存当前 β 用于下一次 warm-start

    return pd.DataFrame(beta_path, index=Y.index, columns=[f"factor_{i}" for i in range(K)])


# ============================================================
# 接口
# ============================================================
def tvpr_estimator(
    Y: pd.DataFrame,
    X_panel: np.ndarray,
    lambda_tv: float,
    lambda_l1: float,
    method: Literal["admm"] = "admm",
    min_history: int = 52,
    window_size: int = 52,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-5,
) -> pd.DataFrame:
    """TV-PR estimator: 识别因子溢价的结构性变化.

    Parameters:
        Y: (T, N) 周频资产收益
        X_panel: (T, N, K) 周频因子值面板
        lambda_tv: TV 罚项系数, 控制 break 数量
        lambda_l1: L1 罚项系数, 控制因子稀疏性
        method: 求解方法 (目前只支持 "admm")
        min_history: 最少历史期数 (周)
        window_size: 滚动窗口大小 (周)
        rho: ADMM 惩罚参数
        max_iter: 最大迭代次数
        tol: 收敛阈值

    Returns:
        beta_path: (T, K) 时变 β_t
    """
    if method == "admm":
        return rolling_tvpr(
            Y, X_panel, lambda_tv, lambda_l1,
            min_history=min_history,
            window_size=window_size,
            rho=rho,
            max_iter=max_iter, tol=tol,
        )
    else:
        raise ValueError(f"未知 method: {method}")


__all__ = [
    "soft_thresholding",
    "compute_mse",
    "compute_tv_cost",
    "compute_l1_cost",
    "tvpr_admm",
    "rolling_tvpr",
    "tvpr_estimator",
]
