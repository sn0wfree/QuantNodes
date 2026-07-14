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
) -> np.ndarray:
    """ADMM 求解 TV-PR.

    增广形式:
      min Σ_t Σ_i (r_{i,t} - x_{i,t}' β_t)^2 
          + λ_1 ||z||_1 
          + λ_2 ||β||_1
          + (ρ/2) ||Δβ - z + u||_2^2

    Parameters:
        Y: (T, N) 资产收益
        X: (T, K) 因子值
        lambda_tv: TV 罚项系数
        lambda_l1: L1 罚项系数
        rho: ADMM 惩罚参数
        max_iter: 最大迭代次数
        tol: 收敛阈值

    Returns:
        beta: (T, K) 时变 β_t
    """
    T, N = Y.shape
    K = X.shape[1]

    # 初始化
    beta = np.zeros((T, K))
    z = np.zeros((T - 1, K))
    u = np.zeros((T - 1, K))

    # 预计算 X'X 和 X'Y (每个时间点)
    XtX = np.zeros((T, K, K))
    XtY = np.zeros((T, K))
    for t in range(T):
        XtX[t] = X[t].T @ X[t]  # (K, K)
        XtY[t] = X[t].T @ Y[t]  # (K,)

    # ADMM 迭代
    for iteration in range(max_iter):
        beta_old = beta.copy()

        # 1. β-update: 固定 z, u, 解 Lasso
        # 对每个时间点 t, 解:
        #   min (1/2) β_t' (X'X + ρI) β_t - β_t' (X'Y + ρ(Δ'z - Δ'u))
        #   + λ_2 ||β_t||_1
        # 
        # 简化: 用 coordinate descent 求解
        for t in range(T):
            # 计算右端项
            rhs = XtY[t].copy()
            if t > 0:
                rhs += rho * (z[t - 1] - u[t - 1])
            if t < T - 1:
                rhs += rho * (z[t] - u[t])

            # 计算左端矩阵
            LHS = XtX[t] + rho * np.eye(K)
            if t > 0:
                LHS += rho * np.eye(K)
            if t < T - 1:
                LHS += rho * np.eye(K)

            # 解线性系统 + soft-thresholding
            try:
                raw = np.linalg.solve(LHS, rhs)
            except np.linalg.LinAlgError:
                raw = np.zeros(K)

            # L1 惩罚
            beta[t] = soft_thresholding(raw, lambda_l1 / rho)

        # 2. z-update: 固定 β, u, 解 soft-thresholding
        diff_beta = np.diff(beta, axis=0)  # (T-1, K)
        z = soft_thresholding(diff_beta + u, lambda_tv / rho)

        # 3. u-update: 对偶变量更新
        u = u + diff_beta - z

        # 4. 收敛检查
        primal_res = np.max(np.abs(beta - beta_old))
        if primal_res < tol:
            break

    return beta


# ============================================================
# 滚动估计
# ============================================================
def rolling_tvpr(
    Y: pd.DataFrame,
    X: pd.DataFrame,
    lambda_tv: float,
    lambda_l1: float,
    min_history: int = 12,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-5,
) -> pd.DataFrame:
    """滚动估计 β_t.

    对每个时间点 t (t ≥ min_history), 用 [0, t] 数据估计 β_t.

    Parameters:
        Y: (T, N) 月频资产收益
        X: (T, K) 月频因子值
        lambda_tv: TV 罚项系数
        lambda_l1: L1 罚项系数
        min_history: 最少历史期数
        rho: ADMM 惩罚参数
        max_iter: 最大迭代次数
        tol: 收敛阈值

    Returns:
        beta_path: (T, K) 时变 β_t
    """
    T = len(Y)
    K = X.shape[1]

    # 初始化
    beta_path = np.zeros((T, K))

    # 滚动估计
    for t in range(min_history, T):
        # 训练集: [0, t]
        Y_train = Y.iloc[:t + 1].values
        X_train = X.iloc[:t + 1].values

        # ADMM 求解
        beta_full = tvpr_admm(
            Y_train, X_train,
            lambda_tv, lambda_l1,
            rho=rho, max_iter=max_iter, tol=tol,
        )

        # 取最后一个时间点的 β_t
        beta_path[t] = beta_full[-1]

    return pd.DataFrame(beta_path, index=Y.index, columns=X.columns)


# ============================================================
# 接口
# ============================================================
def tvpr_estimator(
    Y: pd.DataFrame,
    X: pd.DataFrame,
    lambda_tv: float,
    lambda_l1: float,
    method: Literal["admm"] = "admm",
    min_history: int = 12,
    max_iter: int = 200,
    tol: float = 1e-5,
) -> pd.DataFrame:
    """TV-PR estimator: 识别因子溢价的结构性变化.

    Parameters:
        Y: (T, N) 月频资产收益
        X: (T, K) 月频因子值
        lambda_tv: TV 罚项系数, 控制 break 数量
        lambda_l1: L1 罚项系数, 控制因子稀疏性
        method: 求解方法 (目前只支持 "admm")
        min_history: 最少历史期数
        max_iter: 最大迭代次数
        tol: 收敛阈值

    Returns:
        beta_path: (T, K) 时变 β_t
    """
    if method == "admm":
        return rolling_tvpr(
            Y, X, lambda_tv, lambda_l1,
            min_history=min_history,
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
