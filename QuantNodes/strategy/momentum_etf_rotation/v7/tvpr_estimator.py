# coding=utf-8
"""TV-PR Estimator: Total-Variation Predictive Regression.

Cui et al. (2025) "Breaks and trends in factor premia."

目标函数:
  min Σ_t Σ_i (r_{i,t} - x_{i,t}' β_t)^2
      + λ_1 Σ_t ||β_t - β_{t-1}||_1
      + λ_2 Σ_t ||β_t||_1

求解方法: ADMM (Alternating Direction Method of Multipliers)
  - 标准 ADMM (直接解线性系统, 非前向-后向扫描)
  - 全量样本估计 (一次 ADMM 求解所有时点)
  - 标准 ADMM 收敛准则 (原始残差 + 对偶残差)
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
# ADMM 求解器 (标准 ADMM, 直接解线性系统)
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
    """ADMM 求解 TV-PR (标准 ADMM, 直接解线性系统).

    增广形式:
      min Σ_t Σ_i (r_{i,t} - x_{i,t}' β_t)^2
          + λ_1 ||z||_1
          + λ_2 ||β||_1
          + (ρ/2) ||Δβ - z + u||_2^2

    β-update 公式 (标准 ADMM):
      (X'X + ρ*A^T*A) β = X'Y + ρ*A^T*(z - u)
      其中 A 是差分算子, A^T*A 是三对角矩阵

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

    # 构建 A^T*A 矩阵 (三对角)
    # A 是差分算子: Δβ[t] = β[t+1] - β[t]
    # A^T*A 的对角线: [1, 2, 2, ..., 2, 1]
    # A^T*A 的次对角线: [-1, -1, ...]
    ATA_diag = np.ones(T)
    ATA_diag[1:-1] = 2.0
    ATA_offdiag = -np.ones(T - 1)

    # ADMM 迭代
    for iteration in range(max_iter):
        beta_old = beta.copy()
        z_old = z.copy()

        # 1. β-update: 固定 z, u, 解线性系统
        # (X'X + ρ*A^T*A) β = X'Y + ρ*A^T*(z - u)
        # 对每个因子 k 独立求解 (因为 X'X 是对角的)

        # 计算 A^T*(z - u)
        # A^T 是差分算子的转置
        # A^T*(z-u)[t] = (z[t-1] - u[t-1]) - (z[t] - u[t])  (内部点)
        # A^T*(z-u)[0] = -(z[0] - u[0])  (起点)
        # A^T*(z-u)[T-1] = (z[T-2] - u[T-2])  (终点)
        AT_zu = np.zeros((T, K))
        AT_zu[0] = -(z[0] - u[0])
        for t in range(1, T - 1):
            AT_zu[t] = (z[t - 1] - u[t - 1]) - (z[t] - u[t])
        AT_zu[T - 1] = z[T - 2] - u[T - 2]

        # 对每个因子 k 独立求解
        for k in range(K):
            # 构建 (K, K) 的块状矩阵
            # 对角块: XtX[t] + ρ*A^T*A[t,t]*I
            # 非对角块: ρ*A^T*A[t,s]*I (只有相邻时点)

            # 构建 RHS
            rhs = np.zeros(T)
            for t in range(T):
                rhs[t] = XtY[t, k] + rho * AT_zu[t, k]

            # 构建 LHS 矩阵 (三对角)
            # 对角线: XtX[t, k, k] + ρ*ATA_diag[t]
            # 次对角线: ρ*ATA_offdiag[t]
            diag = np.zeros(T)
            for t in range(T):
                diag[t] = XtX[t, k, k] + rho * ATA_diag[t]

            # 解三对角系统
            # 使用 Thomas 算法 (追赶法)
            beta_k = _solve_tridiag(ATA_offdiag * rho, diag, ATA_offdiag * rho, rhs)
            for t in range(T):
                beta[t, k] = beta_k[t]

        # 2. z-update: 固定 β, u, 解 soft-thresholding
        diff_beta = np.diff(beta, axis=0)  # (T-1, K)
        z = soft_thresholding(diff_beta + u, lambda_tv / rho)

        # 3. u-update: 对偶变量更新
        u = u + diff_beta - z

        # 4. 收敛检查 (原始残差 + 对偶残差)
        # 原始残差: r = Δβ - z
        primal_res = np.linalg.norm(diff_beta - z)
        # 对偶残差: s = ρ * (z - z_old) (简化形式)
        dual_res = rho * np.linalg.norm(z - z_old)

        # 收敛条件: 原始和对偶残差都小于阈值
        if primal_res < tol and dual_res < tol:
            break

    return beta


def _solve_tridiag(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Thomas 算法 (追赶法) 解三对角系统.

    Parameters:
        lower: (T-1,) 下对角线
        diag: (T,) 主对角线
        upper: (T-1,) 上对角线
        rhs: (T,) 右端项

    Returns:
        x: (T,) 解
    """
    T = len(diag)
    if T == 0:
        return np.array([])

    # 前向消元
    c = np.zeros(T - 1)
    d = np.zeros(T)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, T):
        if i < T - 1:
            denom = diag[i] - lower[i - 1] * c[i - 1]
            c[i] = upper[i] / denom
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / (diag[i] - lower[i - 1] * c[i - 1]) if i > 0 else rhs[i] / diag[i]

    # 回代
    x = np.zeros(T)
    x[T - 1] = d[T - 1]
    for i in range(T - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]

    return x


# ============================================================
# 全量估计 (替代滚动窗口)
# ============================================================
def full_sample_tvpr(
    Y: pd.DataFrame,
    X_panel: np.ndarray,
    lambda_tv: float,
    lambda_l1: float,
    min_history: int = 52,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-5,
) -> pd.DataFrame:
    """全量样本估计 β_t (一次 ADMM 求解所有时点).

    直接在全量数据 [0, T] 上运行 ADMM, 提取所有 β_t.
    这是论文的标准方法, 计算效率最高.

    Parameters:
        Y: (T, N) 周频资产收益
        X_panel: (T, N, K) 周频因子值面板
        lambda_tv: TV 罚项系数
        lambda_l1: L1 罚项系数
        min_history: 最少历史期数 (周, 用于设置前 min_history 个 β 为零)
        rho: ADMM 惩罚参数
        max_iter: 最大迭代次数
        tol: 收敛阈值

    Returns:
        beta_path: (T, K) 时变 β_t
    """
    T, N, K = X_panel.shape

    # 在全量数据上运行 ADMM
    beta_full = tvpr_admm(
        Y.values, X_panel,
        lambda_tv, lambda_l1,
        rho=rho, max_iter=max_iter, tol=tol,
        beta_init=None,
    )

    # 前 min_history 个时点的 β 设为零 (数据不足)
    beta_full[:min_history] = 0.0

    return pd.DataFrame(beta_full, index=Y.index, columns=[f"factor_{i}" for i in range(K)])


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
    window_size: int = 52,  # 保留参数兼容性, 但不再使用
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
        window_size: 滚动窗口大小 (不再使用, 保留参数兼容性)
        rho: ADMM 惩罚参数
        max_iter: 最大迭代次数
        tol: 收敛阈值

    Returns:
        beta_path: (T, K) 时变 β_t
    """
    if method == "admm":
        return full_sample_tvpr(
            Y, X_panel, lambda_tv, lambda_l1,
            min_history=min_history,
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
    "full_sample_tvpr",
    "tvpr_estimator",
]
