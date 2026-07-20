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
# ADMM 求解器 (标准 ADMM, 双辅助变量: TV + L1)
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
    l1_weights: np.ndarray | None = None,
) -> np.ndarray:
    """ADMM 求解 TV-PR (标准 ADMM, 双辅助变量).

    目标函数:
      min Σ_t Σ_i (r_{i,t} - x_{i,t}' β_t)^2
          + λ_tv ||Δβ||_1       (TV 罚项: 控制结构性断裂)
          + Σ_t Σ_k w_k(t) |β_{t,k}|  (L1 罚项: 控制因子稀疏性)

    ADMM 分裂 (两个辅助变量):
      min Σ_t Σ_i (r_{i,t} - x_{i,t}' β_t)^2 + λ_tv ||z1||_1 + Σ_t Σ_k w_k(t) |z2_{t,k}|
      s.t. Δβ = z1, β = z2

    β-update:
      (X'X + ρ*A^T*A + ρ*I) β = X'Y + ρ*A^T*(z1 - u1) + ρ*(z2 - u2)

    Parameters:
        Y: (T, N) 资产收益
        X: (T, N, K) 因子值面板
        lambda_tv: TV 罚项系数 (控制 β 时间变化幅度)
        lambda_l1: L1 罚项系数 (控制因子稀疏性)
        rho: ADMM 惩罚参数
        max_iter: 最大迭代次数
        tol: 收敛阈值
        beta_init: (T, K) 初始 β (warm-start), None 则用 zeros
        l1_weights: (T, K) 或 (K,) 或 None, 自适应 L1 权重
                    None 则使用均匀权重 lambda_l1
                    如果提供，z2-update 使用 w_k(t) 代替 lambda_l1

    Returns:
        beta: (T, K) 时变 β_t
    """
    T, N, K = X.shape

    # 初始化
    if beta_init is not None:
        beta = beta_init.copy()
    else:
        beta = np.zeros((T, K))
    # TV 罚项辅助变量 (Δβ = z1)
    z1 = np.zeros((T - 1, K))
    u1 = np.zeros((T - 1, K))
    # L1 罚项辅助变量 (β = z2)
    z2 = np.zeros((T, K))
    u2 = np.zeros((T, K))

    # 处理 l1_weights: 统一为 (T, K) 形状
    if l1_weights is None:
        # 均匀权重
        l1_weights_full = np.full((T, K), lambda_l1)
    elif l1_weights.ndim == 1:
        # (K,) → 广播到 (T, K)
        l1_weights_full = np.tile(l1_weights, (T, 1))
    elif l1_weights.ndim == 2:
        l1_weights_full = l1_weights
    else:
        raise ValueError(f"l1_weights must be 1D or 2D, got {l1_weights.ndim}D")

    # 预计算 X'X 和 X'Y (每个时间点, 处理 NaN)
    # X[t] is (N, K), Y[t] is (N,)
    # X'X = X[t].T @ X[t] is (K, K)
    # X'Y = X[t].T @ Y[t] is (K,)
    XtX = np.zeros((T, K, K))
    XtY = np.zeros((T, K))
    for t in range(T):
        # 过滤 NaN
        valid_mask = ~np.isnan(Y[t]) & ~np.any(np.isnan(X[t]), axis=1)
        if np.sum(valid_mask) < 10:
            # 样本不足 (< 10 个有效资产), 跳过
            # 注意: K > N 时 (如 v7.13 K=46 > N=43), L1 正则化可以处理欠定情况
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
        z1_old = z1.copy()
        z2_old = z2.copy()

        # 1. β-update: 固定 z1, u1, z2, u2, 解线性系统
        # (X'X + ρ*A^T*A + ρ*I) β = X'Y + ρ*A^T*(z1 - u1) + ρ*(z2 - u2)

        # 计算 A^T*(z1 - u1) (差分算子的转置)
        AT_z1u1 = np.zeros((T, K))
        AT_z1u1[0] = -(z1[0] - u1[0])
        for t in range(1, T - 1):
            AT_z1u1[t] = (z1[t - 1] - u1[t - 1]) - (z1[t] - u1[t])
        AT_z1u1[T - 1] = z1[T - 2] - u1[T - 2]

        # 对每个因子 k 独立求解
        for k in range(K):
            # 构建 RHS
            rhs = np.zeros(T)
            for t in range(T):
                # RHS: X'Y + ρ*A^T*(z1-u1) + ρ*(z2-u2)
                rhs[t] = XtY[t, k] + rho * AT_z1u1[t, k] + rho * (z2[t, k] - u2[t, k])

            # 构建 LHS 对角线
            # 对角线: XtX[t,k,k] + ρ*ATA_diag[t] + ρ (多了 +ρ 来自 L1 罚项)
            diag = np.zeros(T)
            for t in range(T):
                diag[t] = XtX[t, k, k] + rho * ATA_diag[t] + rho

            # 解三对角系统
            beta_k = _solve_tridiag(ATA_offdiag * rho, diag, ATA_offdiag * rho, rhs)
            beta[:, k] = beta_k

        # 2. z1-update (TV penalty): z1 = S_{λ_tv/ρ}(Δβ + u1)
        diff_beta = np.diff(beta, axis=0)  # (T-1, K)
        z1 = soft_thresholding(diff_beta + u1, lambda_tv / rho)

        # 3. z2-update (L1 penalty): z2 = S_{w_k(t)/ρ}(β + u2)
        #    使用自适应权重 l1_weights_full[t, k] 代替均匀 lambda_l1
        z2 = soft_thresholding(beta + u2, l1_weights_full / rho)

        # 4. u-update
        u1 = u1 + diff_beta - z1
        u2 = u2 + beta - z2

        # 5. 收敛检查 (原始残差 + 对偶残差)
        # 原始残差: r = [Δβ - z1, β - z2]
        primal_res = np.sqrt(
            np.linalg.norm(diff_beta - z1) ** 2 +
            np.linalg.norm(beta - z2) ** 2
        )
        # 对偶残差: s = [ρ*(z1 - z1_old), ρ*(z2 - z2_old)]
        dual_res = rho * np.sqrt(
            np.linalg.norm(z1 - z1_old) ** 2 +
            np.linalg.norm(z2 - z2_old) ** 2
        )

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
    l1_weights: np.ndarray | None = None,
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
        l1_weights: (T, K) 或 (K,) 或 None, 自适应 L1 权重

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
        l1_weights=l1_weights,
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
    l1_weights: np.ndarray | None = None,
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
        l1_weights: (T, K) 或 (K,) 或 None, 自适应 L1 权重
                    None 则使用均匀权重 lambda_l1

    Returns:
        beta_path: (T, K) 时变 β_t
    """
    if method == "admm":
        return full_sample_tvpr(
            Y, X_panel, lambda_tv, lambda_l1,
            min_history=min_history,
            rho=rho,
            max_iter=max_iter, tol=tol,
            l1_weights=l1_weights,
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
    "expanding_window_tvpr",
    "rolling_window_tvpr",
    "tvpr_estimator",
]


# ============================================================
# OOS 因果估计: expanding-window / rolling-window
# ============================================================
def expanding_window_tvpr(
    Y: pd.DataFrame,
    X_panel: np.ndarray,
    lambda_tv: float,
    lambda_l1: float,
    min_history: int = 52,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-5,
    step: int = 1,
) -> pd.DataFrame:
    """递增窗口 OOS 估计 β_t (无前视偏差).

    对每个预测时点 t = min_history, ..., T-1:
      1. 训练集: Y[:t], X_panel[:t]
      2. 用上一次 beta 做 warm-start
      3. 运行 ADMM → 取 beta_path[-1] 作为 beta_oos[t]

    step > 1 时每 step 周更新一次 beta, 中间时点用最近的值 forward-fill.

    Parameters:
        Y: (T, N) 周频资产收益
        X_panel: (T, N, K) 周频因子值面板
        lambda_tv, lambda_l1: 罚项系数
        min_history: 最少历史期数
        rho, max_iter, tol: ADMM 参数
        step: 更新频率 (1=每周, 4=每4周)

    Returns:
        beta_oos: (T, K) OOS 估计的 β_t, 前 min_history 行为 0
    """
    T, N, K = X_panel.shape
    beta_oos = np.zeros((T, K))
    beta_warm = None
    last_beta = np.zeros(K)

    for t in range(min_history, T, step):
        Y_train = Y.iloc[:t]
        X_train = X_panel[:t]

        if beta_warm is not None:
            beta_init = np.zeros((t, K))
            beta_init[:beta_warm.shape[0]] = beta_warm
        else:
            beta_init = None

        beta_path = tvpr_admm(
            Y_train.values, X_train,
            lambda_tv, lambda_l1,
            rho=rho, max_iter=max_iter, tol=tol,
            beta_init=beta_init,
        )
        last_beta = beta_path[-1]
        beta_warm = beta_path

        # Fill this step's beta into all timepoints from previous step to here
        for s in range(max(t - step, min_history), t + 1):
            if s < T:
                beta_oos[s] = last_beta

    return pd.DataFrame(beta_oos, index=Y.index,
                        columns=[f"factor_{i}" for i in range(K)])


def rolling_window_tvpr(
    Y: pd.DataFrame,
    X_panel: np.ndarray,
    lambda_tv: float,
    lambda_l1: float,
    window: int = 104,
    min_history: int = 52,
    rho: float = 1.0,
    max_iter: int = 200,
    tol: float = 1e-5,
    step: int = 1,
) -> pd.DataFrame:
    """滑动窗口 OOS 估计 β_t (无前视偏差).

    对每个预测时点 t = min_history, ..., T-1:
      1. W_eff = min(window, t)
      2. 训练集: Y[t-W_eff:t], X_panel[t-W_eff:t]
      3. warm-start: 上一次 beta 的最后 W_eff 行
      4. 运行 ADMM → 取 beta_path[-1]

    step > 1 时每 step 周更新一次 beta, 中间时点 forward-fill.

    Parameters:
        Y: (T, N) 周频资产收益
        X_panel: (T, N, K) 因子值面板
        lambda_tv, lambda_l1: 罚项系数
        window: 滚动窗口大小 (周)
        min_history: 最少历史期数
        rho, max_iter, tol: ADMM 参数
        step: 更新频率 (1=每周, 4=每4周)

    Returns:
        beta_oos: (T, K) OOS 估计的 β_t, 前 min_history 行为 0
    """
    T, N, K = X_panel.shape
    beta_oos = np.zeros((T, K))
    beta_warm = None
    last_beta = np.zeros(K)

    for t in range(min_history, T, step):
        W_eff = min(window, t)
        start = t - W_eff
        Y_train = Y.iloc[start:t]
        X_train = X_panel[start:t]

        if beta_warm is not None and beta_warm.shape[0] >= W_eff:
            beta_init = beta_warm[-W_eff:]
        else:
            beta_init = None

        beta_path = tvpr_admm(
            Y_train.values, X_train,
            lambda_tv, lambda_l1,
            rho=rho, max_iter=max_iter, tol=tol,
            beta_init=beta_init,
        )
        last_beta = beta_path[-1]
        beta_warm = beta_path

        for s in range(max(t - step, min_history), t + 1):
            if s < T:
                beta_oos[s] = last_beta

    return pd.DataFrame(beta_oos, index=Y.index,
                        columns=[f"factor_{i}" for i in range(K)])
