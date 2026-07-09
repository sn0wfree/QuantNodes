# coding=utf-8
"""Nagel 风格的因子择时: Ledoit-Wolf 协方差 + λ 权重收缩.

参考论文: 《Optimal Factor Timing in a High-Dimensional Setting》 (Nagel 团队)
A 股复现: QuantML 知识星球《论文复现 | 最优因子择时框架》

核心算法:
1. **Ledoit-Wolf 协方差收缩**:
   cov_lw = (1-δ) · S + δ · F
   其中 S = 样本协方差, F = σ̄² · I (对角目标)
   δ = 最优收缩强度 (analytic 公式)

2. **MVO 权重**:
   w_unnorm = cov_lw⁻¹ · μ   (μ = 平均 IC 向量)
   w_mvo = sign-aligned (long-only via max(0, ...))

3. **λ 权重收缩** (核心):
   w_final = (1-λ) · w_mvo + λ · w_equal
   λ 大 → 接近等权 (静态)
   λ 小 → 信任 MVO (择时)
   论文 OOS 显示 λ=30-100 是稳健选择

4. **总敞口约束**:
   Σ|w_i| = 1 (L1 范数归一化, 论文用绝对值)

5. **滚动验证 λ**:
   训练窗 (扩展窗口) → 验证窗 (12m) → 选最大 Sharpe 的 λ
   然后应用到下一个 OOS 期
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def ledoit_wolf_shrinkage(
    X: np.ndarray,
    block_size: int = 1,
) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf 协方差收缩估计 (Ledoit & Wolf 2004 公式).

    Args:
        X: (T, N) 收益/IC 矩阵 (T 样本, N 因子)
        block_size: 块大小 (1 = iid, >1 = 块 bootstrap)

    Returns:
        cov_lw: 收缩后协方差矩阵 (N, N)
        delta: 最优收缩强度
    """
    T, N = X.shape
    if T < 2:
        return np.eye(N), 1.0

    X = X - X.mean(axis=0, keepdims=True)
    S = (X.T @ X) / T

    var = np.diag(S).mean()
    F = var * np.eye(N)

    X2 = X ** 2
    var_S = ((X2.T @ X2) / T - S ** 2).sum() / N
    if var_S < 1e-12:
        return F.copy(), 1.0

    if block_size == 1:
        X_centered = X - X.mean(axis=0, keepdims=True)
        M = (X_centered.T @ X_centered) / T
        M2 = M ** 2
        diff = S - F
        numerator = (diff ** 2).sum() / N
        pi_mat = ((X2.T @ X2) / T - M2)
        pi_mat = pi_mat.sum() / N
        gamma = (pi_mat - numerator) / T
    else:
        numerator = ((S - F) ** 2).sum() / N
        n_blocks = T // block_size
        gamma = 0.0
        for b in range(n_blocks):
            idx = slice(b * block_size, (b + 1) * block_size)
            Xb = X[idx] - X[idx].mean(axis=0, keepdims=True)
            Mb = (Xb.T @ Xb) / block_size
            diff = Mb - S
            gamma += (diff ** 2).sum() / N
        gamma /= n_blocks

    if gamma < 1e-12:
        delta = 1.0
    else:
        delta = max(0.0, min(1.0, numerator / gamma))

    cov_lw = (1 - delta) * S + delta * F
    return cov_lw, float(delta)


def mvo_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    long_only: bool = True,
    l1_norm: float = 1.0,
) -> np.ndarray:
    """最小方差组合 (MVO) 给定 mean (μ) 和 cov.

    解析解: w ∝ cov⁻¹ · μ
    然后:
        - long-only: max(0, w)
        - L1 归一化: Σ|w| = l1_norm

    Args:
        mu: (N,) 平均 IC 向量
        cov: (N, N) 协方差矩阵
        long_only: 是否仅多头 (论文: 是)
        l1_norm: L1 范数 (论文: 1.0)

    Returns:
        w: (N,) 权重
    """
    N = len(mu)
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov)

    w = cov_inv @ mu

    if long_only:
        w = np.maximum(w, 0.0)

    l1 = np.abs(w).sum()
    if l1 > 1e-12:
        w = w * (l1_norm / l1)
    return w


def compute_lambda_weights(
    w_mvo: np.ndarray,
    w_equal: np.ndarray,
    lam: float,
) -> np.ndarray:
    """λ 收缩: w_final = (1-λ)·w_mvo + λ·w_equal.

    Args:
        w_mvo: (N,) MVO 权重
        w_equal: (N,) 等权 (1/N)
        lam: 收缩强度 [0, 1] 或 [0, ∞)
            - 论文用 log scale, 例如 λ ∈ [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 30, 100]
            - 公式: w_final = (1-λ/(1+λ))·w_mvo + (λ/(1+λ))·w_equal

    Returns:
        w_final: (N,) 收缩后权重
    """
    shrink = lam / (1.0 + lam)
    w_final = (1.0 - shrink) * w_mvo + shrink * w_equal
    return w_final


def select_lambda_rolling(
    ic_history: pd.DataFrame,
    train_window: int,
    val_window: int,
    candidate_lambdas: list[float],
    factor_to_etf: dict[str, tuple[str, ...]],
    nav_panel: pd.DataFrame,
    target_vol: float = 0.05,
) -> float:
    """滚动选择最优 λ: 在验证期最大化 Sharpe.

    Args:
        ic_history: 滚动 IC DataFrame (index=date, columns=factor)
        train_window: 训练期长度 (月, e.g. 60)
        val_window: 验证期长度 (月, e.g. 12)
        candidate_lambdas: 候选 λ 列表 (log scale)
        factor_to_etf: 因子→ETF 映射
        nav_panel: 价格面板 (用于回测验证期)
        target_vol: 目标年化波动 (用于 vol scaling, 论文用)

    Returns:
        best_lambda: 最优 λ
    """
    if len(ic_history) < train_window + val_window:
        return candidate_lambdas[len(candidate_lambdas) // 2]

    train_ic = ic_history.iloc[-train_window - val_window:-val_window]
    val_ic = ic_history.iloc[-val_window:]

    if len(train_ic) < 12 or len(val_ic) < 3:
        return candidate_lambdas[len(candidate_lambdas) // 2]

    equal_w = np.ones(len(train_ic.columns)) / len(train_ic.columns)

    best_lambda = candidate_lambdas[0]
    best_sharpe = -np.inf

    for lam in candidate_lambdas:
        nav_val = simulate_lambda_path(
            train_ic, val_ic, lam, equal_w, factor_to_etf, nav_panel,
        )
        if nav_val is None or len(nav_val) < 3:
            continue
        rets = nav_val.pct_change().dropna()
        if rets.std() == 0:
            continue
        sharpe = rets.mean() / rets.std() * np.sqrt(12)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_lambda = lam

    return best_lambda


def simulate_lambda_path(
    train_ic: pd.DataFrame,
    val_ic: pd.DataFrame,
    lam: float,
    equal_w: np.ndarray,
    factor_to_etf: dict[str, tuple[str, ...]],
    nav_panel: pd.DataFrame,
) -> Optional[pd.Series]:
    """在验证期用给定 λ 模拟 NAV.

    用训练期计算 MVO 权重 (固定), 然后在验证期用相同权重 + 等权混合,
    按因子→ETF 映射持仓.
    """
    factors = list(train_ic.columns)
    X = train_ic[factors].fillna(0.0).values
    if X.std() < 1e-6:
        return None

    mu = X.mean(axis=0)
    cov, _ = ledoit_wolf_shrinkage(X)
    w_mvo = mvo_weights(mu, cov, long_only=True, l1_norm=1.0)
    w_final = compute_lambda_weights(w_mvo, equal_w, lam)

    factor_to_etf_w: dict[str, float] = {}
    for fac, w in zip(factors, w_final):
        if w <= 0:
            continue
        codes = factor_to_etf.get(fac, ())
        if not codes:
            continue
        per_etf = w / len(codes)
        for c in codes:
            factor_to_etf_w[c] = factor_to_etf_w.get(c, 0.0) + per_etf

    val_dates = val_ic.index
    panel_dates = nav_panel.index
    val_panel_dates = [d for d in val_dates if d in panel_dates]
    if len(val_panel_dates) < 2:
        return None

    pos = panel_dates.get_indexer(val_panel_dates)
    pos = pos[pos >= 0]
    if len(pos) < 2:
        return None

    nav = np.ones(len(pos))
    for k in range(1, len(pos)):
        i = pos[k]
        i_prev = pos[k - 1]
        daily_ret = 0.0
        for code, w in factor_to_etf_w.items():
            if code in nav_panel.columns:
                r = nav_panel[code].iloc[i] / nav_panel[code].iloc[i_prev] - 1.0
                if not np.isnan(r):
                    daily_ret += w * r
        nav[k] = nav[k - 1] * (1 + daily_ret)

    return pd.Series(nav, index=nav_panel.index[pos])


__all__ = [
    "ledoit_wolf_shrinkage",
    "mvo_weights",
    "compute_lambda_weights",
    "select_lambda_rolling",
    "simulate_lambda_path",
]
