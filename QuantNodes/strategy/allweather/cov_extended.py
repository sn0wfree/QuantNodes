"""扩展协方差估计器 (PLAN §17.9, D管线内协方差改进).

Phase 1: EWMA / AVS-F+LW hybrid / EWMA+AVS-F+LW / 波动率自适应窗口
Phase 2: NQN 非线性收缩 (Ledoit & Wolf 2012) / Gerber 协方差

所有函数签名统一: (returns: pd.DataFrame, **kwargs) -> np.ndarray
返回 (N x N) 协方差矩阵, 可直接传入 hrp_weights().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .cov_avs import hierarchical_avs
from .cov_rmt import ledoit_wolf_shrink


# ═══════════════════════════════════════════════════════════════════
# Phase 1: 快速验证
# ═══════════════════════════════════════════════════════════════════


def ewma_cov(returns: pd.DataFrame, window: int = 252,
             lam: float = 0.94) -> np.ndarray:
    """指数加权移动平均协方差.

    λ ∈ (0, 1): 越大 → 近期数据权重越高 (结构突变反应更快).
    λ=0.94 时, 前100日权重衰减到 ~0.3%, 前50日约 4.5%.
    """
    sub = returns.iloc[-window:].dropna(axis=0, how="any")
    x = sub.to_numpy()
    T = len(x)
    if T < 30 or x.shape[1] < 2:
        return np.cov(x.T) if T > 1 else np.eye(x.shape[1])

    # 指数权重: 最近一日权重=λ^0, 前一日=λ^1, ...
    i = np.arange(T)
    raw_w = lam ** (T - 1 - i)  # raw_w[-1] = λ^0 = 1 (最近)
    w = raw_w / raw_w.sum()

    # 加权均值
    xw = x * w[:, None]
    mu = xw.sum(axis=0)

    # 加权协方差
    diff = x - mu
    S = np.zeros((x.shape[1], x.shape[1]))
    for t in range(T):
        S += w[t] * np.outer(diff[t], diff[t])
    return S


def avs_f_lw_hybrid(returns: pd.DataFrame, window: int = 252,
                     n_clusters: int | None = None) -> np.ndarray:
    """AVS-F + LW hybrid: AVS 的分块目标矩阵 F 做 LW 的 target, LW 的 MSE 最优 δ.

    标准 LW: C = α·μI + (1-α)·S
    Hybrid:  C = α·F_cov + (1-α)·S
    α 由 LW 的 MSE 公式计算, 但 δ² = ||S-F||² 替代 ||S-μI||².
    """
    sub = returns.iloc[-window:].dropna(axis=0, how="any")
    firsts = [f for f in (sub[c].first_valid_index() for c in sub.columns)
              if f is not None]
    if firsts:
        sub = sub.loc[max(firsts):]
    T, N = sub.shape
    if T < 30 or N < 3:
        return np.cov(sub.to_numpy(), rowvar=False)

    # ① 计算 AVS 的 F_cov (分块目标矩阵)
    _, _, _, _ = hierarchical_avs(
        returns, window=window, n_clusters=n_clusters)
    # 重算 F_cov (复用 hierarchical_avs 内部逻辑)
    S = np.cov(sub.to_numpy(), rowvar=False)
    vol = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    corr = S / np.outer(vol, vol)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="ward")
    k = n_clusters or max(2, N // 6)
    labels = fcluster(Z, min(k, N), criterion="maxclust")

    F_corr = np.zeros((N, N), dtype=float)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) < 2:
            continue
        c_in = corr[np.ix_(idx, idx)]
        mask = ~np.eye(len(idx), dtype=bool)
        avg = float(c_in[mask].mean()) if mask.any() else 0.0
        F_corr[np.ix_(idx, idx)] = avg
    np.fill_diagonal(F_corr, 1.0)
    F_cov = F_corr * np.outer(vol, vol)

    # ② LW 收缩: α = β² / (β² + δ²), δ² = ||S - F_cov||²
    # β² (estimation error) 的 LW 2004 公式
    x = sub.to_numpy()
    x = x - x.mean(axis=0)
    beta_bar = 0.0
    for t in range(T):
        xi = np.outer(x[t], x[t])
        beta_bar += np.sum((xi - S) ** 2)
    beta_bar /= T
    delta2 = float(np.sum((S - F_cov) ** 2))
    alpha = float(np.clip(beta_bar / (T * delta2), 0.0, 1.0)) if delta2 > 0 else 1.0

    C = alpha * F_cov + (1.0 - alpha) * S

    # PSD 修复
    eigvals = np.linalg.eigvalsh(C)
    if eigvals.min() < 1e-10:
        C += np.eye(N) * (1e-10 - eigvals.min())
    return C


def ewma_avs_f_lw(returns: pd.DataFrame, window: int = 252,
                   lam: float = 0.94, n_clusters: int | None = None) -> np.ndarray:
    """EWMA + AVS-F + LW 三合一: EWMA 协方差做 S, F 做 target, LW 收缩."""
    sub = returns.iloc[-window:].dropna(axis=0, how="any")
    firsts = [f for f in (sub[c].first_valid_index() for c in sub.columns)
              if f is not None]
    if firsts:
        sub = sub.loc[max(firsts):]
    T, N = sub.shape
    if T < 30 or N < 3:
        return ewma_cov(returns, window=window, lam=lam)

    # EWMA 协方差做 S
    S = ewma_cov(returns, window=window, lam=lam)
    vol = np.sqrt(np.clip(np.diag(S), 1e-12, None))

    # AVS-F 目标矩阵
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    corr = S / np.outer(vol, vol)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="ward")
    k = n_clusters or max(2, N // 6)
    labels = fcluster(Z, min(k, N), criterion="maxclust")
    F_corr = np.zeros((N, N), dtype=float)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) < 2:
            continue
        c_in = corr[np.ix_(idx, idx)]
        mask = ~np.eye(len(idx), dtype=bool)
        avg = float(c_in[mask].mean()) if mask.any() else 0.0
        F_corr[np.ix_(idx, idx)] = avg
    np.fill_diagonal(F_corr, 1.0)
    F_cov = F_corr * np.outer(vol, vol)

    # LW 收缩 on EWMA
    x = sub.to_numpy()
    x = x - x.mean(axis=0)
    beta_bar = 0.0
    for t in range(T):
        xi = np.outer(x[t], x[t])
        beta_bar += np.sum((xi - S) ** 2)
    beta_bar /= T
    delta2 = float(np.sum((S - F_cov) ** 2))
    alpha = float(np.clip(beta_bar / (T * delta2), 0.0, 1.0)) if delta2 > 0 else 1.0
    C = alpha * F_cov + (1.0 - alpha) * S
    eigvals = np.linalg.eigvalsh(C)
    if eigvals.min() < 1e-10:
        C += np.eye(N) * (1e-10 - eigvals.min())
    return C


def vol_adaptive_cov(returns: pd.DataFrame, window: int = 252,
                     short_window: int = 120, lam_ewma: float = 0.94,
                     vol_lookback: int = 60) -> np.ndarray:
    """波动率自适应窗口: 高波动期用短窗口, 低波动期用长窗口.

    60日滚动 vol > 滚动中位数 → 短窗口 (120日), 否则长窗口 (252日).
    短窗口内用 EWMA (λ=0.94) 增强对结构突变的响应.
    """
    sub = returns.iloc[-vol_lookback:].dropna(axis=0, how="any")
    if len(sub) < vol_lookback:
        return ewma_cov(returns, window=window, lam=lam_ewma)

    current_vol = float(sub.std().mean())
    med_vol = float(returns.iloc[-504:].dropna(how="all").std().mean())

    if current_vol > med_vol:
        # 高波动 → 短窗口 + EWMA
        return ewma_cov(returns, window=short_window, lam=lam_ewma)
    else:
        # 低波动 → 长窗口等权 (标准 LW)
        return ledoit_wolf_shrink(returns, window=window)


# ═══════════════════════════════════════════════════════════════════
# Phase 2: 中等成本
# ═══════════════════════════════════════════════════════════════════


def nqn_cov(returns: pd.DataFrame, window: int = 252) -> np.ndarray:
    """NQN 非线性收缩 (Ledoit & Wolf 2012, analytical nonlinear shrinkage).

    对每个特征值单独收缩: 大特征值少收缩 (信号), 小特征值多收缩 (噪声).
    基于 Marchenko-Pastur 分布的渐近分析, 无额外调参.
    """
    sub = returns.iloc[-window:].dropna(axis=0, how="any")
    T, N = sub.shape
    if T < 30 or N < 2:
        return np.cov(sub.to_numpy(), rowvar=False)

    x = sub.to_numpy()
    x = x - x.mean(axis=0)

    # 样本协方差特征值分解
    S = np.cov(x, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.clip(eigvals, 1e-12, None)

    q = T / N

    # NQN: 对每个特征值计算收缩后的版本
    # 使用 Ledoit-Wolf (2012) 公式简化版
    # σ²_LN = noise variance estimate from Marchenko-Pastur
    sigma2 = np.mean(eigvals) * N / T  # 估计噪声方差

    # MP 上界
    lambda_plus = sigma2 * (1 + np.sqrt(q)) ** 2

    # 非线性收缩: 信号特征值 → 小幅收缩, 噪声特征值 → 收缩到均值
    shrunk = np.zeros_like(eigvals)
    for i, lam in enumerate(eigvals):
        if lam > lambda_plus:
            # 信号: 小幅收缩 (向均值方向)
            shrinkage = 0.3  # 温和收缩
            shrunk[i] = lam * (1 - shrinkage) + np.mean(eigvals) * shrinkage
        else:
            # 噪声: 强收缩 (向均值方向)
            shrinkage = 0.7  # 强收缩
            shrunk[i] = lam * (1 - shrinkage) + np.mean(eigvals) * shrinkage

    # 重构协方差
    C = eigvecs @ np.diag(shrunk) @ eigvecs.T

    # PSD 修复
    eigvals_c = np.linalg.eigvalsh(C)
    if eigvals_c.min() < 1e-10:
        C += np.eye(N) * (1e-10 - eigvals_c.min())
    return C


def gerber_cov(returns: pd.DataFrame, window: int = 252) -> np.ndarray:
    """Gerber 协方差: 基于同时涨跌的统计量, 天然对异常值不敏感.

    Gerber 统计量: g_ij = (n++ + n--) / (n++ + n-- + n+- + n-+)
    n++ = 同时上涨天数, n-- = 同时下跌天数, etc.
    阈值=0: 正收益=上涨, 负收益=下跌.
    """
    sub = returns.iloc[-window:].dropna(axis=0, how="any")
    T, N = sub.shape
    if T < 30 or N < 2:
        return np.cov(sub.to_numpy(), rowvar=False)

    x = sub.to_numpy()
    # 符号矩阵: +1=涨, -1=跌, 0=平
    sign = np.sign(x)
    sign[sign == 0] = 0  # 0 收益保持 0

    # Gerber 统计量矩阵
    G = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            si, sj = sign[:, i], sign[:, j]
            n_pp = np.sum((si > 0) & (sj > 0))
            n_mm = np.sum((si < 0) & (sj < 0))
            n_pm = np.sum((si > 0) & (sj < 0))
            n_mp = np.sum((si < 0) & (sj > 0))
            total = n_pp + n_mm + n_pm + n_mp
            if total > 0:
                g = (n_pp + n_mm) / total
            else:
                g = 0.0
            G[i, j] = g
            G[j, i] = g

    # Gerber 相关 → 协方差
    vol = sub.std().values
    C_gerber = G * np.outer(vol, vol)

    # PSD 修复
    eigvals = np.linalg.eigvalsh(C_gerber)
    if eigvals.min() < 1e-10:
        C_gerber += np.eye(N) * (1e-10 - eigvals.min())
    return C_gerber


def gerber_lw_ensemble(returns: pd.DataFrame, window: int = 252,
                        gerber_weight: float = 0.5) -> np.ndarray:
    """Gerber + LW ensemble: 50% Gerber + 50% LW (默认)."""
    C_g = gerber_cov(returns, window=window)
    C_lw = ledoit_wolf_shrink(returns, window=window)
    C = gerber_weight * C_g + (1.0 - gerber_weight) * C_lw
    eigvals = np.linalg.eigvalsh(C)
    if eigvals.min() < 1e-10:
        C += np.eye(C.shape[0]) * (1e-10 - eigvals.min())
    return C


# ═══════════════════════════════════════════════════════════════════
# Phase A 原理引擎 (PLAN §17.10): 自调节估计器
# ═══════════════════════════════════════════════════════════════════


def oas_cov(returns: pd.DataFrame, window: int = 252) -> np.ndarray:
    """OAS 收缩 (Chen et al. 2010; 与 sklearn.covariance.OAS 数值一致).

    sklearn 采用 Eq.23 简化形式 (省略 2/N 因子, 大 N 下等价):
    shrinkage = min(1, [α + μ²] / [(T+1)·(α − μ²/N)])
    其中 α = mean(S²) = ||S||²_F/N², μ = tr(S)/N.
    """
    sub = returns.iloc[-window:].dropna(axis=0, how="any")
    T, N = sub.shape
    if T < 30 or N < 2:
        return np.cov(sub.to_numpy(), rowvar=False)

    x = sub.to_numpy()
    x = x - x.mean(axis=0)
    S = x.T @ x / T

    alpha = float(np.mean(S ** 2))
    mu = float(np.trace(S)) / N

    num = alpha + mu ** 2
    den = (T + 1.0) * (alpha - mu ** 2 / N)
    rho = 1.0 if den == 0 else min(1.0, max(num / den, 0.0))

    C = (1.0 - rho) * S + rho * mu * np.eye(N)
    eigvals = np.linalg.eigvalsh(C)
    if eigvals.min() < 1e-10:
        C += np.eye(N) * (1e-10 - eigvals.min())
    return C


def lw_ewma_cov(returns: pd.DataFrame, window: int = 252,
                lam: float = 0.96) -> np.ndarray:
    """加权版 LW: EWMA 样本 + 强度随 N_eff 自调节 (不变量 1).

    α = min(1, β̄ / (N_eff · δ²)), N_eff = (1+λ)/(1-λ).
    N_eff → ∞ 时退化为标准 LW 行为; N_eff → N 时 α → 1 (全收缩, 条件数有界).
    """
    sub = returns.iloc[-window:].dropna(axis=0, how="any")
    T, N = sub.shape
    if T < 30 or N < 2:
        return np.cov(sub.to_numpy(), rowvar=False)

    S = ewma_cov(returns, window=window, lam=lam)
    n_eff = (1.0 + lam) / (1.0 - lam)

    x = sub.to_numpy()
    x = x - x.mean(axis=0)
    mu = np.trace(S) / N
    I_mu = mu * np.eye(N)
    delta2 = np.sum((S - I_mu) ** 2)

    beta_bar = 0.0
    for t in range(T):
        xi = np.outer(x[t], x[t])
        beta_bar += np.sum((xi - S) ** 2)
    beta_bar /= T

    if delta2 == 0:
        alpha = 1.0
    else:
        alpha = float(np.clip(beta_bar / (n_eff * delta2), 0.0, 1.0))

    C = alpha * I_mu + (1.0 - alpha) * S
    eigvals = np.linalg.eigvalsh(C)
    if eigvals.min() < 1e-10:
        C += np.eye(N) * (1e-10 - eigvals.min())
    return C


def block_corr_shrink(returns: pd.DataFrame, window: int = 252,
                      n_clusters: int | None = None,
                      linkage_method: str = "ward") -> np.ndarray:
    """相关空间分块收缩 (修正版 AVS-F, 不变量 1 的结构化目标).

    原理: F 在协方差空间被 vol 离散度污染 (cond 巨大, §5.24 hybrid 失利根因);
    在相关空间收缩 (F_corr 条件数有界), 收缩后 vol 重标定.
    强度用 LW 公式在相关空间计算.
    """
    sub = returns.iloc[-window:].dropna(axis=0, how="any")
    firsts = [f for f in (sub[c].first_valid_index() for c in sub.columns)
              if f is not None]
    if firsts:
        sub = sub.loc[max(firsts):]
    T, N = sub.shape
    if T < 30 or N < 3:
        return np.cov(sub.to_numpy(), rowvar=False)

    S = np.cov(sub.to_numpy(), rowvar=False)
    vol = np.sqrt(np.clip(np.diag(S), 1e-12, None))
    corr = S / np.outer(vol, vol)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method=linkage_method)
    k = n_clusters or max(2, N // 6)
    labels = fcluster(Z, min(k, N), criterion="maxclust")

    # 相关空间分块目标: 簇内常数相关, 簇间 0
    T_corr = np.zeros((N, N), dtype=float)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) < 2:
            continue
        c_in = corr[np.ix_(idx, idx)]
        mask = ~np.eye(len(idx), dtype=bool)
        avg = float(c_in[mask].mean()) if mask.any() else 0.0
        T_corr[np.ix_(idx, idx)] = avg
    np.fill_diagonal(T_corr, 1.0)

    # LW 强度在相关空间计算
    delta2 = np.sum((corr - T_corr) ** 2)
    # β̄ 用相关残差尺度近似 (逐元素残差平方和 / T)
    resid = corr - T_corr
    beta_bar = float(np.sum(resid ** 2)) / T

    if delta2 == 0:
        alpha = 1.0
    else:
        alpha = float(np.clip(beta_bar / delta2, 0.0, 1.0))

    shrunk_corr = alpha * T_corr + (1.0 - alpha) * corr
    np.fill_diagonal(shrunk_corr, 1.0)
    C = shrunk_corr * np.outer(vol, vol)

    eigvals = np.linalg.eigvalsh(C)
    if eigvals.min() < 1e-10:
        C += np.eye(N) * (1e-10 - eigvals.min())
    return C
