"""AVS 分层去噪协方差 (设计文稿 §3.1, PLAN §17).

Ward 层次聚类 → 分块目标矩阵 F (簇内保相关/簇间=0)
→ 闭式最优收缩 δ* = min{1, (N+2)/(T+N+2)·||F||²/||S-F||²}
→ C = (1-δ*)S + δ*F

设计要点:
    - Ward 聚类与 HRP 共享同一相关距离, 但各用各的树 (D.0 不做共享优化)
    - 分块目标 F 的簇数由参数控制 (默认 N//6, 最小 2)
    - δ* 随 q=T/N 自适应: q 大 (少样本) → δ* 大 → 强收缩; q 小 → 贴近样本
    - PSD 修复: 负特征值 → 钳制到 1e-10
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from .hrp import MIN_ASSETS_FOR_HRP

# 最少资产 / 观测数 (与 cov_rmt.MIN_OBS 对齐)
MIN_OBS = 60


def hierarchical_avs(
    returns: pd.DataFrame,
    window: int = 252,
    n_clusters: int | None = None,
    linkage_method: str = "ward",
    min_obs_per_asset: int = MIN_OBS,
) -> tuple[np.ndarray, float, object | None, int]:
    """AVS 分层去噪协方差.

    Args:
        returns: (T x N) 收益率面板
        window: 协方差估计窗口
        n_clusters: Ward 树切割簇数; None = max(2, N//6)
        linkage_method: 聚类方法 (ward / single / average)
        min_obs_per_asset: 资产纳入窗口所需最少非空观测

    Returns:
        (C_shrunk, delta, link_mat, n_clusters_used)
        C_shrunk: (N x N) 去噪后协方差矩阵 (PSD)
        delta: 收缩强度 ∈ [0, 1]
        link_mat: scipy linkage 矩阵 (None 若资产不足)
        n_clusters_used: 实际切割簇数
    """
    cols = list(returns.columns)
    w_last = returns.iloc[-window:]
    avail = [c for c in cols if w_last[c].notna().sum() >= min_obs_per_asset]

    if len(avail) < MIN_ASSETS_FOR_HRP:
        sub = returns[avail].iloc[-window:].dropna(axis=0, how="any")
        if len(avail) < 2 or len(sub) < 30:
            S = np.eye(max(len(avail), 1))
            return S, 0.0, None, 0
        S = np.cov(sub.to_numpy(), rowvar=False)
        return S, 0.0, None, 0

    sub = returns[avail].iloc[-window:].dropna(axis=0, how="any")
    firsts = [f for f in (sub[c].first_valid_index() for c in avail) if f is not None]
    if firsts:
        sub = sub.loc[max(firsts):]
    T, N = sub.shape

    if T < 30 or N < 3:
        S = np.cov(sub.to_numpy(), rowvar=False)
        eigvals = np.linalg.eigvalsh(S)
        if eigvals.min() < 1e-10:
            S += np.eye(N) * (1e-10 - eigvals.min())
        return S, 0.0, None, 0

    # ── 样本协方差 ──
    S = np.cov(sub.to_numpy(), rowvar=False)
    vol = np.sqrt(np.clip(np.diag(S), 1e-12, None))

    # ── 相关矩阵 ──
    corr = S / np.outer(vol, vol)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    # ── Ward 聚类 ──
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    dist_cond = squareform(dist, checks=False)
    link_mat = linkage(dist_cond, method=linkage_method)

    # ── 切割树 → 分块目标 F ──
    if n_clusters is None:
        n_clusters = max(2, N // 6)
    n_clusters = min(n_clusters, N)
    labels = fcluster(link_mat, n_clusters, criterion="maxclust")

    F_corr = np.zeros((N, N), dtype=float)
    for lab in np.unique(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) < 2:
            # 单资产簇: 只设对角 = 1
            continue
        c_in = corr[np.ix_(idx, idx)]
        mask = ~np.eye(len(idx), dtype=bool)
        avg_corr = float(c_in[mask].mean()) if mask.any() else 0.0
        F_corr[np.ix_(idx, idx)] = avg_corr
    np.fill_diagonal(F_corr, 1.0)

    # ── 目标矩阵转协方差空间 ──
    F_cov = F_corr * np.outer(vol, vol)

    # ── 闭式最优收缩 δ* ──
    diff = S - F_cov
    norm_F = float(np.sum(F_cov ** 2))
    norm_diff = float(np.sum(diff ** 2))

    if norm_diff < 1e-20:
        delta = 1.0
    else:
        delta = min(1.0, (N + 2) / (T + N + 2) * norm_F / norm_diff)

    # ── 收缩 ──
    C = (1.0 - delta) * S + delta * F_cov

    # ── PSD 修复 ──
    eigvals = np.linalg.eigvalsh(C)
    min_eig = float(eigvals.min())
    if min_eig < 1e-10:
        C += np.eye(N) * (1e-10 - min_eig)

    return C, delta, link_mat, n_clusters
