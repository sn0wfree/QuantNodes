# coding=utf-8
"""AllWeather — 层次风险平价 HRP (PLAN Step 2).

流程 (Lopez de Prado 2016):
    相关矩阵 → 距离 d=√(0.5(1-ρ)) → 层次聚类 (linkage 可选) →
    准对角化 → 递归二分 (簇方差逆比分配) → 长多权重 (Σw=1)

设计要点:
    - linkage 是自由参数 (single/average/ward), D1 备选动作 + Phase 3 对照
    - 渐入池: hrp_weights_from_returns 自动剔除窗口内数据不足的资产
    - 与 RMT 串联: denoise=True 时先 mp_denoise 再 HRP
    - 可用资产 < MIN_ASSETS_FOR_HRP → 退化为等权
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from .config import MIN_ASSETS_FOR_HRP
from .cov_rmt import mp_denoise, sample_cov

VALID_LINKAGES = ("single", "average", "ward")


def _quasi_diag(link_mat: np.ndarray) -> list[int]:
    """层次聚类树 → 叶子排序 (准对角化)."""
    link_mat = link_mat.astype(int)
    sort_ix = pd.Series([link_mat[-1, 0], link_mat[-1, 1]])
    n_items = link_mat[-1, 3]

    while sort_ix.max() >= n_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)
        df0 = sort_ix[sort_ix >= n_items]
        i = df0.index
        j = df0.values - n_items
        sort_ix[i] = link_mat[j, 0]
        s1 = pd.Series(link_mat[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, s1])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _cluster_var(cov: np.ndarray, items: list[int]) -> float:
    """簇方差: 簇内逆方差加权组合的方差."""
    sub = cov[np.ix_(items, items)]
    diag = np.diag(sub)
    if (diag <= 0).any() or np.isnan(diag).any():
        return float(np.nanmean(sub)) if np.isfinite(sub).any() else 1.0
    w = 1.0 / diag
    w = w / w.sum()
    return float(w @ sub @ w)


def hrp_weights(cov: np.ndarray, linkage_method: str = "single") -> np.ndarray:
    """从协方差矩阵计算 HRP 权重.

    Args:
        cov: (N x N) 协方差矩阵 (或相关矩阵, HRP 对缩放不敏感)
        linkage_method: single / average / ward

    Returns:
        (N,) 权重, Σ=1, 非负
    """
    if linkage_method not in VALID_LINKAGES:
        raise ValueError(f"linkage_method 须为 {VALID_LINKAGES}, 得到 {linkage_method}")

    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])

    # 相关矩阵 → 距离
    diag = np.diag(cov)
    std = np.sqrt(np.clip(diag, 1e-12, None))
    corr = cov / np.outer(std, std)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    dist_cond = squareform(dist, checks=False)
    link_mat = linkage(dist_cond, method=linkage_method)

    order = _quasi_diag(link_mat)
    order = [i for i in order if 0 <= i < n]

    weights = pd.Series(1.0, index=order)
    clusters = [order]
    while clusters:
        next_clusters: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]
            left_var = _cluster_var(cov, left)
            right_var = _cluster_var(cov, right)
            total = left_var + right_var
            alpha = 1.0 - left_var / total if total > 0 else 0.5
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            if len(left) > 1:
                next_clusters.append(left)
            if len(right) > 1:
                next_clusters.append(right)
        clusters = next_clusters

    w = np.zeros(n)
    for idx, val in weights.items():
        w[idx] = val
    w = np.clip(w, 1e-12, None)
    return w / w.sum()


def hrp_weights_from_returns(
    returns: pd.DataFrame,
    window: int = 252,
    denoise: str | None = "mp",
    linkage_method: str = "single",
    min_obs_per_asset: int = 60,
) -> pd.Series:
    """从收益率面板计算 HRP 权重 (支持渐入池).

    Args:
        returns: (T x N) 收益率面板, NaN = 未上市/无数据
        window: 协方差估计窗口
        denoise: 去噪方法 'mp' (RMT 裁剪) / 'lw' (Ledoit-Wolf 收缩) / None (原始样本协方差)
        linkage_method: 聚类方法
        min_obs_per_asset: 资产纳入窗口所需的最少非空观测

    Returns:
        pd.Series: 与 returns.columns 同索引; 数据不足的资产权重 = 0;
        可用资产 < MIN_ASSETS_FOR_HRP → 可用资产等权 (仍无 → 全列等权)
    """
    cols = list(returns.columns)
    w_last = returns.iloc[-window:]

    # 窗口内观测充足的资产
    avail = [c for c in cols if w_last[c].notna().sum() >= min_obs_per_asset]

    if len(avail) < MIN_ASSETS_FOR_HRP:
        if not avail:
            return pd.Series(1.0 / len(cols), index=cols)
        eq = pd.Series(0.0, index=cols)
        eq[avail] = 1.0 / len(avail)
        return eq

    sub = returns[avail].iloc[-window:].dropna(axis=0, how="any")
    # 防窗口坍缩: 迟上市资产会砍短 dropna 交集 → 从最晚首有效日起截取
    firsts = [f for f in (sub[c].first_valid_index() for c in avail) if f is not None]
    if firsts:
        sub = sub.loc[max(firsts):]
    n_obs = len(sub)
    if denoise == "lw":
        from .cov_rmt import ledoit_wolf_shrink
        cov = ledoit_wolf_shrink(sub, window=window)
    else:
        cov = sample_cov(sub, window=window)
        if denoise == "mp":
            cov = mp_denoise(cov, n_obs=n_obs)

    w = hrp_weights(cov, linkage_method=linkage_method)
    out = pd.Series(0.0, index=cols)
    out[avail] = w
    # 数值安全: 再归一化 (可用资产权重和已=1, 此步防浮点漂移)
    total = out.sum()
    if total > 0:
        out = out / total
    return out
