# coding=utf-8
"""AllWeather — 协方差去噪 (PLAN §5.3 / Step 1).

两个估计器:
    - mp_denoise: Marchenko-Pastur 特征值裁剪 (Laloux/Bouchaud RMT)
      在相关矩阵单位下计算 λ+ = (1+√(N/T))², 噪声特征值收缩为均值,
      信号特征值原样保留, 再还原回协方差单位.
    - ledoit_wolf_shrink: Ledoit-Wolf (2004) 线性收缩 (向 scaled identity),
      作为 §5.3 的对照方法.

约定 (CFM 期货警告的应对): 期货板块内强相关 → 板块簇真实特征值小,
MP 假设的 i.i.d. 在肥尾下会偏移 — 因此提供 mp 与 lw 两种方法, Phase 3 并行报告.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MIN_OBS = 20


def sample_cov(returns: pd.DataFrame, window: int = 252) -> np.ndarray:
    """滚动窗口样本协方差; NaN 行整行剔除, 样本不足返回单位阵.

    Args:
        returns: (T x N) 收益率 DataFrame
        window: 回看窗口 (交易日)

    Returns:
        (N x N) ndarray; obs < MIN_OBS 时返回 identity
    """
    w = returns.iloc[-window:]
    w = w.dropna(axis=0, how="any")
    n_obs, n_assets = w.shape
    if n_obs < MIN_OBS:
        return np.eye(n_assets)
    cov = np.cov(w.to_numpy(), rowvar=False)
    cov = np.atleast_2d(cov)
    return _fix_psd(cov)


def mp_lambda_plus(n_obs: int, n_assets: int) -> float:
    """Marchenko-Pastur 噪声带上界 (相关矩阵单位): (1+√(N/T))²."""
    q = n_obs / n_assets
    if q <= 0:
        return 2.0
    return float((1 + np.sqrt(1.0 / q)) ** 2)


def _fix_psd(mat: np.ndarray) -> np.ndarray:
    """对称化 + 负特征值截断到 1e-10 (保证半正定)."""
    mat = (mat + mat.T) / 2
    eigvals, eigvecs = np.linalg.eigh(mat)
    if eigvals.min() < 0:
        eigvals = np.clip(eigvals, 1e-10, None)
        mat = eigvecs @ np.diag(eigvals) @ eigvecs.T
        mat = (mat + mat.T) / 2
    return mat


def mp_denoise(cov: np.ndarray, n_obs: int) -> np.ndarray:
    """RMT 去噪: 相关矩阵单位下 MP 裁剪, 还原回协方差.

    输入判定: 对角线全为 1 → 视作相关矩阵 (跳过 vol 还原);
    否则视作协方差, 先转相关矩阵, 去噪后还原.

    Args:
        cov: (N x N) 协方差或相关矩阵
        n_obs: 估计窗口的观测数 T (决定 λ+)

    Returns:
        (N x N) 去噪后矩阵 (与输入同类型/单位)
    """
    cov = np.asarray(cov, dtype=float)
    n_assets = cov.shape[0]
    diag = np.diag(cov)
    is_corr = np.allclose(diag, 1.0, atol=1e-8)

    if is_corr:
        corr = cov
        vol = np.ones(n_assets)
    else:
        vol = np.sqrt(np.clip(diag, 1e-12, None))
        corr = cov / np.outer(vol, vol)

    eigvals, eigvecs = np.linalg.eigh(corr)
    lam_plus = mp_lambda_plus(n_obs, n_assets)

    noise_mask = eigvals <= lam_plus
    if noise_mask.any() and not noise_mask.all():
        # 常数残差法: 噪声特征值替换为噪声均值 (保持 trace)
        eigvals = np.where(noise_mask, eigvals[noise_mask].mean(), eigvals)
    elif noise_mask.all():
        # 无信号: 全部收缩为均值 (等价于按均值对角化)
        eigvals = np.full_like(eigvals, eigvals.mean())

    corr_dn = eigvecs @ np.diag(eigvals) @ eigvecs.T
    corr_dn = _fix_psd(corr_dn)

    if is_corr:
        return corr_dn
    return corr_dn * np.outer(vol, vol)


def ledoit_wolf_shrink(returns: pd.DataFrame, window: int = 252) -> np.ndarray:
    """Ledoit-Wolf (2004) 线性收缩: S* = α·F + (1-α)·S, F = trace(S)/N · I.

    收缩强度 α 由最小化 Frobenius 损失解析给出.
    对照方法 (§5.3): MP 裁剪 vs LW 收缩, Phase 3 并行报告 OOS 波动预测偏差.

    Returns:
        (N x N) 收缩后协方差; obs < MIN_OBS 返回 identity
    """
    w = returns.iloc[-window:]
    w = w.dropna(axis=0, how="any")
    n_obs, n_assets = w.shape
    if n_obs < MIN_OBS:
        return np.eye(n_assets)

    x = w.to_numpy()
    x = x - x.mean(axis=0)
    # ddof=1, 与 sample_cov 的 np.cov 一致
    sample = np.cov(x, rowvar=False)

    # 收缩目标: trace 保均值单位阵
    mu = np.trace(sample) / n_assets
    target = mu * np.eye(n_assets)

    # LW 2004: δ² = ||S - F||²_F, β²̄ = mean ||x_i x_i' - S||²_F
    beta_bar = 0.0
    for i in range(n_obs):
        xi = np.outer(x[i], x[i])
        beta_bar += np.sum((xi - sample) ** 2)
    beta_bar /= n_obs
    delta2 = np.sum((sample - target) ** 2)

    if delta2 == 0:
        alpha = 0.0
    else:
        alpha = float(np.clip(beta_bar / (n_obs * delta2), 0.0, 1.0))

    shrunk = alpha * target + (1 - alpha) * sample
    return _fix_psd(shrunk)
