"""v7.0 Black-Litterman — 5 状态视图的 BL 资产配置.

[Stage 30.5] 5 Macro Dynamic 方案之二.

[核心算法]
    Posterior: E[R|views] = [(τΣ)^-1 + P'Ω^-1 P]^-1 [(τΣ)^-1 π + P'Ω^-1 Q]

[参数]
    - π: prior 期望收益 (默认 equal 7% each)
    - Σ: expanding 252d 日收益协方差 (年化, Ledoit-Wolf shrinkage)
    - τ: prior 不确定度 (Idzorek 推荐 0.05)
    - P: 选股矩阵 I_7 (one view per ETF)
    - Q: state-conditional forward 21d 均值 (state 历史 expanding)
    - Ω: diag(τ × σ² / n_samples) — 桶稀疏修正

[业界对应] 学术经典 / 中信证券动态加权 (BL variant)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_expanding_cov(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    window: int = 252,
    use_ledoit_wolf: bool = True,
) -> np.ndarray:
    """Expanding (or rolling) 日收益协方差矩阵, 年化.

    Args:
        panel: 收盘价面板.
        as_of: cutoff.
        window: rolling window 天数, 0 = expanding.
        use_ledoit_wolf: 是否用 Ledoit-Wolf shrinkage.

    Returns:
        (N_etf, N_etf) 年化协方差矩阵.
    """
    pa = panel.loc[:as_of]
    rets = pa.pct_change().dropna()
    if window and window < len(rets):
        rets = rets.iloc[-window:]
    if len(rets) < 30:
        return np.eye(len(panel.columns)) * 0.04

    cov_daily = rets.cov().values
    if use_ledoit_wolf:
        cov_daily = _ledoit_wolf_shrink(rets.values, cov_daily)
    return cov_daily * 252


def _ledoit_wolf_shrink(X: np.ndarray, sample_cov: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage: cov_shrunk = δ*F + (1-δ)*S, F=diag(mean var)."""
    n, p = X.shape
    if n < 2:
        return sample_cov
    var_mean = np.mean(np.diag(sample_cov))
    F = np.eye(p) * var_mean
    diff = sample_cov - F
    delta = _optimal_shrinkage_intensity(X, sample_cov, F)
    delta = max(0.0, min(1.0, delta))
    return delta * F + (1 - delta) * sample_cov


def _optimal_shrinkage_intensity(X: np.ndarray, S: np.ndarray, F: np.ndarray) -> float:
    """Ledoit-Wolf 闭式解 (简化版)."""
    n, p = X.shape
    X_c = X - X.mean(axis=0)
    diff = X_c.T @ X_c / n - S
    sq = float(np.sum(diff ** 2))
    if sq < 1e-12:
        return 0.0
    trace_F2 = float(np.sum(np.diag(F) ** 2))
    pi_hat = sq
    gamma = (n / (n - 1) ** 2) * trace_F2 if n > 1 else trace_F2
    if gamma < 1e-12:
        return 0.0
    return min(1.0, pi_hat / gamma)


def compute_state_view_q(
    state_means: pd.DataFrame,
    cur_state: str,
    min_samples: int = 3,
    default_q: float = 0.07,
) -> Optional[np.ndarray]:
    """BL 视图 Q: cur_state 下 7 ETF forward 21d 均值 (年化后).

    Args:
        state_means: state × ETF forward 收益 DataFrame (含 'state' 列).
        cur_state: 当前 state.
        min_samples: 最少样本数.
        default_q: fallback Q (prior equal 7%).

    Returns:
        (N_etf,) ndarray 或 None (cold start).
    """
    if state_means.empty or "state" not in state_means.columns:
        return None
    s_data = state_means[state_means["state"] == cur_state]
    if len(s_data) < min_samples:
        return None
    means = s_data.drop(columns="state").mean()
    return means.fillna(0).values * 12


def compute_view_uncertainty_omega(
    state_means: pd.DataFrame,
    cur_state: str,
    sigma: np.ndarray,
    tau: float = 0.05,
    min_samples: int = 3,
) -> np.ndarray:
    """BL 视图不确定性 Ω: diag(τ × σ² / n_samples).

    Args:
        state_means: state × ETF forward 收益 DataFrame.
        cur_state: 当前 state.
        sigma: (N,N) 协方差 (用于 σ²).
        tau: prior 不确定度.
        min_samples: 最少样本数 (Q 失效时返回大 Ω).

    Returns:
        (N_etf, N_etf) 对角矩阵.
    """
    n_etf = sigma.shape[0]
    if state_means.empty or "state" not in state_means.columns:
        return np.eye(n_etf) * 1.0
    s_data = state_means[state_means["state"] == cur_state]
    n_samples = max(1, len(s_data))
    if n_samples < min_samples:
        return np.eye(n_etf) * 1.0
    var_annual = np.diag(sigma) * 0.1
    omega_diag = tau * var_annual / n_samples
    return np.diag(omega_diag)


def compute_bl_posterior(
    pi: np.ndarray,
    sigma: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray,
    tau: float = 0.05,
) -> np.ndarray:
    """Black-Litterman 后验: E[R|views].

    E[R|views] = [(τΣ)^-1 + P'Ω^-1 P]^-1 [(τΣ)^-1 π + P'Ω^-1 Q]
    """
    tau_sigma = tau * sigma
    tau_sigma_inv = np.linalg.pinv(tau_sigma)
    omega_inv = np.linalg.pinv(omega)
    p_omega_inv = p.T @ omega_inv

    precision = tau_sigma_inv + p_omega_inv @ p
    precision_inv = np.linalg.pinv(precision)
    rhs = tau_sigma_inv @ pi + p_omega_inv @ q
    return precision_inv @ rhs


def compute_bl_weights(
    posterior_returns: np.ndarray,
    sigma: np.ndarray,
    lambda_risk: float = 1.0,
    max_weight: float = 0.30,
    etf_universe: Optional[list[str]] = None,
) -> dict[str, float]:
    """从 posterior 导出 long-only 权重 (w ∝ posterior, 30% cap, normalize).

    Args:
        posterior_returns: (N,) 后验收益.
        sigma: (N,N) 协方差 (用于风险预算).
        lambda_risk: 风险厌恶系数.
        max_weight: 单只 ETF 最大权重.
        etf_universe: ETF 名称列表.

    Returns:
        dict[etf_code] -> weight, sum=1.
    """
    n = len(posterior_returns)
    if etf_universe is None:
        etf_universe = [f"e{i}" for i in range(n)]
    raw = np.maximum(0.0, posterior_returns)
    if raw.sum() < 1e-9:
        return {c: 1.0 / n for c in etf_universe}
    w = raw / raw.sum()
    w = np.minimum(w, max_weight)
    w = w / w.sum()
    return {c: float(w[i]) for i, c in enumerate(etf_universe)}


def run_bl_v7_backtest(
    panel: pd.DataFrame,
    tl_df: pd.DataFrame,
    tau: float = 0.05,
    lambda_risk: float = 1.0,
    max_weight: float = 0.30,
    forward_days: int = 21,
    min_samples: int = 3,
    cov_window: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Walk-forward BL 调仓.

    Args:
        panel: 收盘价面板.
        tl_df: HMM timeline.
        tau: prior 不确定度.
        lambda_risk: 风险厌恶系数.
        max_weight: 单 ETF 权重上限.
        forward_days: state-conditional forward window.
        min_samples: state 最少月数.
        cov_window: 协方差 window (0 = expanding).

    Returns:
        (nav_df, weights_history, metrics)
    """
    from .dynamic_allocation import compute_state_conditional_means, _compute_metrics

    etf_universe = list(panel.columns)
    n_etf = len(etf_universe)
    pi = np.full(n_etf, 0.07)

    rebal_dates = []
    for d in panel.resample("ME").last().index:
        if d >= tl_df.index[0] and d in panel.index:
            rebal_dates.append(d)

    nav_path = []
    weights_log = []
    for i, d in enumerate(rebal_dates):
        if i == 0:
            w = {c: 1.0 / n_etf for c in etf_universe}
            cur_state = "init"
        else:
            fwd = compute_state_conditional_means(panel, tl_df, rebal_dates[i - 1], forward_days)
            cur_state_series = tl_df["regime"].reindex([d], method="ffill")
            cur_state = cur_state_series.iloc[0] if len(cur_state_series) else "init"
            sigma = compute_expanding_cov(panel, rebal_dates[i - 1], window=cov_window)
            q = compute_state_view_q(fwd, cur_state, min_samples=min_samples)
            omega = compute_view_uncertainty_omega(fwd, cur_state, sigma, tau=tau, min_samples=min_samples)
            if q is None:
                w = {c: 1.0 / n_etf for c in etf_universe}
            else:
                p = np.eye(n_etf)
                posterior = compute_bl_posterior(pi, sigma, p, q, omega, tau=tau)
                w = compute_bl_weights(
                    posterior, sigma, lambda_risk=lambda_risk,
                    max_weight=max_weight, etf_universe=etf_universe,
                )
        weights_log.append({"date": d, "state": cur_state, **w})

        next_d = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else panel.index[-1]
        seg = panel.loc[d:next_d]
        if len(seg) < 2:
            continue
        seg_ret = seg.iloc[-1] / seg.iloc[0]
        port_ret = sum(w.get(c, 0) * (seg_ret.get(c, 1) - 1) for c in etf_universe) + 1
        nav_path.append({"date": next_d, "nav": port_ret})

    nav_df = pd.DataFrame(nav_path).set_index("date")
    nav_df["nav_cum"] = nav_df["nav"].cumprod()
    nav_df["daily_ret"] = nav_df["nav_cum"].pct_change()
    weights_df = pd.DataFrame(weights_log).set_index("date")
    metrics = _compute_metrics(nav_df["nav_cum"])
    return nav_df, weights_df, metrics
