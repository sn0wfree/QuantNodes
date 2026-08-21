# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/sign_constrained_lasso.py

§2.2 符号约束Lasso + walk-forward 时序交叉验证.

目标: 在每个重估时点 T 用 [T-156, T) 周拟合 ΔlnM_t = α + Σ w_j X_j,t
约束: w_j ≥ 0,  L1 惩罚: λ Σ|w_j|

λ 候选: 17 个 logspace(-7, -3, 17)
重估频率: 26 周 (半年)
最短训练: 156 周 (3 年)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error


def rolling_zscore(s: pd.Series, window: int = 104, min_periods: int = 13) -> pd.Series:
    """滚动 z-score (支持最小窗口).

    z = (x - μ_w) / σ_w, σ_w=0 时返回 0 (避免除零)

    参数:
        window: 默认窗口 (建议52周)
        min_periods: 最小窗口 (13周), 小于此值返回NaN
    """
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    z = (s - mu) / sd.replace(0, 1)
    return z


def fit_sign_constrained(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float,
    max_iter: int = 5000,
) -> tuple[np.ndarray, float]:
    """单次符号约束Lasso 拟合 (positive=True 强制 w_j ≥ 0)."""
    Xv = X.values
    yv = y.values
    model = Lasso(alpha=alpha, positive=True, fit_intercept=True,
                  max_iter=max_iter, selection="random")
    model.fit(Xv, yv)
    coefs = np.asarray(model.coef_)
    return coefs, float(model.intercept_)


def cv_select_alpha(
    X: pd.DataFrame,
    y: pd.Series,
    alphas: np.ndarray,
    n_splits: int = 5,
    embargo: int = 13,
) -> tuple[float, list[float]]:
    """时序交叉验证选 λ.

    采用 ExpandingWindow 时间序列CV, 折间留 embargo 周避免信息泄露.
    """
    n = len(X)
    fold_size = n // (n_splits + 1)
    if fold_size < 26:
        return float(alphas[0]), [float("inf")] * len(alphas)

    splits = []
    for k in range(1, n_splits + 1):
        train_end = k * fold_size
        test_start = train_end + embargo
        test_end = min(test_start + fold_size, n)
        if test_end - test_start < 13:
            continue
        splits.append((train_end, test_start, test_end))

    if not splits:
        return float(alphas[0]), [float("inf")] * len(alphas)

    mse_per_alpha = np.zeros(len(alphas))
    for (tr_end, te_start, te_end) in splits:
        for i, a in enumerate(alphas):
            Xtr = X.iloc[:tr_end].values
            ytr = y.iloc[:tr_end].values
            Xte = X.iloc[te_start:te_end].values
            yte = y.iloc[te_start:te_end].values

            model = Lasso(alpha=float(a), positive=True, fit_intercept=True,
                          max_iter=5000, selection="random")
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            mse_per_alpha[i] += mean_squared_error(yte, pred)

    best_idx = int(np.argmin(mse_per_alpha))
    return float(alphas[best_idx]), mse_per_alpha.tolist()


def walk_forward_lasso(
    X: pd.DataFrame,
    y: pd.Series,
    refit_weeks: int = 26,
    min_train_weeks: int = 156,
    alpha_grid: np.ndarray | None = None,
    return_oos: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward 时序Lasso.

    返回:
        coefs_df: 行=重估时点, 列=各因子系数
        oos_pred: OOS 预测 ΔlnM_t (供后续还原基本面中枢)
    """
    if alpha_grid is None:
        alpha_grid = np.logspace(-7, -3, 17)

    n = len(X)
    refit_points = list(range(min_train_weeks, n, refit_weeks))
    if n - 1 not in refit_points:
        refit_points.append(n - 1)

    coef_rows = []
    oos_records = []

    for T in refit_points:
        X_hist = X.iloc[:T].dropna()
        y_hist = y.iloc[:T].reindex(X_hist.index).dropna()
        if len(X_hist) < min_train_weeks or len(y_hist) < min_train_weeks:
            continue

        try:
            best_alpha, _ = cv_select_alpha(X_hist, y_hist, alpha_grid)
            coefs, intercept = fit_sign_constrained(X_hist, y_hist, alpha=best_alpha)
        except Exception:
            best_alpha = float(alpha_grid[len(alpha_grid) // 2])
            coefs, intercept = fit_sign_constrained(X_hist, y_hist, alpha=best_alpha)

        coef_rows.append({"refit_idx": T, "alpha": best_alpha, "intercept": intercept,
                          "refit_date": X.index[T]})
        for j, col in enumerate(X.columns):
            coef_rows[-1][col] = float(coefs[j])

        # OOS: 用本次系数预测到下次重估之前的所有样本
        next_T = next((tp for tp in refit_points if tp > T), n)
        if return_oos and T < n:
            X_future = X.iloc[T:next_T].copy()
            X_future_clean = X_future.dropna()
            if len(X_future_clean) > 0:
                for fdate in X_future_clean.index:
                    xv = X_future_clean.loc[fdate].values
                    if not np.isnan(xv).any():
                        pred = intercept + np.dot(xv, coefs)
                        oos_records.append({"trade_date": fdate, "oos_pred": float(pred)})

    coefs_df = pd.DataFrame(coef_rows).set_index("refit_date")
    oos_df = pd.DataFrame(oos_records).set_index("trade_date") if oos_records else pd.DataFrame()
    return coefs_df, oos_df


def rolling_phy_center(
    mu_market: pd.Series,
    oos_pred: pd.Series,
    window: int = 13,
) -> pd.Series:
    """13周滚动基本面中枢还原 (报告§2.2).

    p_t^phy = M_{t-13} * exp(Σ_{s=t-12..t} ŷ_s)

    注意: 锚点 M_{t-13} 每周向前滚动, 窗口保持13周.
    NaN 处理: 前 window-1 周以及 OOS 预测起点之前无值 → 保留 NaN, 不外推.
    """
    pred_aligned = oos_pred.reindex(mu_market.index)
    cum_pred = pred_aligned.cumsum()
    anchor_idx = mu_market.shift(window)
    anchor_log = np.log(anchor_idx)
    window_sum = cum_pred - cum_pred.shift(window)
    phy_log = anchor_log + window_sum
    return pd.Series(np.exp(phy_log), index=mu_market.index)


def compute_valuation_gap(mu_market: pd.Series, p_phy: pd.Series) -> pd.Series:
    """估值缺口 = ln(p_phy / M)."""
    return pd.Series(np.log(p_phy / mu_market), index=mu_market.index)


__all__ = [
    "rolling_zscore",
    "fit_sign_constrained",
    "cv_select_alpha",
    "walk_forward_lasso",
    "rolling_phy_center",
    "compute_valuation_gap",
]