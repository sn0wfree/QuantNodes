"""v7.0 Macro Beta Regression — 5 宏观特征 × 7 ETF 回归驱动.

[Stage 30.5] 5 Macro Dynamic 方案之三.

[核心算法]
    对每只 ETF, 在 expanding window 上回归日收益 on 5 macro features:
        r_etf,d = α + β_1*PMI_d + β_2*CPI_d + β_3*M2_d + β_4*ΔCN10Y + β_5*ΔUS10Y + ε
    调仓日: 用截至 t-1 的 β, current macro × β → expected return
    top K ETF 等权

[业界对应] 中信证券动态加权法 (滚动回归)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _get_macro_features(tl_df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """截至 as_of, 提取 5 维 macro features (zscore)."""
    cols = ["PMI_zscore", "CPI_zscore", "M2_zscore", "CN10Y_zscore", "US10Y_zscore"]
    return tl_df.loc[:as_of, cols].dropna()


def compute_etf_macro_betas(
    panel: pd.DataFrame,
    tl_df: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int = 252,
    min_samples: int = 60,
) -> Optional[pd.DataFrame]:
    """计算 7 ETF × 5 macro features 的回归系数 (OLS).

    Args:
        panel: 收盘价面板.
        tl_df: HMM timeline (含 zscore 列).
        as_of: cutoff.
        lookback: rolling window (0 = expanding).
        min_samples: 最少样本天数.

    Returns:
        DataFrame: index=etf_codes, columns=['const'] + 5 features.
        或 None (样本不足).
    """
    pa = panel.loc[:as_of]
    rets = pa.pct_change().dropna()
    if lookback and lookback < len(rets):
        rets = rets.iloc[-lookback:]

    macro = _get_macro_features(tl_df, as_of)
    common_idx = rets.index.intersection(macro.index)
    if len(common_idx) < min_samples:
        return None

    rets_a = rets.loc[common_idx]
    macro_a = macro.loc[common_idx]

    X = np.column_stack([np.ones(len(macro_a)), macro_a.values])
    betas = {}
    for etf in rets_a.columns:
        y = rets_a[etf].values
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            betas[etf] = coef
        except np.linalg.LinAlgError:
            betas[etf] = np.zeros(6)
    return pd.DataFrame(betas, index=["const", "PMI", "CPI", "M2", "CN10Y", "US10Y"]).T


def predict_etf_returns(
    betas: pd.DataFrame,
    current_macro: pd.Series,
) -> pd.Series:
    """current macro × beta → expected return (annualized).

    Args:
        betas: (N_etf, 6) 回归系数.
        current_macro: 5 维 macro zscore.

    Returns:
        Series: 7 ETF expected return.
    """
    if current_macro is None or len(current_macro) != 5:
        return pd.Series(0.0, index=betas.index)
    feature_order = ["PMI", "CPI", "M2", "CN10Y", "US10Y"]
    x = np.array([1.0] + [current_macro[f] for f in feature_order])
    expected = (betas.values * x).sum(axis=1)
    return pd.Series(expected * 252, index=betas.index)


def run_beta_v7_backtest(
    panel: pd.DataFrame,
    tl_df: pd.DataFrame,
    lookback: int = 252,
    k: int = 5,
    min_samples: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Walk-forward Macro Beta 回归 Top-K 调仓.

    Args:
        panel: 收盘价面板.
        tl_df: HMM timeline.
        lookback: 回归窗口 (0 = expanding).
        k: top K 数量.
        min_samples: 最少样本天数.

    Returns:
        (nav_df, weights_history, metrics)
    """
    from .dynamic_allocation import _compute_metrics

    etf_universe = list(panel.columns)
    rebal_dates = []
    for d in panel.resample("BME").last().index:
        if d >= tl_df.index[0] and d in panel.index:
            rebal_dates.append(d)

    feature_cols = ["PMI_zscore", "CPI_zscore", "M2_zscore", "CN10Y_zscore", "US10Y_zscore"]
    feature_names = ["PMI", "CPI", "M2", "CN10Y", "US10Y"]

    nav_path = []
    weights_log = []
    for i, d in enumerate(rebal_dates):
        if i == 0:
            w = {c: 1.0 / len(etf_universe) for c in etf_universe}
            cur_state = "init"
        else:
            betas = compute_etf_macro_betas(panel, tl_df, rebal_dates[i - 1], lookback, min_samples)
            cur_state_series = tl_df["regime"].reindex([d], method="ffill")
            cur_state = cur_state_series.iloc[0] if len(cur_state_series) else "init"
            if betas is None or betas.empty:
                w = {c: 1.0 / len(etf_universe) for c in etf_universe}
            else:
                cur_macro_vals = tl_df.loc[:d, feature_cols].iloc[-1]
                cur_macro = pd.Series(cur_macro_vals.values, index=feature_names)
                pred = predict_etf_returns(betas, cur_macro)
                pred = pred.reindex(etf_universe).dropna()
                kk = min(k, len(pred))
                top = pred.nlargest(kk).index.tolist()
                w = {c: (1.0 / kk if c in top else 0.0) for c in etf_universe}
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
