# coding=utf-8
"""5 维度归因分析 (contribution analysis) - Stage 7.

参考 reports/contribution_analysis.md 和 reports/*.csv.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from .universe import ETFPool


# 周期定义 (来自 period_contribution.csv)
DEFAULT_PERIODS: list[tuple[str, str, str]] = [
    ("2019 普涨", "2019-01-01", "2019-12-31"),
    ("2020H1 COVID", "2020-01-01", "2020-06-30"),
    ("2020H2-2021 反弹", "2020-07-01", "2021-12-31"),
    ("2022 熊市", "2022-01-01", "2022-12-31"),
    ("2023 修复", "2023-01-01", "2023-12-31"),
    ("2024 调整", "2024-01-01", "2024-12-31"),
    ("2025 YTD", "2025-01-01", "2025-07-06"),
]


def reconstruct_daily_weights(
    states: list,
    rebalance_dates: list[pd.Timestamp],
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """重建每日权重.

    Args:
        states: PortfolioState 列表 (每个 rebalance 一个)
        rebalance_dates: 调仓日列表
        trading_dates: 完整交易日索引

    Returns:
        DataFrame: index=trading_dates, columns=codes, 值 = 每日权重
    """
    if not states or not rebalance_dates:
        return pd.DataFrame(index=trading_dates)

    all_codes = set()
    for s in states:
        all_codes.update(s.weights.keys())
    all_codes = sorted(all_codes)

    weights_df = pd.DataFrame(0.0, index=trading_dates, columns=all_codes)

    for state, rebal_date in zip(states, rebalance_dates):
        if rebal_date not in weights_df.index:
            continue
        for code, w in state.weights.items():
            if code in weights_df.columns:
                weights_df.loc[rebal_date, code] = w

    weights_df = weights_df.replace(0.0, np.nan).ffill().fillna(0.0)
    return weights_df


def etf_contribution(
    nav_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    pool: ETFPool,
) -> pd.DataFrame:
    """ETF 维度贡献.

    Returns DataFrame with columns:
        code, frequency, avg_weight, total_return, return_contrib
    """
    rows = []
    for code in weights_df.columns:
        w = weights_df[code]
        nonzero = w > 0
        freq = float(nonzero.mean())
        avg_w = float(w[nonzero].mean()) if nonzero.any() else 0.0

        if code in nav_df.columns:
            nav = nav_df[code].dropna()
            if len(nav) > 1:
                total_ret = float(nav.iloc[-1] / nav.iloc[0] - 1)
            else:
                total_ret = 0.0
        else:
            total_ret = 0.0

        ret_contrib = avg_w * total_ret
        rows.append({
            "code": code,
            "frequency": freq,
            "avg_weight": avg_w,
            "total_return": total_ret,
            "return_contrib": ret_contrib,
        })

    df = pd.DataFrame(rows).sort_values("return_contrib", ascending=False)
    return df.reset_index(drop=True)


def category_contribution(
    weights_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    pool: ETFPool,
) -> pd.DataFrame:
    """类别维度贡献.

    Returns DataFrame with columns:
        category, avg_weight, return_contrib, frequency, n_codes
    """
    cat_data: dict[str, dict] = {}
    for code in weights_df.columns:
        try:
            cat = pool.category_of(code).value
        except KeyError:
            continue
        if cat not in cat_data:
            cat_data[cat] = {
                "codes": [],
                "weight_sum": 0.0,
                "weight_count": 0,
                "weighted_ret": 0.0,
            }

        w = weights_df[code]
        nonzero = w > 0
        if nonzero.any():
            avg_w = float(w[nonzero].mean())
            cat_data[cat]["weight_sum"] += avg_w
            cat_data[cat]["weight_count"] += 1
            cat_data[cat]["codes"].append(code)

            if code in nav_df.columns:
                nav = nav_df[code].dropna()
                if len(nav) > 1:
                    total_ret = float(nav.iloc[-1] / nav.iloc[0] - 1)
                    cat_data[cat]["weighted_ret"] += avg_w * total_ret

    rows = []
    for cat, data in cat_data.items():
        avg_weight = data["weight_sum"] / max(data["weight_count"], 1)
        # frequency: 加权平均
        n_codes = len(data["codes"])
        freq = data["weight_count"] / max(len(weights_df), 1)
        rows.append({
            "category": cat,
            "avg_weight": avg_weight,
            "return_contrib": data["weighted_ret"],
            "frequency": freq,
            "n_codes": n_codes,
        })

    df = pd.DataFrame(rows).sort_values("return_contrib", ascending=False)
    return df.reset_index(drop=True)


def risk_contribution(
    weights_df: pd.DataFrame,
    cov: np.ndarray,
    codes: list[str] | None = None,
) -> pd.DataFrame:
    """风险贡献.

    vol_contrib_i = w_i * (Σw @ cov)_i / total_vol
    var_contrib_i = vol_contrib_i * corr_i_p

    Returns DataFrame with columns: code, avg_weight, vol_contrib, var_contrib
    """
    if codes is None:
        codes = list(weights_df.columns)
    weights_arr = np.array([weights_df[c].mean() if c in weights_df.columns else 0.0 for c in codes])
    cov_sub = cov[np.ix_([codes.index(c) for c in codes if c in codes], [codes.index(c) for c in codes if c in codes])]

    if weights_arr.sum() == 0:
        return pd.DataFrame(columns=["code", "avg_weight", "vol_contrib", "var_contrib"])

    port_vol = np.sqrt(weights_arr @ cov_sub @ weights_arr)
    if port_vol == 0:
        port_vol = 1e-12

    contrib = weights_arr * (cov_sub @ weights_arr) / port_vol

    rows = []
    for i, code in enumerate(codes):
        if i < len(contrib):
            var_c = contrib[i] * weights_arr[i]
            rows.append({
                "code": code,
                "avg_weight": float(weights_arr[i]),
                "vol_contrib": float(contrib[i]),
                "var_contrib": float(var_c),
            })

    return pd.DataFrame(rows).sort_values("var_contrib", ascending=False).reset_index(drop=True)


def marginal_contribution(
    weights_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    cov: np.ndarray,
    codes: list[str] | None = None,
) -> pd.DataFrame:
    """边际贡献 (dSharpe/dw_i).

    Returns DataFrame with columns: code, correlation, cov_i_p, marginal_sharpe
    """
    if codes is None:
        codes = list(weights_df.columns)
    weights_arr = np.array([weights_df[c].mean() if c in weights_df.columns else 0.0 for c in codes])

    if returns_df is None or returns_df.empty:
        return pd.DataFrame(columns=["code", "correlation", "cov_i_p", "marginal_sharpe"])

    rets = returns_df[codes].dropna(how="all") if all(c in returns_df.columns for c in codes) else returns_df.dropna(how="all")
    if rets.empty:
        return pd.DataFrame(columns=["code", "correlation", "cov_i_p", "marginal_sharpe"])

    port_ret = (rets * weights_arr).sum(axis=1)
    if port_ret.std() == 0:
        return pd.DataFrame(columns=["code", "correlation", "cov_i_p", "marginal_sharpe"])

    rows = []
    for i, code in enumerate(codes):
        if code not in rets.columns:
            continue
        corr = float(rets[code].corr(port_ret))
        cov_ip = float(rets[code].cov(port_ret))
        # 边际 Sharpe = corr_i_p / (sigma_p * sigma_i)
        sigma_p = port_ret.std()
        sigma_i = rets[code].std()
        if sigma_p > 0 and sigma_i > 0:
            marginal_sharpe = corr * (sigma_p / sigma_i) if sigma_i > 0 else 0.0
        else:
            marginal_sharpe = 0.0
        rows.append({
            "code": code,
            "correlation": corr,
            "cov_i_p": cov_ip,
            "marginal_sharpe": marginal_sharpe,
        })

    return pd.DataFrame(rows).sort_values("marginal_sharpe", ascending=False).reset_index(drop=True)


def period_contribution(
    weights_df: pd.DataFrame,
    nav_df: pd.DataFrame,
    periods: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """周期贡献.

    Returns DataFrame with columns: period, n_days, ann_return, max_drawdown,
                                     calmar, period_return, return_pct_of_total
    """
    periods = periods or DEFAULT_PERIODS

    # 总收益 (基准)
    total_return = 0.0
    nav_all = nav_df.dropna(how="all")
    if not nav_all.empty and len(nav_all) > 1:
        total_return = float((nav_all.sum(axis=1).iloc[-1] / nav_all.sum(axis=1).iloc[0]) - 1) / max(len(nav_all), 1) * len(nav_all)

    rows = []
    for name, start, end in periods:
        mask = (weights_df.index >= start) & (weights_df.index <= end)
        sub_w = weights_df[mask]
        sub_n = nav_df.loc[start:end] if start in nav_df.index else nav_df.iloc[0:0]

        if sub_w.empty or sub_n.empty or len(sub_n) < 2:
            rows.append({
                "period": name, "n_days": 0,
                "ann_return": 0.0, "max_drawdown": 0.0,
                "calmar": 0.0, "period_return": 0.0,
                "return_pct_of_total": 0.0,
            })
            continue

        # 用权重 × 价格 算组合 nav
        common_idx = sub_w.index.intersection(sub_n.index)
        if len(common_idx) < 2:
            rows.append({
                "period": name, "n_days": 0,
                "ann_return": 0.0, "max_drawdown": 0.0,
                "calmar": 0.0, "period_return": 0.0,
                "return_pct_of_total": 0.0,
            })
            continue
        sub_w_aligned = sub_w.loc[common_idx]
        sub_n_aligned = sub_n.loc[common_idx]

        rets = sub_n_aligned.pct_change().fillna(0.0)
        port_ret = (rets * sub_w_aligned).sum(axis=1).fillna(0.0)
        nav = (1 + port_ret).cumprod()

        n_days = len(nav)
        if nav.iloc[-1] > 0:
            period_ret = float(nav.iloc[-1] - 1)
        else:
            period_ret = 0.0
        ann_ret = float((1 + period_ret) ** (252 / max(n_days, 1)) - 1) if period_ret > -1 else -1.0

        cummax = nav.cummax()
        dd = (nav / cummax - 1)
        max_dd = float(dd.min()) if not dd.empty else 0.0

        calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0

        # 占总收益比例 (用 ann_ret 归一)
        rows.append({
            "period": name,
            "n_days": n_days,
            "ann_return": ann_ret,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "period_return": period_ret,
            "return_pct_of_total": period_ret,  # 简化, 不归一
        })

    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_PERIODS",
    "reconstruct_daily_weights",
    "etf_contribution",
    "category_contribution",
    "risk_contribution",
    "marginal_contribution",
    "period_contribution",
]
