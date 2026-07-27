# coding=utf-8
"""R&D 共享工具: metrics, IC, beta 诊断, 日期对齐.

消除 24+ 个脚本中的重复代码. 所有函数都有完整类型注解和 docstring.

用法:
    from QuantNodes.strategy.momentum_etf_rotation.v7.rd_utils import (
        compute_weekly_metrics,
        compute_daily_metrics,
        compute_beta_stability,
        compute_tv_norm,
        compute_cross_sectional_ic,
        compute_ic_summary,
        align_daily_to_weekly,
    )
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


# ============================================================
# 1. Performance Metrics (替代 24 个文件中的重复定义)
# ============================================================
def compute_weekly_metrics(nav: pd.Series, label: str = "") -> dict:
    """周频 NAV → 年化收益/波动/夏普/最大回撤/卡尔玛.

    Parameters:
        nav: 周频净值序列
        label: 标签 (可选)

    Returns:
        dict: ann_return, ann_vol, sharpe, max_dd, calmar, n_weeks
    """
    if nav.empty or len(nav) < 2:
        return dict(label=label, ann_return=0, ann_vol=0, sharpe=0, max_dd=0, calmar=0, n_weeks=0)
    ret = nav.pct_change().dropna()
    ann_ret = float(ret.mean() * 52)
    ann_vol = float(ret.std() * np.sqrt(52))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    peak = nav.cummax()
    dd = (nav - peak) / peak
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if abs(max_dd) > 0 else 0.0
    return dict(label=label, ann_return=ann_ret, ann_vol=ann_vol,
                sharpe=sharpe, max_dd=max_dd, calmar=calmar, n_weeks=len(nav))


def compute_daily_metrics(nav_daily: pd.Series) -> dict:
    """日频 NAV → 年化收益/波动/夏普/最大回撤/最大回撤持续天数.

    Parameters:
        nav_daily: 日频净值序列

    Returns:
        dict: ann_return, ann_vol, sharpe, max_dd, max_dd_duration_days
    """
    if nav_daily.empty or len(nav_daily) < 2:
        return dict(ann_return=0, ann_vol=0, sharpe=0, max_dd=0, max_dd_duration_days=0)
    rets = nav_daily.pct_change().dropna()
    n_years = len(rets) / 252
    total_ret = nav_daily.iloc[-1] / nav_daily.iloc[0] - 1
    ann_ret = float((1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1)
    ann_vol = float(rets.std() * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    dd = nav_daily / nav_daily.cummax() - 1
    max_dd = float(dd.min())
    underwater = dd < -1e-6
    if underwater.any():
        groups = (~underwater).cumsum()
        dd_durations = underwater.groupby(groups).sum()
        max_dd_duration = int(dd_durations.max())
    else:
        max_dd_duration = 0
    return dict(ann_return=round(ann_ret, 4), ann_vol=round(ann_vol, 4),
                sharpe=round(sharpe, 4), max_dd=round(max_dd, 4),
                max_dd_duration_days=max_dd_duration)


# ============================================================
# 2. Beta 诊断 (替代 4+2 个文件中的重复定义)
# ============================================================
def compute_beta_stability(beta: pd.DataFrame) -> float:
    """Beta 稳定性: 相邻周 beta 变化的 RMS.

    Parameters:
        beta: (T, K) beta DataFrame

    Returns:
        float: RMS of beta changes
    """
    diff = beta.diff().iloc[1:]
    return float(np.sqrt((diff ** 2).sum(axis=1)).mean())


def compute_tv_norm(beta_arr: np.ndarray) -> float:
    """Beta 总变差: |beta_{t+1} - beta_t| 的总和.

    Parameters:
        beta_arr: (T, K) beta 数组

    Returns:
        float: total variation norm
    """
    return float(np.sum(np.abs(np.diff(beta_arr, axis=0))))


# ============================================================
# 3. IC 计算 (替代 6+ 个文件中的重复定义)
# ============================================================
def compute_cross_sectional_ic(
    X_panel: np.ndarray,
    Y: pd.DataFrame,
    factor_idx: int,
    min_obs: int = 10,
    start_t: int = 52,
) -> list[float]:
    """单因子的截面 Spearman IC 时间序列.

    Parameters:
        X_panel: (T, N, K) 因子面板
        Y: (T, N) 周频收益 DataFrame
        factor_idx: 因子索引
        min_obs: 最小有效资产数
        start_t: 起始时间步

    Returns:
        list[float]: IC 值列表
    """
    T = len(Y)
    Y_shifted = Y.shift(-1).iloc[:-1].values
    X_shifted = X_panel[:-1]

    ic_list = []
    for t in range(start_t, T - 1):
        x_t = X_shifted[t, :, factor_idx]
        y_t = Y_shifted[t]
        valid = ~np.isnan(x_t) & ~np.isnan(y_t)
        if valid.sum() > min_obs:
            corr, _ = spearmanr(x_t[valid], y_t[valid])
            ic_list.append(corr)
    return ic_list


def compute_ic_summary(ic_list: list[float]) -> dict:
    """IC 统计摘要: mean, std, ICIR, 正IC占比.

    Parameters:
        ic_list: IC 值列表

    Returns:
        dict: ic_mean, ic_std, icir, pct_positive, n_obs
    """
    if not ic_list:
        return dict(ic_mean=0, ic_std=0, icir=0, pct_positive=0, n_obs=0)
    arr = np.array(ic_list)
    ic_mean = float(np.mean(arr))
    ic_std = float(np.std(arr))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    pct_pos = float(np.mean(arr > 0))
    return dict(ic_mean=ic_mean, ic_std=ic_std, icir=icir, pct_positive=pct_pos, n_obs=len(ic_list))


def compute_time_series_ic(
    factor_ts: np.ndarray,
    market_ts: np.ndarray,
) -> tuple[float, float]:
    """时序因子 IC: Pearson 相关 + p-value.

    Parameters:
        factor_ts: (T,) 因子时间序列
        market_ts: (T,) 市场收益时间序列

    Returns:
        (corr, pvalue)
    """
    valid = ~np.isnan(factor_ts) & ~np.isnan(market_ts)
    if valid.sum() < 10:
        return 0.0, 1.0
    corr, pval = pearsonr(factor_ts[valid], market_ts[valid])
    return float(corr), float(pval)


def print_ic_table(
    factor_names: list[str],
    X_panel: np.ndarray,
    Y: pd.DataFrame,
    time_series_factors: set[str] | None = None,
    start_t: int = 52,
) -> None:
    """打印完整的 IC 表格 (截面 + 时序).

    Parameters:
        factor_names: 因子名称列表
        X_panel: (T, N, K) 因子面板
        Y: (T, N) 周频收益 DataFrame
        time_series_factors: 时序因子名称集合 (截面 IC 无意义)
        start_t: 起始时间步
    """
    if time_series_factors is None:
        time_series_factors = set()

    T = len(Y)
    Y_shifted = Y.shift(-1).iloc[:-1].values
    X_shifted = X_panel[:-1]
    market_ret = Y.shift(-1).iloc[:-1].mean(axis=1).values

    # 截面因子
    cs_names = [f for f in factor_names if f not in time_series_factors]
    if cs_names:
        print("\n  截面因子 IC:")
        print(f"  {'因子':<30} {'IC_mean':>10} {'IC_std':>10} {'ICIR':>10} {'pct_pos':>10}")
        print(f"  {'-'*70}")
        for k, fname in enumerate(factor_names):
            if fname in time_series_factors:
                continue
            ic_list = compute_cross_sectional_ic(X_panel, Y, k, start_t=start_t)
            summary = compute_ic_summary(ic_list)
            print(f"  {fname:<30} {summary['ic_mean']:>+10.4f} {summary['ic_std']:>10.4f} "
                  f"{summary['icir']:>+10.3f} {summary['pct_positive']:>9.1%}")

    # 时序因子
    ts_names = [f for f in factor_names if f in time_series_factors]
    if ts_names:
        print("\n  时序因子 IC:")
        print(f"  {'因子':<30} {'IC(Pearson)':>14} {'p-value':>10}")
        print(f"  {'-'*54}")
        for k, fname in enumerate(factor_names):
            if fname not in time_series_factors:
                continue
            factor_ts = X_shifted[start_t:T-1, 0, k]
            mkt_ts = market_ret[start_t:T-1]
            corr, pval = compute_time_series_ic(factor_ts, mkt_ts)
            print(f"  {fname:<30} {corr:>+14.4f} {pval:>10.4f}")


# ============================================================
# 4. 日期对齐 (替代 3 个文件中的重复定义)
# ============================================================
def align_daily_to_weekly(
    daily_data: pd.DataFrame | pd.Series,
    weekly_dates: pd.DatetimeIndex,
    tolerance_days: int = 7,
) -> np.ndarray:
    """日频数据对齐到周频日期 (最近匹配).

    Parameters:
        daily_data: 日频 DataFrame (T_daily, N) 或 Series (T_daily,)
        weekly_dates: 周频日期索引
        tolerance_days: 最大容差天数

    Returns:
        np.ndarray: (T_weekly, N) 或 (T_weekly,)
    """
    is_series = isinstance(daily_data, pd.Series)
    if is_series:
        daily_data = daily_data.to_frame("value")

    T_w = len(weekly_dates)
    N = len(daily_data.columns)
    result = np.full((T_w, N), np.nan)

    for i, target_date in enumerate(weekly_dates):
        diffs = abs(daily_data.index - target_date)
        if len(diffs) > 0:
            closest_idx = diffs.argmin()
            if diffs[closest_idx].days <= tolerance_days:
                result[i] = daily_data.iloc[closest_idx].values

    if is_series:
        return result[:, 0]
    return result


def align_daily_panel_to_weekly(
    daily_panel: np.ndarray,
    daily_dates: pd.DatetimeIndex,
    weekly_dates: pd.DatetimeIndex,
) -> np.ndarray:
    """日频因子面板对齐到周频日期.

    Parameters:
        daily_panel: (T_daily, N, K) 因子面板
        daily_dates: 日频日期索引
        weekly_dates: 周频日期索引

    Returns:
        np.ndarray: (T_weekly, N, K)
    """
    T_w = len(weekly_dates)
    N, K = daily_panel.shape[1], daily_panel.shape[2]
    result = np.full((T_w, N, K), np.nan)

    for i, w_date in enumerate(weekly_dates):
        mask = daily_dates <= w_date
        if mask.any():
            last_day_idx = np.where(mask)[0][-1]
            if last_day_idx < daily_panel.shape[0]:
                result[i] = daily_panel[last_day_idx]

    return result
