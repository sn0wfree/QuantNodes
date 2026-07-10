"""v7.0 Top-K Simple Dynamic — 5 状态驱动的 top-K ETF 排名配置.

[Stage 30.5] 5 Macro Dynamic 方案之一.

[核心算法]
    1. 截至 as_of, 计算每月最后交易日的 forward 21d 收益
    2. 按 state × ETF 聚合, 得到均值收益表 (5 state × 7 ETF)
    3. 当前 state 下, 取 top K ETF 等权

[PIT 关键]
    - state-conditional 均值只用 ≤ as_of 的历史月
    - ETF 历史 < 60 天则剔除 (主要 512760 2019 前)
    - state 历史 < 3 月则 fallback 到等权

[业界对应] 中泰证券 / 中银证券 "Top-K 排名" 配置
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_state_conditional_means(
    panel: pd.DataFrame,
    tl_df: pd.DataFrame,
    as_of: pd.Timestamp,
    forward_days: int = 21,
) -> pd.DataFrame:
    """截至 as_of, 计算 state × ETF 的 forward N 日均值收益.

    Args:
        panel: ETF 收盘价面板 (index=date, columns=etf_codes).
        tl_df: HMM timeline DataFrame (index=date, columns=['regime', ...]).
        as_of: cutoff 日期, 严格 ≤ as_of.
        forward_days: forward window, 默认 21 trading days (1 月).

    Returns:
        DataFrame: index=state (str), columns=etf_codes, values=mean forward return.
    """
    pa = panel.loc[:as_of]
    ta = tl_df.loc[:as_of, "regime"]

    monthly = pa.resample("BME").last().index
    monthly = [d for d in monthly if d in pa.index]
    sa = ta.reindex(monthly, method="ffill")

    rows = []
    for d in monthly:
        idx = pa.index.get_loc(d)
        if idx + forward_days < len(pa):
            ret = (pa.iloc[idx + forward_days] / pa.iloc[idx] - 1).to_dict()
            ret["state"] = sa.loc[d]
            ret["date"] = d
            rows.append(ret)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("date")
    return df


def compute_dynamic_topk_weights(
    state_means: pd.DataFrame,
    cur_state: str,
    k: int = 5,
    min_samples: int = 3,
    etf_universe: Optional[list[str]] = None,
) -> dict[str, float]:
    """cur_state 下, 取 top K ETF 等权.

    Args:
        state_means: state × ETF 均值收益 DataFrame.
        cur_state: 当前 HMM state (e.g., 'recovery').
        k: top K 数量, 默认 5.
        min_samples: state 历史最少月数, < 则 fallback 等权.
        etf_universe: 候选 ETF 池, 默认 state_means.columns.

    Returns:
        dict[etf_code] -> weight, sum=1.
    """
    if etf_universe is None:
        etf_universe = list(state_means.columns)
    n = len(etf_universe)
    fallback = {c: 1.0 / n for c in etf_universe}

    if state_means.empty or "state" not in state_means.columns:
        return fallback

    s_data = state_means[state_means["state"] == cur_state]
    if len(s_data) < min_samples:
        return fallback

    means = s_data.drop(columns="state").mean() * 100
    means = means.reindex(etf_universe).dropna()
    if means.empty:
        return fallback

    k = min(k, len(means))
    top = means.nlargest(k).index.tolist()
    w = 1.0 / k
    return {c: (w if c in top else 0.0) for c in etf_universe}


def run_topk_v7_backtest(
    panel: pd.DataFrame,
    tl_df: pd.DataFrame,
    k: int = 5,
    start_date: Optional[pd.Timestamp] = None,
    forward_days: int = 21,
    min_samples: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Walk-forward 月度调仓 Top-K 动态策略.

    Args:
        panel: ETF 收盘价面板.
        tl_df: HMM timeline.
        k: top K 数量.
        start_date: 回测起始, 默认 tl_df 第一天.
        forward_days: state-conditional 收益的 forward window.
        min_samples: state 最少月数.

    Returns:
        (nav_df, weights_history, metrics)
        - nav_df: index=date, columns=['nav', 'nav_cum', 'daily_ret']
        - weights_history: index=date, columns=etf_codes + ['state']
        - metrics: dict with ann, vol, sharpe, dd, calmar
    """
    etf_universe = list(panel.columns)
    if start_date is None:
        start_date = tl_df.index[0]

    rebal_dates = []
    for d in panel.resample("BME").last().index:
        if d >= start_date and d in panel.index:
            rebal_dates.append(d)

    nav_path = []
    weights_log = []
    for i, d in enumerate(rebal_dates):
        if i == 0:
            w = {c: 1.0 / len(etf_universe) for c in etf_universe}
            cur_state = "init"
        else:
            fwd = compute_state_conditional_means(panel, tl_df, rebal_dates[i - 1], forward_days)
            cur_state_series = tl_df["regime"].reindex([d], method="ffill")
            cur_state = cur_state_series.iloc[0] if len(cur_state_series) else "init"
            w = compute_dynamic_topk_weights(
                fwd, cur_state, k=k, min_samples=min_samples, etf_universe=etf_universe
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


def _compute_metrics(nav: pd.Series) -> dict:
    s = nav.dropna()
    if len(s) < 2:
        return {"ann": 0.0, "vol": 0.0, "sharpe": 0.0, "dd": 0.0, "calmar": 0.0}
    n_days = (s.index[-1] - s.index[0]).days
    if n_days < 1:
        return {"ann": 0.0, "vol": 0.0, "sharpe": 0.0, "dd": 0.0, "calmar": 0.0}
    total_ret = s.iloc[-1] / s.iloc[0] - 1
    ann = (1 + total_ret) ** (365.25 / n_days) - 1
    monthly_ret = s.pct_change().dropna()
    ann_vol = monthly_ret.std() * np.sqrt(12) if len(monthly_ret) > 1 else 0.0
    dd = (s / s.cummax() - 1).min()
    return {
        "ann": ann,
        "vol": ann_vol,
        "sharpe": ann / ann_vol if ann_vol > 0 else 0.0,
        "dd": dd,
        "calmar": ann / abs(dd) if dd != 0 else 0.0,
    }
