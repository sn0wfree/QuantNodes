"""v7.0 State Conditional Momentum — state 内 ETF 动量排名.

[Stage 30.5] 5 Macro Dynamic 方案之四.

[核心算法]
    调仓日 d:
        1. 每 ETF 算过去 lookback 天动量 (e.g., 63d = 3 月)
        2. 当前 state 下的所有历史日: 取 state 内 ETF 平均动量 (state-conditional)
        3. expected = 0.5 × current_momentum + 0.5 × state_conditional_momentum
        4. top K 等权

[业界对应] 海通证券 "近一季风格动量差" + 宏观状态过滤
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_etf_momentum(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int = 63,
) -> pd.Series:
    """截至 as_of, 算 7 ETF 过去 lookback 天动量 (累计收益).

    Args:
        panel: 收盘价面板.
        as_of: cutoff.
        lookback: 回看天数 (默认 63 = 3 月).

    Returns:
        Series: 7 ETF momentum.
    """
    pa = panel.loc[:as_of]
    if len(pa) < lookback + 1:
        return pd.Series(0.0, index=panel.columns)
    end = pa.iloc[-1]
    start = pa.iloc[-lookback - 1]
    return (end / start - 1).rename("momentum")


def compute_state_conditional_momentum(
    panel: pd.DataFrame,
    tl_df: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int = 63,
) -> pd.Series:
    """state-conditional momentum: 当前 state 历史日的 ETF 平均动量.

    对每个 ETF: 在 state 历史出现过的所有日, 算动量, 取均值.
    """
    pa = panel.loc[:as_of]
    ta = tl_df.loc[:as_of, "regime"]
    common_idx = pa.index.intersection(ta.index)
    pa = pa.loc[common_idx]
    ta = ta.loc[common_idx]

    state_dates = ta.index[ta == ta.iloc[-1]]
    if len(state_dates) < 5:
        return pd.Series(0.0, index=panel.columns)

    mom_matrix = []
    for d in state_dates:
        idx = pa.index.get_loc(d)
        if idx >= lookback:
            start = pa.iloc[idx - lookback]
            end = pa.iloc[idx]
            mom = (end / start - 1)
            mom_matrix.append(mom)
    if not mom_matrix:
        return pd.Series(0.0, index=panel.columns)
    return pd.DataFrame(mom_matrix).mean()


def run_momentum_v7_backtest(
    panel: pd.DataFrame,
    tl_df: pd.DataFrame,
    lookback: int = 63,
    k: int = 5,
    blend_state: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Walk-forward state-conditional momentum Top-K 调仓.

    Args:
        panel: 收盘价面板.
        tl_df: HMM timeline.
        lookback: 动量回看天数.
        k: top K 数量.
        blend_state: state-conditional 与 current momentum 混合权重 (0-1).

    Returns:
        (nav_df, weights_history, metrics)
    """
    from .dynamic_allocation import _compute_metrics

    etf_universe = list(panel.columns)
    rebal_dates = []
    for d in panel.resample("BME").last().index:
        if d >= tl_df.index[0] and d in panel.index:
            rebal_dates.append(d)

    nav_path = []
    weights_log = []
    for i, d in enumerate(rebal_dates):
        if i == 0:
            w = {c: 1.0 / len(etf_universe) for c in etf_universe}
            cur_state = "init"
        else:
            cur_mom = compute_etf_momentum(panel, rebal_dates[i - 1], lookback)
            state_mom = compute_state_conditional_momentum(panel, tl_df, rebal_dates[i - 1], lookback)
            combined = blend_state * state_mom.reindex(cur_mom.index).fillna(0) + \
                       (1 - blend_state) * cur_mom
            cur_state_series = tl_df["regime"].reindex([d], method="ffill")
            cur_state = cur_state_series.iloc[0] if len(cur_state_series) else "init"
            combined = combined.reindex(etf_universe).dropna()
            kk = min(k, len(combined))
            top = combined.nlargest(kk).index.tolist()
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
