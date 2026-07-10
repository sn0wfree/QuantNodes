"""v7.0 State Conditional Inverse Volatility — 风险平价 / inverse vol 权重.

[Stage 30.5] 5 Macro Dynamic 方案之五.

[核心算法]
    调仓日 d:
        1. 算每 ETF 过去 lookback 天的 vol
        2. weight = (1/vol) / Σ(1/vol)
        3. cap max_weight (默认 30%)
        4. 长期约束: long-only, sum=1

[业界对应] 风险平价 / Bridgewater All Weather (简化版, 无杠杆)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def compute_etf_vol(
    panel: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback: int = 252,
) -> pd.Series:
    """截至 as_of, 算 7 ETF 过去 lookback 天年化 vol."""
    pa = panel.loc[:as_of]
    rets = pa.pct_change().dropna()
    if lookback and lookback < len(rets):
        rets = rets.iloc[-lookback:]
    if len(rets) < 30:
        return pd.Series(0.20, index=panel.columns)
    return rets.std() * np.sqrt(252)


def compute_inverse_vol_weights(
    vol_series: pd.Series,
    max_weight: float = 0.30,
    etf_universe: Optional[list[str]] = None,
) -> dict[str, float]:
    """从 vol 序列导出 long-only inverse vol 权重, 带 cap.

    Args:
        vol_series: 7 ETF 年化 vol.
        max_weight: 单 ETF 权重上限.
        etf_universe: ETF 池.

    Returns:
        dict[etf_code] -> weight, sum=1.
    """
    if etf_universe is None:
        etf_universe = list(vol_series.index)
    n = len(etf_universe)
    fallback = {c: 1.0 / n for c in etf_universe}
    vol = vol_series.reindex(etf_universe)
    if vol.isna().all() or (vol <= 0).any():
        return fallback
    inv_vol = 1.0 / vol.clip(lower=0.01)
    if inv_vol.sum() < 1e-9:
        return fallback
    w = inv_vol / inv_vol.sum()
    w = w.clip(upper=max_weight)
    w = w / w.sum()
    return w.to_dict()


def run_iv_v7_backtest(
    panel: pd.DataFrame,
    tl_df: pd.DataFrame,
    lookback: int = 252,
    max_weight: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Walk-forward inverse vol 调仓.

    Args:
        panel: 收盘价面板.
        tl_df: HMM timeline (本策略不强依赖, 但保持签名一致).
        lookback: vol window.
        max_weight: 单 ETF 权重上限.

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
            vol = compute_etf_vol(panel, rebal_dates[i - 1], lookback)
            w = compute_inverse_vol_weights(vol, max_weight=max_weight, etf_universe=etf_universe)
            cur_state_series = tl_df["regime"].reindex([d], method="ffill")
            cur_state = cur_state_series.iloc[0] if len(cur_state_series) else "init"
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
