# coding=utf-8
"""v1 端到端回测: 月度调仓循环 (原始CICC复现, 无 VT, 无 Cost)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from ..common.universe import ETFPool
from .portfolio_v1 import (
    PortfolioState_v1,
    RotationConfig_v1,
    apply_stops_v1,
    equal_weights_v1,
    inverse_vol_weights_v1,
    select_and_weight_v1,
)


@dataclass
class BacktestConfig_v1:
    rotation: RotationConfig_v1 = field(default_factory=RotationConfig_v1)
    freq: str = "ME"
    init_value: float = 1.0


@dataclass
class RotationBacktestResult_v1:
    nav: pd.Series
    states: list = field(default_factory=list)
    rebalance_dates: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def run_rotation_backtest_v1(
    etf_nav: pd.DataFrame,
    pool: ETFPool,
    cfg: BacktestConfig_v1 | None = None,
) -> RotationBacktestResult_v1:
    """v1 纯动量轮动回测 (无债券, 无 VT, 无 Cost)."""
    cfg = cfg or BacktestConfig_v1()
    rot = cfg.rotation
    etf = etf_nav.dropna(how="all")
    etf_norm = etf / etf.iloc[0]
    dates = etf.index
    rebal_dates = pd.Series(dates).groupby(dates.to_period("M")).max().tolist()
    lookback = rot.lookback
    valid = [d for d in rebal_dates if dates.searchsorted(d) >= lookback]
    if not valid:
        raise ValueError(f"数据不足: 需要 {lookback} 天")
    rebal_dates = valid
    first_rebal = rebal_dates[0]
    first_pos = dates.searchsorted(first_rebal)
    sim_start = dates[max(0, first_pos - lookback)]
    dates = dates[dates >= sim_start]
    etf_norm = etf_norm.loc[dates]

    nav = np.ones(len(dates))
    prev_weights: dict[str, float] = {}
    states: list[PortfolioState_v1] = []
    actual: list[pd.Timestamp] = []

    for i, date in enumerate(dates):
        if date in rebal_dates:
            if prev_weights:
                state = apply_stops_v1(etf_norm, pool, rot, prev_weights, date)
            else:
                state = select_and_weight_v1(etf_norm, pool, rot, date)
            if not state.weights:
                state.weights = equal_weights_v1(list(etf_norm.columns))
            total = sum(state.weights.values())
            if total > 0:
                state.weights = {k: v / total for k, v in state.weights.items()}
            prev_weights = state.weights
            states.append(state)
            actual.append(date)
            if i > 0:
                nav[i] = nav[i - 1]
            else:
                nav[i] = 1.0
        else:
            if i > 0 and prev_weights:
                daily_ret = 0.0
                for code, w in prev_weights.items():
                    if code in etf_norm.columns:
                        col = etf_norm[code]
                        a = col.iloc[i] if hasattr(col, 'iloc') else col
                        b = col.iloc[i - 1] if hasattr(col, 'iloc') else col
                        if isinstance(a, pd.Series): a = a.iloc[0]
                        if isinstance(b, pd.Series): b = b.iloc[0]
                        if not pd.isna(a) and not pd.isna(b) and b != 0:
                            daily_ret += w * (a / b - 1)
                nav[i] = nav[i - 1] * (1 + daily_ret)
            else:
                nav[i] = 1.0 if i == 0 else nav[i - 1]

    nav_series = pd.Series(nav, index=dates, name="rotation_v1")
    return RotationBacktestResult_v1(
        nav=nav_series, states=states, rebalance_dates=actual,
    )


def run_equal_weight_baseline_v1(
    etf_nav: pd.DataFrame,
    pool: ETFPool,
    cfg: BacktestConfig_v1 | None = None,
) -> RotationBacktestResult_v1:
    """v1 等权 baseline."""
    from ..common.fi_plus import performance_metrics as pm
    cfg = cfg or BacktestConfig_v1()
    rot = cfg.rotation
    etf = etf_nav.dropna(how="all")
    etf_norm = etf / etf.iloc[0]
    dates = etf.index
    rebal_dates = pd.Series(dates).groupby(dates.to_period("M")).max().tolist()
    lookback = rot.lookback
    valid = [d for d in rebal_dates if dates.searchsorted(d) >= lookback]
    if not valid:
        raise ValueError(f"数据不足: 需要 {lookback} 天")
    rebal_dates = valid
    first_rebal = rebal_dates[0]
    first_pos = dates.searchsorted(first_rebal)
    sim_start = dates[max(0, first_pos - lookback)]
    dates = dates[dates >= sim_start]
    etf_norm = etf_norm.loc[dates]

    nav = np.ones(len(dates))
    prev_weights: dict[str, float] = {}
    states: list = []
    actual: list[pd.Timestamp] = []

    for i, date in enumerate(dates):
        if date in rebal_dates:
            top_codes = list(etf_norm.columns[:rot.top_n])
            weights = {c: 1.0 / len(top_codes) for c in top_codes}
            state = PortfolioState_v1(
                date=date, ranked=top_codes, chosen=top_codes, weights=weights,
            )
            prev_weights = state.weights
            states.append(state)
            actual.append(date)
            if i > 0:
                nav[i] = nav[i - 1]
            else:
                nav[i] = 1.0
        else:
            if i > 0 and prev_weights:
                daily_ret = 0.0
                for code, w in prev_weights.items():
                    if code in etf_norm.columns:
                        col = etf_norm[code]
                        a = col.iloc[i] if hasattr(col, 'iloc') else col
                        b = col.iloc[i - 1] if hasattr(col, 'iloc') else col
                        if isinstance(a, pd.Series): a = a.iloc[0]
                        if isinstance(b, pd.Series): b = b.iloc[0]
                        if not pd.isna(a) and not pd.isna(b) and b != 0:
                            daily_ret += w * (a / b - 1)
                nav[i] = nav[i - 1] * (1 + daily_ret)
            else:
                nav[i] = 1.0 if i == 0 else nav[i - 1]

    nav_series = pd.Series(nav, index=dates, name="equal_v1")
    return RotationBacktestResult_v1(nav=nav_series, states=states, rebalance_dates=actual)


__all__ = [
    "BacktestConfig_v1",
    "RotationBacktestResult_v1",
    "run_rotation_backtest_v1",
    "run_equal_weight_baseline_v1",
]