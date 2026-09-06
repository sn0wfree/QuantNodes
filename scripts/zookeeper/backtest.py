# coding: utf-8
"""Zookeeper 因子月度再平衡截面回测 — P4 实装.

流程 (论文 §4):
  1. 月末 t: 用 L-En 预测 r̂ (t+1 月收益)
  2. 横截面排序: 前 25% Q4 / 后 25% Q1
  3. t 收盘等权做多 Q4 / 做空 Q1, 完全保证金
  4. t+1 末重复, 月度再平衡
  5. **严格扣 cost_model.json** (知乎陷阱 §7.1)

⚠️ 换手监控: 月度单边换手 > 100% 报警 (沿 config.TURN_WARN).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import (QUARTILE_SIZE, TURN_WARN)

LOG = logging.getLogger(__name__)


def cross_section_backtest(predictions: pd.DataFrame,
                           returns: pd.DataFrame,
                           cost_bps: float = 0.0015) -> dict:
    """Zookeeper 因子月度截面回测.

    Args:
        predictions: DataFrame, index=MonthEnd (预测下期), columns=product.
                    论文口径: t 末预测 t+1, t 期排序→t+1 持仓.
        returns: DataFrame, index=MonthEnd (实现收益), columns=product.
        cost_bps: 单边成本 (基础 15bps + 换月滑点).

    Returns:
        dict: nav / monthly_returns / turnover / metrics
    """
    LOG.info(f"截面回测: predictions {predictions.shape}, cost_bps={cost_bps:.0%}")
    common = predictions.index.intersection(returns.index)
    pred = predictions.loc[common]
    ret = returns.loc[common]
    n_products = len(pred.columns)
    q_size = max(1, int(n_products * QUARTILE_SIZE))

    long_ret, short_ret, turnovers = [], [], []
    prev_long, prev_short = set(), set()
    for t in common:
        if t not in ret.index:
            continue
        row = pred.loc[t].dropna()
        if len(row) < 2 * q_size:
            long_ret.append(np.nan)
            short_ret.append(np.nan)
            turnovers.append(np.nan)
            continue
        # 排序: top Q4 / bottom Q1
        sorted_row = row.sort_values(ascending=False)
        long_set = set(sorted_row.head(q_size).index)
        short_set = set(sorted_row.tail(q_size).index)

        # 多头等权收益
        long_p = ret.loc[t, list(long_set)].mean()
        short_p = ret.loc[t, list(short_set)].mean()
        # 多空等权组合 (扣成本前)
        long_ret.append(long_p)
        short_ret.append(short_p)

        # 换手: 单边变动比例
        all_set = long_set | short_set
        prev_all = prev_long | prev_short
        new_all = long_set | short_set
        diff = (new_all ^ prev_all)
        turnover = len(diff) / len(all_set) if all_set else 0.0
        turnovers.append(turnover)
        prev_long, prev_short = long_set, short_set

    long_series = pd.Series(long_ret, index=common[:len(long_ret)])
    short_series = pd.Series(short_ret, index=long_series.index)
    turnover_series = pd.Series(turnovers, index=long_series.index)
    # 多空组合 (扣成本后)
    raw_ls = long_series - short_series
    cost_drag = turnover_series * 2 * cost_bps  # 双边: 多头+空头
    ls_net = raw_ls - cost_drag
    nav = (1 + ls_net).cumprod()
    nav.iloc[0] = 1.0 if not nav.empty else 1.0

    metrics = compute_metrics(ls_net, turnover_series)
    check_turnover_alert(turnover_series)
    LOG.info(f"  LS 净 Sharpe={metrics['sharpe']:.2f}, 年化={metrics['annual_ret']:.2%}, "
             f"换手={turnover_series.mean():.1%}, DD={metrics['max_dd']:.2%}")
    return {
        "nav": nav,
        "long_ret": long_series,
        "short_ret": short_series,
        "ls_raw": raw_ls,
        "ls_net": ls_net,
        "turnover": turnover_series,
        "metrics": metrics,
    }


def apply_costs(nav: pd.Series, turnover: pd.Series,
                cost_bps: float = 0.0015) -> pd.Series:
    """成本扣减 (沿用 §22 cost_model.json 校准).

    已纳入 cross_section_backtest, 此处为独立工具函数.
    """
    return nav / (1 + turnover * cost_bps)


def compute_metrics(monthly_ret: pd.Series, turnover: pd.Series) -> dict:
    """评估指标: 年化 / Sharpe / t-stat / 最大回撤 / 月度换手."""
    r = monthly_ret.dropna()
    if r.empty:
        return {"annual_ret": np.nan, "sharpe": np.nan, "t_stat": np.nan,
                "max_dd": np.nan, "turnover_mean": np.nan, "n_months": 0}
    annual_ret = r.mean() * 12
    sharpe = r.mean() / r.std() * np.sqrt(12) if r.std() > 0 else np.nan
    t_stat = r.mean() / r.std() * np.sqrt(len(r)) if r.std() > 0 else np.nan
    # 最大回撤
    nav = (1 + r).cumprod()
    nav.iloc[0] = 1.0
    peak = nav.cummax()
    dd = (nav - peak) / peak
    max_dd = dd.min()
    return {
        "annual_ret": float(annual_ret),
        "sharpe": float(sharpe),
        "t_stat": float(t_stat),
        "max_dd": float(max_dd),
        "turnover_mean": float(turnover.mean()) if not turnover.empty else np.nan,
        "n_months": int(len(r)),
    }


def check_turnover_alert(turnover: pd.Series, threshold: float = TURN_WARN) -> dict:
    """换手监控 (知乎陷阱 §7.1 报警)."""
    t = turnover.dropna()
    if t.empty:
        return {"alert": False, "msg": "无数据"}
    mean_to = t.mean()
    if mean_to > threshold:
        LOG.warning(f"⚠️ 换手过高: 月度单边 {mean_to:.1%} > {threshold:.0%} "
                    f"(知乎陷阱 §7.1, 交易成本致命)")
        return {"alert": True, "mean_turnover": float(mean_to), "msg": "换手过高"}
    return {"alert": False, "mean_turnover": float(mean_to)}
