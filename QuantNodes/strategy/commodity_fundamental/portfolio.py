# coding=utf-8
"""QuantNodes/strategy/commodity_fundamental/portfolio.py

§5.1 组合构建:
  1. 未截断仓位: ŵ_i = s_i * min(σ*/σ_i, 1), σ_i = EWMA 半衰期26周
  2. 现金管理: S = Σ ŵ_i ≤ 1 直接配置; >1 按比例压缩
  3. 调仓: 偏离>5% 或距上次调仓 ≥8 周
  4. 成本: 单边 0.05% × |Δw| 之和
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PortfolioConfig:
    target_vol_annual: float = 0.15
    ewma_halflife_weeks: int = 26
    rebal_threshold_pp: float = 0.05
    rebal_min_gap_weeks: int = 8
    cost_bps_per_leg: float = 5.0     # 0.05%
    cash_yield_annual: float = 0.01


def ewma_vol(weekly_ret: pd.Series, halflife_weeks: int = 26) -> pd.Series:
    """EWMA 年化波动率."""
    alpha = 1 - np.exp(-np.log(2) / halflife_weeks)
    vol2 = weekly_ret.pow(2).ewm(alpha=alpha, adjust=False).mean()
    return (vol2 ** 0.5) * np.sqrt(52)


def build_target_weights(
    signals: dict[str, pd.Series],
    weekly_returns: pd.DataFrame,
    cfg: PortfolioConfig = PortfolioConfig(),
) -> pd.DataFrame:
    """对每个品种计算目标权重序列."""
    w_targets = {}
    for symbol, pos in signals.items():
        ret = weekly_returns[symbol]
        vol = ewma_vol(ret, cfg.ewma_halflife_weeks)
        scale = (cfg.target_vol_annual / vol).clip(upper=1.0)
        w_targets[symbol] = pos * scale
    W = pd.concat(w_targets, axis=1)
    W.columns = list(signals.keys())

    S = W.sum(axis=1)
    W_final = W.copy()
    overflow = S > 1.0
    W_final.loc[overflow] = W.loc[overflow].div(S.loc[overflow], axis=0)
    return W_final


def simulate_portfolio(
    weekly_returns: pd.DataFrame,
    signals: dict[str, pd.Series],
    cfg: PortfolioConfig = PortfolioConfig(),
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """模拟组合: 返回 NAV、weekly_ret、actual_weights.

    调仓触发: |w_target - w_actual| > 5pp 或距上次调仓 ≥ 8 周
    成本: 单边 5bps × Σ|Δw_i|
    """
    target_w = build_target_weights(signals, weekly_returns, cfg)
    target_w = target_w.dropna(how="all")

    common = target_w.index.intersection(weekly_returns.index)
    target_w = target_w.loc[common].fillna(0.0)
    weekly_returns = weekly_returns.loc[common].fillna(0.0)

    symbols = list(target_w.columns)
    actual_w = pd.DataFrame(0.0, index=target_w.index, columns=symbols)
    nav = pd.Series(1.0, index=target_w.index)
    port_ret = pd.Series(0.0, index=target_w.index)
    cash_w = pd.Series(0.0, index=target_w.index)

    last_rebal_idx = -10_000
    for i, t in enumerate(target_w.index):
        if i == 0:
            actual_w.iloc[i] = target_w.iloc[i]
            cash_w.iloc[i] = 1.0 - actual_w.iloc[i].sum()
            nav.iloc[i] = 1.0
        else:
            r = weekly_returns.iloc[i].reindex(symbols).fillna(0.0).values
            w_prev = actual_w.iloc[i - 1].values
            cash_prev = cash_w.iloc[i - 1]
            gross_ret = float(np.dot(w_prev, r)) + cash_prev * (cfg.cash_yield_annual / 52)
            nav.iloc[i] = nav.iloc[i - 1] * (1 + gross_ret)

            w_target = target_w.iloc[i].values
            delta = np.abs(w_target - w_prev)
            big_diff = bool(np.any(delta > cfg.rebal_threshold_pp))
            weeks_passed = i - last_rebal_idx
            if big_diff or weeks_passed >= cfg.rebal_min_gap_weeks:
                cost = (cfg.cost_bps_per_leg / 10000.0) * delta.sum()
                nav.iloc[i] *= (1 - cost)
                actual_w.iloc[i] = w_target
                cash_w.iloc[i] = 1.0 - w_target.sum()
                last_rebal_idx = i
            else:
                actual_w.iloc[i] = w_prev
                cash_w.iloc[i] = cash_prev * (1 + cfg.cash_yield_annual / 52) / (1 + gross_ret) if (1 + gross_ret) > 0 else cash_prev

        port_ret.iloc[i] = nav.iloc[i] / nav.iloc[i - 1] - 1 if i > 0 else 0.0

    return nav, port_ret, actual_w


def compute_metrics(nav: pd.Series, ret: pd.Series) -> dict[str, float]:
    """策略指标: 年化收益/波动/Sharpe/最大回撤/胜率."""
    nav = nav.dropna()
    ret = ret.dropna()
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_weeks = len(ret)
    n_years = n_weeks / 52
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    ann_vol = ret.std() * np.sqrt(52)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = nav.cummax()
    dd = nav / cummax - 1
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    win_rate = (ret > 0).mean()
    return {
        "n_weeks": n_weeks,
        "total_ret": total_ret,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
    }


__all__ = [
    "PortfolioConfig",
    "ewma_vol",
    "build_target_weights",
    "simulate_portfolio",
    "compute_metrics",
]