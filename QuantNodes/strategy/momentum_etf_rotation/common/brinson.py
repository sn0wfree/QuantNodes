# coding=utf-8
"""Brinson 归因模型.

公式 (从 archive/quantnodes_deprecated/brinson.py 复现):
    allocation  = (wp - wb) @ rb    # 择时/配置贡献
    selection  = wb @ (rp - rb)      # 选股贡献
    interaction = (wp - wb) @ (rp - rb)  # 交互贡献
    active      = allocation + selection + interaction

参考 reports/brinson.json.
"""
from __future__ import annotations


import pandas as pd

from .universe import ETFPool


CATEGORIES: list[str] = [
    "a_broad", "a_sector", "hk", "commodity", "overseas"
]


def _aligned_returns(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """从权重 + 价格 算每日组合收益."""
    common = weights.index.intersection(prices.index)
    if len(common) < 2:
        return pd.DataFrame()
    w = weights.loc[common]
    p = prices.loc[common]
    rets = p.pct_change().fillna(0.0)
    return (rets * w).sum(axis=1)


def _categorical_equal_bench(
    returns: pd.DataFrame,
    pool: ETFPool,
    bench_codes: list[str] | None = None,
) -> pd.DataFrame:
    """等权基准: 同一类别内等权."""
    if bench_codes is None:
        bench_codes = [m.code for m in pool.members]
    bench_codes = [c for c in bench_codes if c in returns.columns]
    return returns[bench_codes].mean(axis=1)


def brinson_attribution(
    portfolio_weights: pd.DataFrame,
    portfolio_returns: pd.DataFrame,
    benchmark_weights: pd.DataFrame | None,
    benchmark_returns: pd.DataFrame | None,
    pool: ETFPool,
) -> dict:
    """Brinson 归因.

    Args:
        portfolio_weights: index=date, columns=codes, values=weight
        portfolio_returns: index=date, columns=codes, values=daily return
        benchmark_weights: 同上, 缺省用等权
        benchmark_returns: 同上, 缺省用等权组合
        pool: ETFPool (用于类别)

    Returns dict with 9 keys:
        allocation_abs, selection_abs, interaction_abs
        allocation_pct, selection_pct, interaction_pct
        total_active, port_total_return, bench_total_return
    """
    if portfolio_weights.empty or portfolio_returns.empty:
        return {
            "allocation_abs": 0.0, "selection_abs": 0.0, "interaction_abs": 0.0,
            "allocation_pct": 0.0, "selection_pct": 0.0, "interaction_pct": 0.0,
            "total_active": 0.0,
            "port_total_return": 0.0, "bench_total_return": 0.0,
        }

    codes = list(portfolio_weights.columns)
    codes = [c for c in codes if c in portfolio_returns.columns]
    if not codes:
        return {
            "allocation_abs": 0.0, "selection_abs": 0.0, "interaction_abs": 0.0,
            "allocation_pct": 0.0, "selection_pct": 0.0, "interaction_pct": 0.0,
            "total_active": 0.0,
            "port_total_return": 0.0, "bench_total_return": 0.0,
        }

    common = portfolio_weights.index.intersection(portfolio_returns.index)
    pw = portfolio_weights.loc[common, codes]
    pr = portfolio_returns.loc[common, codes]

    if benchmark_weights is None or benchmark_returns is None:
        bw = pd.DataFrame(1.0 / len(codes), index=common, columns=codes)
        br = pr.copy()  # 基准 = 组合自身 (active = 0)
    else:
        bw = benchmark_weights.reindex(columns=codes).fillna(0.0)
        br = benchmark_returns.reindex(columns=codes).fillna(0.0)

    # Brinson 单期
    wp_minus_wb = (pw - bw).fillna(0.0)
    rp_minus_rb = (pr - br).fillna(0.0)
    rb = br.fillna(0.0)
    wb = bw.fillna(0.0)

    # 单期: allocation_i = (wp_i - wb_i) * rb_i
    alloc_per_period = (wp_minus_wb * rb).sum(axis=1)
    select_per_period = (wb * rp_minus_rb).sum(axis=1)
    interact_per_period = (wp_minus_wb * rp_minus_rb).sum(axis=1)

    alloc_abs = float(alloc_per_period.sum())
    select_abs = float(select_per_period.sum())
    interact_abs = float(interact_per_period.sum())
    total_active = alloc_abs + select_abs + interact_abs

    # 组合与基准总收益
    port_ret = _aligned_returns(pw, pr.add(1).cumprod())
    bench_ret = _aligned_returns(bw, br.add(1).cumprod())
    port_total = float((1 + port_ret).prod() - 1) if not port_ret.empty else 0.0
    bench_total = float((1 + bench_ret).prod() - 1) if not bench_ret.empty else 0.0

    if total_active != 0:
        alloc_pct = alloc_abs / total_active
        select_pct = select_abs / total_active
        interact_pct = interact_abs / total_active
    else:
        alloc_pct = 0.0
        select_pct = 0.0
        interact_pct = 0.0

    return {
        "allocation_abs": alloc_abs,
        "selection_abs": select_abs,
        "interaction_abs": interact_abs,
        "allocation_pct": alloc_pct,
        "selection_pct": select_pct,
        "interaction_pct": interact_pct,
        "total_active": total_active,
        "port_total_return": port_total,
        "bench_total_return": bench_total,
    }


__all__ = ["CATEGORIES", "brinson_attribution"]
