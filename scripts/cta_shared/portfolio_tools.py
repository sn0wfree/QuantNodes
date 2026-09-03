# coding=utf-8
"""cta_shared.portfolio_tools — 组合层资金分配可复用组件.

沉淀自 cta_allocation_optimization / cta_dynamic_leverage 两个研究脚本.

组件:
    adaptive_factor        单子策略自适应权重因子 (滚动Sharpe逆境降权)
    adaptive_factors       面板版 (逐列, 可跳过主策略列)
    dynamic_leverage       动态杠杆序列 (vol驱动, 线性插值)
    levered_portfolio      杠杆组合 (正确借款口径: 只对超额部分计息)
    yearly_diff            两组合逐年收益差分解

设计约定:
    - 杠杆借款成本只对超额部分 (w_main × (lev-1)) 计息, 不对全部资金计息
    - adaptive_factor 的 expanding median 只用历史数据 (PIT合规)
    - 权重调制本身不产生交易成本 (子策略内部交易成本已含在其NAV中)
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

DEFAULT_LEV_COST = 0.03
DEFAULT_MAIN_COL = "主CTA"


# ── 自适应子策略权重 ──────────────────────────────────────────────────────────

def adaptive_factor(
    ret: pd.Series,
    window: int = 60,
    cut: float = 0.5,
    min_periods: int | None = None,
    ann: int = 252,
) -> pd.Series:
    """单子策略自适应权重因子.

    规则: 滚动window日Sharpe < expanding中位数×0.5 且 Sharpe<0 → 权重=cut, 否则1.0.
    expanding median 只用历史数据 (PIT合规).

    Args:
        ret: 子策略日收益
        window: 滚动Sharpe窗口 (默认60)
        cut: 逆境权重 (默认0.5=减半)
        min_periods: expanding median最短样本 (默认2×window)
        ann: 年化因子

    Returns:
        权重因子序列 (与ret同index)
    """
    if min_periods is None:
        min_periods = window * 2
    roll_sharpe = ret.rolling(window).mean() / ret.rolling(window).std() * np.sqrt(ann)
    median = roll_sharpe.expanding(min_periods=min_periods).median()
    factor = pd.Series(1.0, index=ret.index)
    factor[(roll_sharpe < median * 0.5) & (roll_sharpe < 0)] = cut
    return factor.fillna(1.0)


def multi_window_factor(
    ret: pd.Series,
    windows: tuple[int, ...] = (20, 40, 60),
    cut: float = 0.5,
    method: str = "multiply",
    ann: int = 252,
) -> pd.Series:
    """多窗口独立评估 + 合成的渐进降权因子.

    每个窗口独立判定逆境 (rolling Sharpe<expanding med×0.5 & <0), 触发则×cut.
    合成方式:
      - multiply: f = Π f_w (各窗口因子相乘; 1票触发→cut, 2票→cut², …)
      - mean:     f = (Σ f_w) / n (各窗口因子均值; 1票→(1+(n-1)*1)/n 渐进)

    Args:
        ret: 子策略日收益
        windows: 多时间尺度窗口 (默认(20,40,60))
        cut: 单窗口触发后的因子值 (默认0.5)
        method: 'multiply' 或 'mean'
        ann: 年化因子

    Returns:
        权重因子序列 (与ret同index)
    """
    factors = [adaptive_factor(ret, w, cut=cut, ann=ann) for w in windows]
    if method == "multiply":
        out = factors[0].copy()
        for f in factors[1:]:
            out = out * f
        return out
    if method == "mean":
        return sum(factors) / len(factors)
    raise ValueError(f"method 必须是 'multiply' 或 'mean', 得到: {method}")


def adaptive_factors(
    rets: pd.DataFrame,
    skip_cols: tuple = (DEFAULT_MAIN_COL,),
    **kwargs,
) -> pd.DataFrame:
    """面板版自适应因子, skip_cols中的列恒为1.0 (如主CTA不做降权)."""
    out = {}
    for col in rets.columns:
        if col in skip_cols:
            out[col] = pd.Series(1.0, index=rets.index)
        else:
            out[col] = adaptive_factor(rets[col], **kwargs)
    return pd.DataFrame(out)


def multi_window_factors(
    rets: pd.DataFrame,
    skip_cols: tuple = (DEFAULT_MAIN_COL,),
    windows: tuple[int, ...] = (20, 40, 60),
    cut: float = 0.5,
    method: str = "multiply",
) -> pd.DataFrame:
    """面板版多窗口渐进降权."""
    out = {}
    for col in rets.columns:
        if col in skip_cols:
            out[col] = pd.Series(1.0, index=rets.index)
        else:
            out[col] = multi_window_factor(rets[col], windows=windows,
                                            cut=cut, method=method)
    return pd.DataFrame(out)


# ── 动态杠杆 ────────────────────────────────────────────────────────────────

def dynamic_leverage(
    main_ret: pd.Series,
    window: int = 60,
    vol_lo: float = 0.08,
    vol_hi: float = 0.12,
    lev_lo: float = 1.0,
    lev_hi: float = 1.5,
    ann: int = 252,
) -> pd.Series:
    """vol驱动的动态杠杆序列.

    vol<=vol_lo → lev_hi; vol>=vol_hi → lev_lo; 中间线性插值.
    vol缺省期 (窗口未满) 用lev_hi.
    """
    vol = main_ret.rolling(window).std() * np.sqrt(ann)
    frac = ((vol - vol_lo) / (vol_hi - vol_lo)).clip(0, 1)
    lev = lev_hi - frac * (lev_hi - lev_lo)
    return lev.fillna(lev_hi)


# ── 杠杆组合 ────────────────────────────────────────────────────────────────

def levered_portfolio(
    rets: pd.DataFrame,
    weights: Mapping[str, float],
    leverage: float = 1.0,
    lev_series: pd.Series | None = None,
    cost_annual: float = DEFAULT_LEV_COST,
    main_col: str = DEFAULT_MAIN_COL,
    ann: int = 252,
) -> pd.Series:
    """正确借款口径的杠杆组合日收益.

    组合 = Σ w_i × ret_i + w_main × (lev-1) × ret_main - w_main × (lev-1)⁺ × cost/ann
    借款只对超额部分 (w_main × (lev-1)) 计息; 杠杆=1时无借款项.

    Args:
        rets: 各策略日收益面板
        weights: 资金权重 (主策略列的权重是"资金", 敞口=资金×杠杆)
        leverage: 静态杠杆 (lev_series为None时生效)
        lev_series: 逐日动态杠杆 (优先于leverage)
        cost_annual: 借款年化成本
        main_col: 主策略列名
        ann: 年化因子

    Returns:
        组合日收益序列
    """
    w = pd.Series(weights).reindex(rets.columns).fillna(0)
    combo = (rets * w).sum(axis=1)
    lev = lev_series if lev_series is not None else pd.Series(leverage, index=rets.index)
    if main_col in w.index and w[main_col] > 0:
        excess = (lev - 1.0).clip(lower=0)
        combo = combo + w[main_col] * excess * rets[main_col]
        combo = combo - w[main_col] * excess * cost_annual / ann
    return combo.fillna(0)


# ── 分解工具 ────────────────────────────────────────────────────────────────

def yearly_diff(combo_a: pd.Series, combo_b: pd.Series) -> pd.DataFrame:
    """两组合逐年收益与差值分解 (贡献归因用)."""
    def _yr(c: pd.Series) -> pd.Series:
        return c.groupby(c.index.year).apply(lambda x: (1 + x).prod() - 1)

    yr_a, yr_b = _yr(combo_a), _yr(combo_b)
    return pd.DataFrame({"A": yr_a, "B": yr_b, "diff": yr_a - yr_b})


# ── 换手成本 ────────────────────────────────────────────────────────────────

def turnover_cost(
    wf_panel: pd.DataFrame,
    cost_bps: float | pd.Series = 5.0,
    w_main: float | None = None,
    lev_series: pd.Series | None = None,
    main_col: str = DEFAULT_MAIN_COL,
) -> pd.Series:
    """组合层换手成本 (日序列).

    包括两部分:
      1. 子策略权重切换: |Δ(w_i × f_i)| × cost_bps_i / 1e4
      2. 动态杠杆调整: |Δlev| × w_main × cost_main / 1e4 (若提供lev_series)

    子策略NAV构建已含其内部交易成本, 此处仅扣组合层调制带来的额外仓位调整.

    Args:
        wf_panel: 权重×因子面板 (各列为 w_i × f_i, 应满足 Σ≈1.0)
        cost_bps: 单边成本率 (bps), 标量或按列的Series
        w_main: 主策略资金权重 (提供lev_series时必填)
        lev_series: 动态杠杆序列 (若None则不扣杠杆调整成本)
        main_col: 主策略列名

    Returns:
        日成本序列 (正值, 从组合收益中扣除)
    """
    if isinstance(cost_bps, pd.Series):
        cbps = cost_bps.reindex(wf_panel.columns).fillna(5.0)
    else:
        cbps = pd.Series(cost_bps, index=wf_panel.columns)
    cost = (wf_panel.diff().abs() * cbps / 1e4).sum(axis=1)
    if lev_series is not None and w_main is not None:
        lev_cost = w_main * lev_series.diff().abs() * float(cbps.get(main_col, 5.0)) / 1e4
        cost = cost.fillna(0) + lev_cost.fillna(0)
    return cost.fillna(0)
