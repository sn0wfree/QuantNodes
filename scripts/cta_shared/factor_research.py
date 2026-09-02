# coding=utf-8
"""factor_research.py — 因子研究可复用组件库.

从 cta_cftc_strategy / cta_inventory_strategy / cta_broker_holding / cta_zn_inventory
四个研究脚本去重沉淀, 后续因子研究(CU/AU子策略等)直接复用.

组件:
    - rolling_zscore: 滚动z-score
    - compute_ic: 时序IC + 滚动IC + t-stat
    - quintile_spread: 五分位年化价差
    - is_oos_consistency: IS/OOS方向一致性
    - vol_target_backtest: 多品种vol target回测 (含按品种成本)
    - period_stats: 分期统计 (Sharpe/AnnRet/MaxDD/Calmar)
    - yearly_returns: 年度收益
    - load_cost_bps: cost_model.json 加载

方法论门槛 (固化):
    - IC显著 ≠ 可交易 (CFTC教训)
    - 五分位分层 IS/OOS方向一致率 < 60% → 止损
    - IS定方向, OOS独立验证, OOS Sharpe < 0.15 → 止损
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

REPORT_DIR = Path("reports/cta")
DEFAULT_IS_END = "2022-01-01"


# ── 基础变换 ────────────────────────────────────────────────────────────────

def rolling_zscore(s: pd.Series, window: int = 252,
                   min_periods: int | None = None) -> pd.Series:
    """滚动z-score."""
    mp = min_periods or window // 2
    m = s.rolling(window, min_periods=mp).mean()
    sd = s.rolling(window, min_periods=mp).std().replace(0, np.nan)
    return (s - m) / sd


# ── IC检验 ──────────────────────────────────────────────────────────────────

def compute_ic(factor: pd.Series, forward_ret: pd.Series,
               roll_window: int = 52, min_obs: int = 200) -> dict:
    """时序IC + 滚动IC均值 + t-stat.

    Args:
        factor: 因子值 (日频或周频)
        forward_ret: 未来收益 (已shift对齐)
        roll_window: 滚动IC窗口
        min_obs: 最小样本数

    Returns:
        dict(ic, ic_roll_mean, t_stat, n)
    """
    valid = factor.notna() & forward_ret.notna()
    if valid.sum() < min_obs:
        return {"ic": np.nan, "ic_roll_mean": np.nan, "t_stat": np.nan, "n": int(valid.sum())}
    f, r = factor[valid], forward_ret[valid]
    ic = f.corr(r)
    roll = f.rolling(roll_window).corr(r).dropna()
    t_stat = (roll.mean() / (roll.std() / np.sqrt(len(roll)))
              if len(roll) > 10 else np.nan)
    return {"ic": ic, "ic_roll_mean": roll.mean(), "t_stat": t_stat, "n": int(valid.sum())}


def quintile_spread(factor: pd.Series, forward_ret: pd.Series,
                    mask: pd.Series | None = None, n_bins: int = 5,
                    periods: int = 5) -> float | None:
    """五分位年化价差 (Q5-Q1, 年化).

    Args:
        factor: 因子值
        forward_ret: 未来收益
        mask: 布尔掩码 (如IS期/OOS期)
        periods: forward_ret的持有期(天), 用于年化
    """
    if mask is not None:
        m = mask & factor.notna() & forward_ret.notna()
    else:
        m = factor.notna() & forward_ret.notna()
    fv, rv = factor[m], forward_ret[m]
    if len(fv) < 150:
        return None
    q = pd.qcut(fv.rank(method="first"), n_bins, labels=False)
    return (rv[q == n_bins - 1].mean() - rv[q == 0].mean()) * 252 / periods


# ── IS/OOS方向一致性 ────────────────────────────────────────────────────────

def is_oos_direction_consistency(
    factor: pd.Series, forward_ret: pd.Series,
    is_end: str = DEFAULT_IS_END, min_is_spread: float = 0.05,
) -> dict | None:
    """IS/OOS五分位价差方向一致性检验.

    Returns:
        dict(is_spread, oos_spread, agree, is_significant) 或 None(样本不足)
    """
    idx = factor.index
    is_mask = pd.Series(idx < is_end, index=idx)
    s_is = quintile_spread(factor, forward_ret, is_mask)
    s_oos = quintile_spread(factor, forward_ret, ~is_mask)
    if s_is is None or s_oos is None:
        return None
    return {
        "is_spread": s_is, "oos_spread": s_oos,
        "agree": bool(np.sign(s_is) == np.sign(s_oos) and abs(s_is) > min_is_spread),
        "is_significant": bool(abs(s_is) > min_is_spread),
    }


# ── 回测 ────────────────────────────────────────────────────────────────────

def vol_target_backtest(
    signal: pd.DataFrame, close: pd.DataFrame,
    cost_bps: Mapping[str, float] | float = 5.0,
    target_vol: float = 0.08, vol_window: int = 20,
    max_scale: float = 1.0, pit_shift: int = 0,
) -> dict:
    """多品种vol target日频回测 (含按品种成本).

    Args:
        signal: (T, N) 信号面板 (±1/0, 已含方向; NaN按0处理)
        close: (T, N) 收盘价
        cost_bps: {product: 单边bp} 或标量
        target_vol: 单品种目标年化波动
        vol_window: 波动率窗口
        max_scale: vol scale上限
        pit_shift: 信号PIT延迟交易日 (0=当日起仓, 1=T+1执行)

    Returns:
        dict(sharpe, ann_ret, vol, max_dd, calmar, nav, pnl, turnover_annual, positions)
    """
    ret = close.pct_change(fill_method=None)
    vol = ret.rolling(vol_window).std() * np.sqrt(252)
    scale = (target_vol / vol).clip(upper=max_scale).fillna(0)
    sig = signal.fillna(0)
    if pit_shift > 0:
        sig = sig.shift(pit_shift).fillna(0)
    pos = sig * scale

    pnl_gross = pos.shift(1) * ret
    turnover = pos.diff().abs()
    if isinstance(cost_bps, Mapping):
        cbps = pd.Series(cost_bps).reindex(pos.columns).fillna(5.0)
    else:
        cbps = pd.Series(cost_bps, index=pos.columns)
    cost = turnover * cbps / 10000
    pnl = (pnl_gross - cost).sum(axis=1).dropna()

    nav = (1 + pnl).cumprod()
    n = len(pnl)
    ann_ret = nav.iloc[-1] ** (252 / n) - 1 if n > 0 else 0.0
    ann_vol = pnl.std() * np.sqrt(252) if n > 1 else 0.0
    dd = (nav / nav.cummax() - 1).min() if n > 0 else 0.0
    return {
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else 0.0,
        "ann_ret": ann_ret, "vol": ann_vol, "max_dd": dd,
        "calmar": ann_ret / abs(dd) if dd != 0 else 0.0,
        "nav": nav, "pnl": pnl, "positions": pos,
        "turnover_annual": turnover.sum(axis=1).mean() * 252 if n > 0 else 0.0,
    }


def period_stats(pnl: pd.Series, is_end: str = DEFAULT_IS_END) -> dict:
    """分期统计: 全期/IS/OOS 的 Sharpe/AnnRet/MaxDD/Calmar."""
    def _one(r: pd.Series) -> dict:
        if len(r) < 50:
            return {"sharpe": 0.0, "ann_ret": 0.0, "max_dd": 0.0, "calmar": 0.0}
        nav_r = (1 + r).cumprod()
        ann = nav_r.iloc[-1] ** (252 / len(r)) - 1
        dd = (nav_r / nav_r.cummax() - 1).min()
        return {
            "sharpe": ann / (r.std() * np.sqrt(252)) if r.std() > 0 else 0.0,
            "ann_ret": ann, "max_dd": dd,
            "calmar": ann / abs(dd) if dd != 0 else 0.0,
        }

    return {
        "full": _one(pnl.dropna()),
        "is": _one(pnl[pnl.index < is_end]),
        "oos": _one(pnl[pnl.index >= is_end]),
    }


def yearly_returns(pnl: pd.Series) -> pd.Series:
    """年度收益."""
    return pnl.groupby(pnl.index.year).apply(lambda x: (1 + x).prod() - 1)


# ── 成本 ────────────────────────────────────────────────────────────────────

def load_cost_bps(cost_file: Path | str = REPORT_DIR / "cost_model.json",
                  default_slip: float = 5.0,
                  default_comm: float = 0.1) -> dict[str, float]:
    """加载按品种成本 (滑点+手续费, 单边bp)."""
    table = json.loads(Path(cost_file).read_text())
    return {
        p: float(v.get("slippage_bps", default_slip))
        + float(v.get("commission_bps", default_comm))
        for p, v in table.items()
    }


def summarize_factor_test(ic: dict, cons: dict | None,
                          name: str = "") -> str:
    """单因子-品种检验结果的一行文本摘要 (打印用)."""
    ic_s = f"IC={ic['ic']:+.3f} rollIC={ic['ic_roll_mean']:+.3f} t={ic['t_stat']:+.1f}"
    if cons is None:
        return f"{name:<28} {ic_s} 样本不足"
    ag = "✓一致" if cons["agree"] else ("✗翻转" if cons["is_significant"] else "-")
    return (f"{name:<28} {ic_s} "
            f"IS={cons['is_spread']:+.1%} OOS={cons['oos_spread']:+.1%} {ag}")


__all__ = [
    "rolling_zscore", "compute_ic", "quintile_spread",
    "is_oos_direction_consistency", "vol_target_backtest",
    "period_stats", "yearly_returns", "load_cost_bps", "summarize_factor_test",
    "DEFAULT_IS_END", "REPORT_DIR",
]
