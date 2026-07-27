# coding=utf-8
"""统一业绩指标计算 — 消除 v0-v10 中 6 种不同实现.

用法:
    from QuantNodes.strategy.momentum_etf_rotation.common.metrics import compute_metrics

    result = compute_metrics(nav_series)
    # result = {'AnnRet': 0.08, 'Vol': 0.12, 'Sharpe': 0.67, ...}

    # 指定 OOS 窗口
    result = compute_metrics(nav_series, oos_start='2022-01-01')
    # result 包含 'Full' 和 'OOS' 两个子字典

核心约定:
    - 年化收益: (1 + total_return) ^ (1 / calendar_years) - 1
      其中 calendar_years = (index[-1] - index[0]).days / 365.25
    - 波动率: daily_std * sqrt(freq)
    - freq 自动检测: 中位数日间隔 > 4天 → 52 (周频), 否则 252 (日频)
    - Sharpe = AnnRet / Vol
    - Sortino = AnnRet / downside_vol
    - Calmar = AnnRet / |MaxDD|
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_freq(nav: pd.Series) -> int:
    """自动检测数据频率.

    规则: 中位数日间隔 > 4天 → 周频(52), 否则日频(252).

    Parameters:
        nav: NAV 序列 (DatetimeIndex)

    Returns:
        年化因子 (52 或 252)
    """
    if len(nav) < 2:
        return 252
    median_gap = (nav.index[1:] - nav.index[:-1]).median()
    if isinstance(median_gap, (int, float)):
        return 252
    return 52 if median_gap > pd.Timedelta(days=4) else 252


def _ann_return(nav: pd.Series) -> float:
    """年化收益 = (1 + total_return) ^ (1 / calendar_years) - 1."""
    if nav.empty or len(nav) < 2:
        return 0.0
    total = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = (nav.index[-1] - nav.index[0]).days / 365.25
    if n_years <= 0:
        return 0.0
    return float((1 + total) ** (1 / n_years) - 1)


def _max_drawdown(nav: pd.Series) -> tuple[float, int]:
    """最大回撤及持续天数."""
    if nav.empty or len(nav) < 2:
        return 0.0, 0
    cummax = nav.cummax()
    dd = nav / cummax - 1
    max_dd = float(dd.min())
    # 最大回撤天数
    is_dd = dd < 0
    max_run = 0
    cur_run = 0
    for v in is_dd.values:
        if v:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_dd, max_run


def compute_metrics(
    nav: pd.Series,
    freq: int | str | None = None,
    oos_start: str | None = None,
) -> dict:
    """统一指标计算 (所有版本共用).

    Parameters:
        nav: NAV 序列 (DatetimeIndex, 日频或周频)
        freq: 年化因子 ('D'=252, 'W'=52, 'M'=12, None=自动检测)
        oos_start: OOS 起始日 (如 '2022-01-01')。指定时返回 {'Full': {...}, 'OOS': {...}}

    Returns:
        指标字典 (或嵌套字典)
    """
    if nav is None or nav.empty:
        return _empty_result()

    valid = nav.dropna()
    if len(valid) < 2:
        return _empty_result()

    # 自动检测频率
    if freq is None:
        freq = detect_freq(valid)
    elif isinstance(freq, str):
        freq = {"D": 252, "W": 52, "M": 12, "Q": 4}.get(freq.upper(), 252)

    result = _compute_single(valid, freq)

    if oos_start is not None:
        oos = valid.loc[oos_start:]
        if len(oos) >= 2:
            result = {"Full": result, "OOS": _compute_single(oos, freq)}
        else:
            result = {"Full": result, "OOS": _empty_result()}

    return result


def performance_metrics_legacy(nav: pd.Series, freq: int = 252) -> dict:
    """兼容层: 返回扁平键名 (ann_return/sharpe/...) — 供历史代码使用.

    内部委托 compute_metrics, 然后做键名转换.
    """
    base = compute_metrics(nav, freq=freq)
    return {
        "ann_return": base["AnnRet"],
        "ann_vol": base["Vol"],
        "sharpe": base["Sharpe"],
        "max_drawdown": base["MaxDD"],
        "calmar": base["Calmar"],
    }


def _compute_single(nav: pd.Series, freq: int) -> dict:
    """计算单段指标."""
    rets = nav.pct_change().dropna()
    if rets.empty:
        return _empty_result()

    ann_ret = _ann_return(nav)
    vol = float(rets.std() * np.sqrt(freq))
    sharpe = ann_ret / vol if vol > 0 else 0.0

    # Sortino
    downside = rets[rets < 0]
    dv = float(downside.std() * np.sqrt(freq)) if not downside.empty else 0.0
    sortino = ann_ret / dv if dv > 1e-9 else 0.0

    # MaxDD
    max_dd, mdd_days = _max_drawdown(nav)

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd < -1e-6 else 0.0

    # WinRate
    win_rate = float((rets > 0).mean())

    # PayoffRatio
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    payoff = float(wins.mean() / abs(losses.mean())) if len(wins) > 0 and len(losses) > 0 else 0.0

    return {
        "AnnRet": ann_ret,
        "Vol": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": max_dd,
        "MaxDDDays": mdd_days,
        "Calmar": calmar,
        "WinRate": win_rate,
        "PayoffRatio": payoff,
        "Years": (nav.index[-1] - nav.index[0]).days / 365.25,
        "Freq": freq,
    }


def _empty_result() -> dict:
    """返回空指标字典."""
    return {
        "AnnRet": 0.0, "Vol": 0.0, "Sharpe": 0.0, "Sortino": 0.0,
        "MaxDD": 0.0, "MaxDDDays": 0, "Calmar": 0.0,
        "WinRate": 0.0, "PayoffRatio": 0.0, "Years": 0.0, "Freq": 252,
    }


def format_metrics_table(
    results: dict[str, dict],
    sort_by: str = "Sharpe",
    ascending: bool = False,
) -> str:
    """格式化指标表为文本.

    Parameters:
        results: {策略名: metrics_dict}
        sort_by: 排序字段
        ascending: 升序

    Returns:
        格式化文本
    """
    if not results:
        return ""

    rows = []
    for name, m in results.items():
        rows.append({
            "策略": name,
            "年化收益": f"{m['AnnRet']*100:+.2f}%",
            "波动率": f"{m['Vol']*100:.2f}%",
            "Sharpe": f"{m['Sharpe']:.3f}",
            "Sortino": f"{m['Sortino']:.3f}",
            "MaxDD": f"{m['MaxDD']*100:.2f}%",
            "Calmar": f"{m['Calmar']:.3f}",
            "WinRate": f"{m['WinRate']:.1%}",
        })

    df = pd.DataFrame(rows)
    if sort_by in df.columns:
        # 数值排序
        key_col = sort_by
        if sort_by in ("AnnRet", "Vol", "MaxDD", "WinRate"):
            df["_sort"] = df[sort_by].str.rstrip("%").astype(float)
        elif sort_by in ("Sharpe", "Sortino", "Calmar"):
            df["_sort"] = df[sort_by].astype(float)
        else:
            df["_sort"] = range(len(df))
        df = df.sort_values("_sort", ascending=ascending).drop(columns=["_sort"])

    return df.to_string(index=False)
