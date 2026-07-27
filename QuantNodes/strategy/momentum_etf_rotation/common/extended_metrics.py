# coding=utf-8
"""17 个业绩指标 (extended metrics) - Stage 7.

基础指标委托给 common.metrics.compute_metrics, 本模块只保留扩展指标 (VaR, CVaR, info_ratio, max_monthly_loss 等).

参考 reports/extended_metrics.json (17 个指标).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import compute_metrics


def _info_ratio(nav: pd.Series, bench: pd.Series | None = None, freq: int = 252) -> float:
    """Info ratio vs benchmark (默认 vs 沪深300 累计收益)."""
    if nav.empty or bench is None or bench.empty:
        return 0.0
    rets_n = nav.pct_change().dropna()
    rets_b = bench.pct_change().dropna()
    common = rets_n.index.intersection(rets_b.index)
    if len(common) < 2:
        return 0.0
    excess = rets_n.loc[common] - rets_b.loc[common]
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(freq))


def _var_cvar(nav: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    """历史法 VaR / CVaR."""
    if nav.empty or len(nav) < 2:
        return 0.0, 0.0
    rets = nav.pct_change().dropna()
    if rets.empty:
        return 0.0, 0.0
    var = float(rets.quantile(alpha))
    cvar = float(rets[rets <= var].mean()) if (rets <= var).any() else var
    return var, cvar


def _max_monthly_loss(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    monthly = nav.resample("M").last().pct_change().dropna()
    if monthly.empty:
        return 0.0
    return float(monthly.min())


def _profit_months_ratio(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    monthly = nav.resample("M").last().pct_change().dropna()
    if monthly.empty:
        return 0.0
    return float((monthly > 0).mean())


def _avg_dd(nav: pd.Series) -> float:
    """平均回撤深度 (Calmar(avg DD) 公式的分子)."""
    if nav.empty or len(nav) < 2:
        return 0.0
    cummax = nav.cummax()
    dd = (nav / cummax - 1)
    is_dd = dd < 0
    if not is_dd.any():
        return 0.0

    cur_dds = []
    depths = []
    for v, flag in zip(dd.values, is_dd.values):
        if flag:
            cur_dds.append(v)
        else:
            if cur_dds:
                depths.append(np.mean(cur_dds))
                cur_dds = []
    if cur_dds:
        depths.append(np.mean(cur_dds))
    return float(np.mean(depths)) if depths else 0.0


def extended_metrics(
    nav: pd.Series,
    benchmark_nav: pd.Series | None = None,
    rebalance_dates: list | None = None,
    freq: int = 252,
) -> dict:
    """17 个业绩指标 (与 reports/extended_metrics.json 字段对齐).

    基础指标 (ann_return/ann_vol/sharpe/calmar/sortino/max_drawdown/win_rate 等)
    委托给 common.metrics.compute_metrics.
    """
    if nav.empty or len(nav) < 2:
        return {}

    # 基础指标 (委托给 compute_metrics)
    base = compute_metrics(nav, freq=freq)

    # 扩展指标 (本模块特有)
    var_95, cvar_95 = _var_cvar(nav, alpha=0.05)
    avg_dd = _avg_dd(nav)
    calmar_avg_dd = (base["Calmar"] / abs(avg_dd)) if avg_dd < 0 else 0.0

    return {
        # 基础指标 (扁平键名, 兼容旧接口)
        "ann_return": base["AnnRet"],
        "ann_vol": base["Vol"],
        "sharpe": base["Sharpe"],
        "max_drawdown": base["MaxDD"],
        "calmar": base["Calmar"],
        "sortino": base["Sortino"],
        "downside_dev": base["Vol"] * base["Sortino"] / base["Sharpe"] if base["Sharpe"] > 0 else 0.0,
        "info_ratio": _info_ratio(nav, benchmark_nav, freq),
        "win_rate": base["WinRate"],
        "profit_loss_ratio": base["PayoffRatio"],
        "max_dd_duration": base["MaxDDDays"],
        "calmar_avg_dd": calmar_avg_dd,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "ann_turnover": 0.0,  # 占位, 未建模
        "max_monthly_loss": _max_monthly_loss(nav),
        "profit_months_ratio": _profit_months_ratio(nav),
    }


def kelly_audit(nav: pd.Series, freq: int = 252) -> dict:
    """Kelly 比例审计 (来自 10_TURTLE_TRADING_MATHEMATICS.md ACT-2).

    计算:
        max_log_growth = Sharpe²/2 (满 Kelly 理论增长率)
        actual_log_growth = ln(1 + CAGR)
        kelly_fraction = actual / max (当前 sizing 位置)

    返回:
        dict with keys:
            sharpe: Sharpe 比率
            cagr: 年化收益率
            max_log_growth: 满 Kelly 理论增长率
            actual_log_growth: 实际对数增长率
            kelly_fraction: 当前 Kelly 比例
            status: "SAFE" (<50%) | "CAUTION" (50-80%) | "OVER-KELLY" (>80%)
    """
    if nav.empty or len(nav) < 2:
        return {
            "sharpe": 0.0,
            "cagr": 0.0,
            "max_log_growth": 0.0,
            "actual_log_growth": 0.0,
            "kelly_fraction": 0.0,
            "status": "UNKNOWN",
        }

    # 委托给公共指标模块
    base = compute_metrics(nav, freq=freq)
    sharpe = base["Sharpe"]
    cagr = base["AnnRet"]

    # 满 Kelly 理论增长率
    max_log_growth = sharpe**2 / 2

    # 实际对数增长率
    actual_log_growth = np.log(1 + cagr) if cagr > -1 else float('-inf')

    # Kelly 比例
    kelly_fraction = actual_log_growth / max_log_growth if max_log_growth > 0 else 0.0

    # 状态判断
    if kelly_fraction < 0.5:
        status = "SAFE"
    elif kelly_fraction < 0.8:
        status = "CAUTION"
    else:
        status = "OVER-KELLY"

    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "max_log_growth": max_log_growth,
        "actual_log_growth": actual_log_growth,
        "kelly_fraction": kelly_fraction,
        "status": status,
    }


def format_metrics_table(
    metrics_inv_vol: dict,
    metrics_equal_weight: dict | None = None,
    output_path: str | Path | None = None,
) -> str:
    """生成 17 指标对比 markdown 表.

    与 reports/extended_metrics.md 格式对齐.
    """
    lines = [
        "| # | 指标 | 逆波动 | 等权 | 差异 |",
        "|---|------|--------|------|------|",
    ]
    labels = [
        ("ann_return", "年化收益"),
        ("ann_vol", "年化波动"),
        ("sharpe", "夏普比率"),
        ("max_drawdown", "最大回撤"),
        ("calmar", "Calmar"),
        ("sortino", "Sortino"),
        ("info_ratio", "Info Ratio (vs 沪深300)"),
        ("win_rate", "日胜率"),
        ("profit_loss_ratio", "盈亏比"),
        ("max_dd_duration", "最大 DD 天数"),
        ("calmar_avg_dd", "Calmar(avg DD)"),
        ("var_95", "VaR (95%)"),
        ("cvar_95", "CVaR (95%)"),
        ("downside_dev", "下行偏差"),
        ("ann_turnover", "年化换手"),
        ("max_monthly_loss", "最大月跌"),
        ("profit_months_ratio", "盈利月比例"),
    ]

    for i, (key, label) in enumerate(labels, 1):
        iv = metrics_inv_vol.get(key, 0.0)
        ew = metrics_equal_weight.get(key, 0.0) if metrics_equal_weight else 0.0
        if key == "ann_return":
            iv_s = f"{iv * 100:.2f}%"
            ew_s = f"{ew * 100:.2f}%" if metrics_equal_weight else "-"
            diff_s = f"{iv - ew:+.2%}" if metrics_equal_weight else "-"
        elif key in ("max_drawdown", "var_95", "cvar_95", "downside_dev", "max_monthly_loss"):
            iv_s = f"{iv * 100:.2f}%"
            ew_s = f"{ew * 100:.2f}%" if metrics_equal_weight else "-"
            diff_s = f"{(iv - ew) * 100:+.2f}%" if metrics_equal_weight else "-"
        elif key in ("win_rate", "profit_months_ratio"):
            iv_s = f"{iv * 100:.2f}%"
            ew_s = f"{ew * 100:.2f}%" if metrics_equal_weight else "-"
            diff_s = f"{(iv - ew) * 100:+.2f}%" if metrics_equal_weight else "-"
        elif key in ("max_dd_duration",):
            iv_s = f"{iv:.0f}"
            ew_s = f"{ew:.0f}" if metrics_equal_weight else "-"
            diff_s = f"{iv - ew:+.0f}" if metrics_equal_weight else "-"
        else:
            iv_s = f"{iv:.2f}"
            ew_s = f"{ew:.2f}" if metrics_equal_weight else "-"
            diff_s = f"{iv - ew:+.2f}" if metrics_equal_weight else "-"

        lines.append(f"| {i} | {label} | **{iv_s}** | {ew_s} | {diff_s} |")

    md = "\n".join(lines)
    if output_path:
        Path(output_path).write_text(md, encoding="utf-8")
    return md


__all__ = [
    "extended_metrics",
    "format_metrics_table",
    "kelly_audit",
]
