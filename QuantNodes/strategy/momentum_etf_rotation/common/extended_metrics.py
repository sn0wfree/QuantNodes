# coding=utf-8
"""17 个业绩指标 (extended metrics) - Stage 7.

参考 reports/extended_metrics.json (17 个指标).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ann_return(nav: pd.Series, freq: int = 252) -> float:
    if nav.empty or len(nav) < 2:
        return 0.0
    rets = nav.pct_change().dropna()
    if rets.empty:
        return 0.0
    n_years = len(rets) / freq
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    return float((1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1)


def _ann_vol(nav: pd.Series, freq: int = 252) -> float:
    if nav.empty or len(nav) < 2:
        return 0.0
    rets = nav.pct_change().dropna()
    return float(rets.std() * np.sqrt(freq)) if not rets.empty else 0.0


def _sharpe(nav: pd.Series, freq: int = 252) -> float:
    vol = _ann_vol(nav, freq)
    if vol == 0:
        return 0.0
    return _ann_return(nav, freq) / vol


def _sortino(nav: pd.Series, freq: int = 252) -> float:
    if nav.empty:
        return 0.0
    rets = nav.pct_change().dropna()
    if rets.empty:
        return 0.0
    downside = rets[rets < 0]
    dd = float(downside.std() * np.sqrt(freq)) if not downside.empty else 0.0
    if dd == 0:
        return 0.0
    return _ann_return(nav, freq) / dd


def _max_drawdown(nav: pd.Series) -> tuple[float, int]:
    if nav.empty or len(nav) < 2:
        return 0.0, 0
    cummax = nav.cummax()
    dd = (nav / cummax - 1)
    is_dd = dd < 0
    max_dd = float(dd.min())

    # 最大回撤天数: 最长连续回撤段
    max_run = 0
    cur_run = 0
    for v in is_dd.values:
        if v:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    return max_dd, max_run


def _calmar(nav: pd.Series, freq: int = 252) -> float:
    md, _ = _max_drawdown(nav)
    if md >= 0:
        return 0.0
    return _ann_return(nav, freq) / abs(md)


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


def _downside_dev(nav: pd.Series, freq: int = 252) -> float:
    if nav.empty:
        return 0.0
    rets = nav.pct_change().dropna()
    if rets.empty:
        return 0.0
    downside = rets[rets < 0]
    return float(downside.std() * np.sqrt(freq)) if not downside.empty else 0.0


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


def _win_rate(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    rets = nav.pct_change().dropna()
    if rets.empty:
        return 0.0
    return float((rets > 0).mean())


def _profit_loss_ratio(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    rets = nav.pct_change().dropna()
    if rets.empty:
        return 0.0
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    return float(wins.mean() / abs(losses.mean()))


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

    # 提取每段回撤的平均深度
    in_dd = False
    cur_dds = []
    depths = []
    for v, flag in zip(dd.values, is_dd.values):
        if flag:
            cur_dds.append(v)
            in_dd = True
        else:
            if cur_dds:
                depths.append(np.mean(cur_dds))
                cur_dds = []
            in_dd = False
    if cur_dds:
        depths.append(np.mean(cur_dds))
    return float(np.mean(depths)) if depths else 0.0


def extended_metrics(
    nav: pd.Series,
    benchmark_nav: pd.Series | None = None,
    rebalance_dates: list | None = None,
    freq: int = 252,
) -> dict:
    """17 个业绩指标 (与 reports/extended_metrics.json 字段对齐)."""
    if nav.empty or len(nav) < 2:
        return {}

    md, max_dd_days = _max_drawdown(nav)
    var_95, cvar_95 = _var_cvar(nav, alpha=0.05)
    avg_dd = _avg_dd(nav)

    calmar_avg_dd = (
        _calmar(nav, freq) / abs(avg_dd) if avg_dd < 0 else 0.0
    )

    return {
        "ann_return": _ann_return(nav, freq),
        "ann_vol": _ann_vol(nav, freq),
        "sharpe": _sharpe(nav, freq),
        "max_drawdown": md,
        "calmar": _calmar(nav, freq),
        "sortino": _sortino(nav, freq),
        "downside_dev": _downside_dev(nav, freq),
        "info_ratio": _info_ratio(nav, benchmark_nav, freq),
        "win_rate": _win_rate(nav),
        "profit_loss_ratio": _profit_loss_ratio(nav),
        "max_dd_duration": max_dd_days,
        "calmar_avg_dd": calmar_avg_dd,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "ann_turnover": 0.0,  # 占位, 未建模
        "max_monthly_loss": _max_monthly_loss(nav),
        "profit_months_ratio": _profit_months_ratio(nav),
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
]