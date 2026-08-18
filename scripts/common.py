#!/usr/bin/env python3
"""fut_v1 策略脚本公共工具模块.

提供:
- compute_metrics: 统一指标计算
- DEFAULT_CONFIG: 默认ComboConfig参数
- run_strategy: 快速运行策略并返回结果
- print_metrics: 格式化打印指标

用法:
    from scripts.common import compute_metrics, DEFAULT_CONFIG, run_strategy, print_metrics
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "/home/ll/Public/QuantNodes")
import time
import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.strategies.factor_modules import (
    FactorModuleComboStrategy, ComboConfig, DEFAULT_PRODUCTS,
)


# ── 默认配置 ────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = dict(
    initial_cash=5_000_000,
    target_vol=0.30,
    max_leverage=3.0,
    vol_lookback=20,
    momentum_filter_window=60,
    momentum_filter_coef=0.0,
    filter_enabled=True,
    module_weighting="inv_vol",
    continuous=True,
    cont_z_window=252,
    cont_z_clip=3.0,
    margin_call=True,
    execution_lag=1,
    aggregation="tvpr",
    commission_rate=0.0002,
    min_position_change=0.50,
    slippage_bps=0.5,
    position_ema_alpha=0.10,
    tail_risk_enabled=True,
    tail_risk_corr_threshold=0.30,
    enable_cluster_filter=False,
)

DEFAULT_START = "2018-01-01"
DEFAULT_END = "2025-12-31"


# ── 指标计算 ────────────────────────────────────────────────────────────────

def compute_metrics(nav: pd.Series, rf: float = 0.02) -> dict:
    """统一业绩指标计算.

    Parameters:
        nav: NAV序列 (DatetimeIndex, 日频)
        rf: 无风险利率 (年化)

    Returns:
        指标字典: AnnRet, AnnVol, Sharpe, Sortino, MaxDD, MaxDDDays, Calmar,
                  WinRate, ProfitFactor, TotalRet, Years, BestDay, WorstDay
    """
    r = nav.pct_change().dropna()
    if len(r) == 0:
        return _empty_metrics()

    total = nav.iloc[-1] / nav.iloc[0] - 1
    years = len(r) / 252
    ann_ret = (1 + total) ** (1 / max(years, 0.01)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0

    dd = ((nav - nav.cummax()) / nav.cummax()).min()
    calmar = ann_ret / abs(dd) if abs(dd) > 0 else 0

    sortino_denom = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else 0.001
    sortino = (ann_ret - rf) / sortino_denom

    win_rate = (r > 0).mean()
    profit_factor = r[r > 0].sum() / abs(r[r < 0].sum()) if (r < 0).any() else 999

    # 最大回撤天数
    cummax = nav.cummax()
    is_dd = (nav < cummax)
    max_dd_days = 0
    cur_run = 0
    for v in is_dd.values:
        if v:
            cur_run += 1
            max_dd_days = max(max_dd_days, cur_run)
        else:
            cur_run = 0

    return {
        "AnnRet": ann_ret,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": dd,
        "MaxDDDays": max_dd_days,
        "Calmar": calmar,
        "WinRate": win_rate,
        "ProfitFactor": profit_factor,
        "TotalRet": total,
        "Years": years,
        "BestDay": float(r.max()),
        "WorstDay": float(r.min()),
    }


def _empty_metrics() -> dict:
    return {
        "AnnRet": 0.0, "AnnVol": 0.0, "Sharpe": 0.0, "Sortino": 0.0,
        "MaxDD": 0.0, "MaxDDDays": 0, "Calmar": 0.0,
        "WinRate": 0.0, "ProfitFactor": 0.0,
        "TotalRet": 0.0, "Years": 0.0,
        "BestDay": 0.0, "WorstDay": 0.0,
    }


# ── 策略运行 ────────────────────────────────────────────────────────────────

def run_strategy(
    products=None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    **config_overrides,
) -> tuple:
    """快速运行fut_v1策略.

    Parameters:
        products: 品种列表 (默认66品种)
        start: 起始日期
        end: 结束日期
        **config_overrides: 覆盖默认配置的参数

    Returns:
        (result, metrics): CtaBacktestResult和指标字典
    """
    products = products or DEFAULT_PRODUCTS
    cfg_dict = {**DEFAULT_CONFIG, **config_overrides}
    cfg = ComboConfig(**cfg_dict)

    strat = FactorModuleComboStrategy(
        products=products, config=cfg, start=start, end=end,
    )
    result = strat.run()
    metrics = compute_metrics(result.nav)

    return result, metrics


# ── 打印工具 ────────────────────────────────────────────────────────────────

def print_metrics(metrics: dict, title: str = "Performance Metrics"):
    """格式化打印指标."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  {'Metric':<18} {'Value':>12}")
    print(f"  {'-'*32}")

    rows = [
        ("Annual Return", "AnnRet", "{:>11.1%}"),
        ("Annual Vol", "AnnVol", "{:>11.1%}"),
        ("Sharpe Ratio", "Sharpe", "{:>11.2f}"),
        ("Sortino Ratio", "Sortino", "{:>11.2f}"),
        ("Max Drawdown", "MaxDD", "{:>11.1%}"),
        ("Max DD Duration", "MaxDDDays", "{:>10d} d"),
        ("Calmar Ratio", "Calmar", "{:>11.2f}"),
        ("Win Rate", "WinRate", "{:>11.0%}"),
        ("Profit Factor", "ProfitFactor", "{:>11.2f}"),
        ("Total Return", "TotalRet", "{:>11.1%}"),
        ("Best Day", "BestDay", None),
        ("Worst Day", "WorstDay", None),
        ("Years", "Years", "{:>11.1f}"),
    ]

    for label, key, fmt in rows:
        val = metrics.get(key, 0)
        if fmt is None:
            # BestDay/WorstDay 用百分比
            print(f"  {label:<18} {val*100:>10.2f}%")
        elif key == "MaxDDDays":
            print(f"  {label:<18} {fmt.format(int(val))}")
        else:
            print(f"  {label:<18} {fmt.format(val)}")


def print_comparison(
    results: dict[str, dict],
    metrics: list[str] = None,
    title: str = "Comparison",
):
    """打印多组结果对比表."""
    if metrics is None:
        metrics = ["AnnRet", "AnnVol", "Sharpe", "Sortino", "MaxDD", "Calmar"]

    names = list(results.keys())
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    # 表头
    header = f"  {'Metric':<14}" + "".join(f" {n:>14}" for n in names)
    print(header)
    print(f"  {'-'*(14 + 15*len(names))}")

    # 数据行
    for key in metrics:
        row = f"  {key:<14}"
        for name in names:
            val = results[name].get(key, 0)
            if key in ("AnnRet", "AnnVol", "MaxDD"):
                row += f" {val:>13.1%}"
            else:
                row += f" {val:>14.2f}"
        print(row)


# ── 年度分析 ────────────────────────────────────────────────────────────────

def yearly_breakdown(nav: pd.Series) -> pd.DataFrame:
    """按年度分解NAV指标."""
    years = sorted(nav.index.year.unique())
    rows = []
    for y in years:
        yr_nav = nav[nav.index.year == y]
        if len(yr_nav) < 10:
            continue
        m = compute_metrics(yr_nav)
        rows.append({
            "Year": y,
            "Return": m["AnnRet"],
            "MaxDD": m["MaxDD"],
            "Vol": m["AnnVol"],
            "Sharpe": m["Sharpe"],
            "Calmar": m["Calmar"],
        })
    return pd.DataFrame(rows)


def print_yearly(nav: pd.Series):
    """打印年度绩效表."""
    df = yearly_breakdown(nav)
    print(f"\n{'='*70}")
    print("  Year-by-Year Performance")
    print(f"{'='*70}")
    print(f"  {'Year':<6} {'Return':>8} {'MaxDD':>8} {'Vol':>8} {'Sharpe':>8} {'Calmar':>8}")
    print(f"  {'-'*48}")
    for _, row in df.iterrows():
        print(f"  {int(row['Year']):<6} {row['Return']:>7.1%} {row['MaxDD']:>7.1%} "
              f"{row['Vol']:>7.1%} {row['Sharpe']:>8.2f} {row['Calmar']:>8.2f}")


# ── 持仓分析 ────────────────────────────────────────────────────────────────

def analyze_positions(result, products=None) -> dict:
    """分析持仓特征.

    Returns:
        持仓分析字典: active_count, concentration, turnover, etc.
    """
    idx = pd.DatetimeIndex(result.dates)
    pos = pd.DataFrame(result.positions, index=idx)

    if products is None:
        products = [c for c in pos.columns if (pos[c] != 0).any()]

    # 活跃品种数
    active_count = (pos[products].abs() > 0).sum(axis=1)

    # 持仓集中度
    pos_pct = pos[products].abs().div(pos[products].abs().sum(axis=1), axis=0)
    avg_pos_pct = pos_pct.mean().sort_values(ascending=False)

    # 换手率
    pos_change = pos[products].diff().abs().sum(axis=1)
    avg_exposure = pos[products].abs().sum(axis=1).mean()
    annual_turnover = pos_change.mean() * 252 / avg_exposure if avg_exposure > 0 else 0

    return {
        "active_count_mean": float(active_count.mean()),
        "active_count_max": int(active_count.max()),
        "top5_pct": float(avg_pos_pct.head(5).sum()),
        "top10_pct": float(avg_pos_pct.head(10).sum()),
        "annual_turnover": float(annual_turnover),
        "avg_position": pos[products].abs().mean(),
        "position_pct": avg_pos_pct,
    }


def print_position_analysis(result, products=None):
    """打印持仓分析."""
    stats = analyze_positions(result, products)

    print(f"\n{'='*60}")
    print("  Position Analysis")
    print(f"{'='*60}")
    print(f"  Active products (avg):  {stats['active_count_mean']:.1f}")
    print(f"  Active products (max):  {stats['active_count_max']}")
    print(f"  Top 5 concentration:    {stats['top5_pct']:.1%}")
    print(f"  Top 10 concentration:   {stats['top10_pct']:.1%}")
    print(f"  Annual turnover:        {stats['annual_turnover']:.1f}x")

    print(f"\n  Top 10 by avg position:")
    top10 = stats["avg_position"].sort_values(ascending=False).head(10)
    for p, v in top10.items():
        print(f"    {p:<8} {v:>8.1f} lots")


# ── 图表工具 ────────────────────────────────────────────────────────────────

def plot_equity_curve(
    nav: pd.Series,
    metrics: dict = None,
    title: str = "Equity Curve",
    save_path: str = None,
):
    """绘制净值曲线、回撤、日收益三合一图."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. 净值曲线
    ax = axes[0]
    nav_norm = nav / nav.iloc[0]
    ax.plot(nav_norm.index, nav_norm.values, color="#2196F3", linewidth=1.2, label="NAV")
    ax.fill_between(nav_norm.index, 1, nav_norm.values, alpha=0.15, color="#2196F3")
    ax.axhline(1, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("NAV (normalized)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    if metrics:
        info = (f"AnnRet={metrics['AnnRet']:.1%}  Sharpe={metrics['Sharpe']:.2f}  "
                f"MaxDD={metrics['MaxDD']:.1%}  Calmar={metrics['Calmar']:.2f}")
        ax.text(0.02, 0.05, info, transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # 2. 回撤
    ax = axes[1]
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax
    ax.fill_between(drawdown.index, 0, drawdown.values, color="#F44336", alpha=0.6)
    ax.set_ylabel("Drawdown")
    ax.set_ylim(drawdown.min() * 1.2, 0.02)
    ax.grid(True, alpha=0.3)

    # 3. 日收益
    ax = axes[2]
    daily_ret = nav.pct_change().dropna()
    colors = ["#4CAF50" if x >= 0 else "#F44336" for x in daily_ret]
    ax.bar(daily_ret.index, daily_ret.values, color=colors, alpha=0.6, width=1)
    ax.set_ylabel("Daily Return")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Chart saved: {save_path}")
    plt.close()


# ── CLI工具 ──────────────────────────────────────────────────────────────────

def parse_args(description: str = "fut_v1 backtest"):
    """通用命令行参数解析."""
    import argparse
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--start", default=DEFAULT_START, help="起始日期")
    parser.add_argument("--end", default=DEFAULT_END, help="结束日期")
    parser.add_argument("--quick", action="store_true", help="10品种快速模式")
    parser.add_argument("--output", default="output", help="输出目录")
    return parser.parse_args()


def get_products(quick: bool = False) -> list:
    """获取品种列表."""
    if quick:
        return ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]
    return DEFAULT_PRODUCTS


# ── 导出 ────────────────────────────────────────────────────────────────────

__all__ = [
    "compute_metrics",
    "DEFAULT_CONFIG",
    "DEFAULT_START",
    "DEFAULT_END",
    "run_strategy",
    "print_metrics",
    "print_comparison",
    "yearly_breakdown",
    "print_yearly",
    "analyze_positions",
    "print_position_analysis",
    "plot_equity_curve",
    "parse_args",
    "get_products",
]
