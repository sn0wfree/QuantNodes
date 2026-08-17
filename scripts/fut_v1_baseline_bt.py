#!/usr/bin/env python3
"""fut_v1 基线回测 — 输出指标 + 净值曲线"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, "/home/ll/Public/QuantNodes")
import time, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from QuantNodes.strategy.cta.strategies.factor_modules import FactorModuleComboStrategy, ComboConfig, DEFAULT_PRODUCTS

PRODUCTS = DEFAULT_PRODUCTS  # 72个品种

START, END = "2018-01-01", "2025-12-31"

OUT_DIR = "/home/ll/Public/QuantNodes/output"
os.makedirs(OUT_DIR, exist_ok=True)


def compute_metrics(nav):
    r = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0] - 1
    years = len(r) / 252
    ann_ret = (1 + total) ** (1 / max(years, 0.01)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / ann_vol if ann_vol > 0 else 0
    dd = ((nav - nav.cummax()) / nav.cummax()).min()
    dd_end = nav.cummax()
    dd_duration = 0
    in_dd = False
    dd_start_date = None
    max_dd_days = 0
    for i in range(len(nav)):
        if nav.iloc[i] < dd_end.iloc[i]:
            if not in_dd:
                in_dd = True
                dd_start_date = nav.index[i]
        else:
            if in_dd:
                days = (nav.index[i] - dd_start_date).days
                max_dd_days = max(max_dd_days, days)
                in_dd = False
    if in_dd:
        days = (nav.index[-1] - dd_start_date).days
        max_dd_days = max(max_dd_days, days)
    calmar = ann_ret / abs(dd) if abs(dd) > 0 else 0
    sortino_denom = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else 0.001
    sortino = (ann_ret - 0.02) / sortino_denom
    win_rate = (r > 0).mean()
    profit_factor = r[r > 0].sum() / abs(r[r < 0].sum()) if (r < 0).any() else 999
    return {
        "AnnRet": ann_ret, "AnnVol": ann_vol, "Sharpe": sharpe, "Sortino": sortino,
        "MaxDD": dd, "MaxDDDays": max_dd_days, "Calmar": calmar,
        "WinRate": win_rate, "ProfitFactor": profit_factor,
        "TotalRet": total, "Years": years,
        "BestDay": r.max(), "WorstDay": r.min(),
    }


def plot_equity(nav, metrics, title, filename):
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
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {filename}")


def main():
    t0 = time.time()
    print("fut_v1 Baseline Backtest (72 Products)")
    print("=" * 60)

    cfg = ComboConfig(
        initial_cash=5_000_000, target_vol=0.30, max_leverage=3.0,
        vol_lookback=20, momentum_filter_window=60, momentum_filter_coef=0.0,
        filter_enabled=True, module_weighting="inv_vol", continuous=True,
        cont_z_window=252, cont_z_clip=3.0, margin_call=True,
        execution_lag=1, aggregation="tvpr",
    )

    print(f"Products: {PRODUCTS}")
    print(f"Period: {START} to {END}")
    print(f"Initial cash: 5,000,000 CNY")

    print("\nRunning fut_v1 baseline...")
    strat = FactorModuleComboStrategy(products=PRODUCTS, config=cfg, start=START, end=END)
    result = strat.run()
    print(f"  Done in {time.time()-t0:.0f}s")

    # 计算指标
    m = compute_metrics(result.nav)

    print(f"\n{'='*60}")
    print("fut_v1 Baseline Performance Metrics")
    print(f"{'='*60}")
    print(f"  {'Metric':<18} {'Value':>12}")
    print(f"  {'-'*32}")
    print(f"  {'Annual Return':<18} {m['AnnRet']:>11.1%}")
    print(f"  {'Annual Vol':<18} {m['AnnVol']:>11.1%}")
    print(f"  {'Sharpe Ratio':<18} {m['Sharpe']:>11.2f}")
    print(f"  {'Sortino Ratio':<18} {m['Sortino']:>11.2f}")
    print(f"  {'Max Drawdown':<18} {m['MaxDD']:>11.1%}")
    print(f"  {'Max DD Duration':<18} {m['MaxDDDays']:>10d} d")
    print(f"  {'Calmar Ratio':<18} {m['Calmar']:>11.2f}")
    print(f"  {'Win Rate':<18} {m['WinRate']:>11.0%}")
    print(f"  {'Profit Factor':<18} {m['ProfitFactor']:>11.2f}")
    print(f"  {'Total Return':<18} {m['TotalRet']:>11.1%}")
    print(f"  {'Best Day':<18} {m['BestDay']*100:>10.2f}%")
    print(f"  {'Worst Day':<18} {m['WorstDay']*100:>10.2f}%")
    print(f"  {'Years':<18} {m['Years']:>11.1f}")

    # 保存NAV
    nav_path = os.path.join(OUT_DIR, "fut_v1_72p_nav.parquet")
    result.nav.to_frame("nav").to_parquet(nav_path)
    print(f"\n  NAV saved: {nav_path}")

    # 生成图表
    chart_path = os.path.join(OUT_DIR, "fut_v1_72p_equity_curve.png")
    plot_equity(result.nav, m, f"fut_v1 Baseline (72 Products) — Equity Curve", chart_path)

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
