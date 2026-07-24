#!/usr/bin/env python3
"""v7.10 周一开盘执行测试: 对比原始 vs 周一开盘执行.

用法:
  python3.10 scripts/v7_10_monday_open_test.py

输出:
  reports/momentum_etf_rotation/v7_10_monday_open_comparison.md
  reports/momentum_etf_rotation/v7_10_monday_open_nav.png
  reports/momentum_etf_rotation/v7_10_monday_open_drawdown.png
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_v7_10_data,
    load_daily_etf_returns,
    load_weekly_monday_open_returns,
    load_weekly_ohlcv_returns,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import (
    expanding_window_tvpr,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config,
    construct_portfolio,
    calculate_daily_nav,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

MIN_HISTORY = 52
FREQ_PER_YEAR = 52
N_CORES = 20

# v7.10 最优 λ (两阶段 CV 选出)
BEST_LAMBDA_TV = 0.06
BEST_LAMBDA_L1 = 0.105


def compute_metrics(nav: pd.Series) -> dict:
    if nav.empty or len(nav) < 2:
        return dict(ann_return=0, ann_vol=0, sharpe=0, max_dd=0, calmar=0)
    rets = nav.pct_change().dropna()
    n_years = len(rets) / FREQ_PER_YEAR
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = float((1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1)
    ann_vol = float(rets.std() * np.sqrt(FREQ_PER_YEAR))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = nav.cummax()
    max_dd = float((nav / cummax - 1).min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    return dict(ann_return=round(ann_ret, 4), ann_vol=round(ann_vol, 4),
                sharpe=round(sharpe, 4), max_dd=round(max_dd, 4), calmar=round(calmar, 4))


def compute_daily_metrics(nav_daily: pd.Series) -> dict:
    if nav_daily.empty or len(nav_daily) < 2:
        return dict(ann_return=0, ann_vol=0, sharpe=0, max_dd=0, max_dd_duration_days=0)
    rets = nav_daily.pct_change().dropna()
    n_years = len(rets) / 252
    total_ret = nav_daily.iloc[-1] / nav_daily.iloc[0] - 1
    ann_ret = float((1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1)
    ann_vol = float(rets.std() * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    dd = nav_daily / nav_daily.cummax() - 1
    max_dd = float(dd.min())
    underwater = dd < -1e-6
    if underwater.any():
        groups = (~underwater).cumsum()
        dd_durations = underwater.groupby(groups).sum()
        max_dd_duration = int(dd_durations.max())
    else:
        max_dd_duration = 0
    return dict(ann_return=round(ann_ret, 4), ann_vol=round(ann_vol, 4),
                sharpe=round(sharpe, 4), max_dd=round(max_dd, 4),
                max_dd_duration_days=max_dd_duration)


def main() -> int:
    logging.info("=" * 60)
    logging.info("v7.10 周一开盘执行测试")
    logging.info("=" * 60)

    out_dir = REPO / "reports/momentum_etf_rotation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    t0 = time.time()
    X_panel, Y_df, codes = load_v7_10_data()
    T, N, K = X_panel.shape
    daily_returns = load_daily_etf_returns()
    monday_open_returns = load_weekly_monday_open_returns(codes)
    ohlc_returns = load_weekly_ohlcv_returns(codes)  # 同源周五到周五收益
    logging.info("  数据加载: X=%s, K=%d, 耗时=%.1fs", X_panel.shape, K, time.time() - t0)

    # 用 OHLCV 同源收益替代 Y (NAV 收益), 确保两种模式数据源一致
    common_idx = Y_df.index.intersection(ohlc_returns.index).intersection(monday_open_returns.index)
    monday_open_returns = monday_open_returns.loc[common_idx]
    ohlc_returns = ohlc_returns.loc[common_idx]
    Y_aligned = ohlc_returns  # 用 OHLCV 收益替代 NAV 收益
    X_aligned = X_panel[-len(common_idx):]
    T_aligned = len(common_idx)
    logging.info("  对齐后: T=%d (原 %d)", T_aligned, T)

    # 2. Beta estimation (用 OHLCV Fri-to-Fri 收益)
    logging.info("=" * 60)
    logging.info("Beta 估计 (expanding window)...")
    t0 = time.time()
    Y_arr = Y_aligned.values
    beta = expanding_window_tvpr(
        Y_aligned, X_aligned,
        BEST_LAMBDA_TV, BEST_LAMBDA_L1,
        min_history=MIN_HISTORY, max_iter=200, tol=1e-5, step=4,
    )
    logging.info("  Beta 估计耗时: %.1fs", time.time() - t0)

    # 3. Backtest: 原始 (OHLCV 周五到周五, 含周末隔夜)
    logging.info("=" * 60)
    logging.info("Backtest: 原始 (OHLCV 周五到周五, 含周末隔夜)...")
    cfg = V7_6Config()
    nav_original, weights_original = construct_portfolio(
        Y_aligned, X_aligned, beta, cfg, return_weights=True,
    )
    oos_start = MIN_HISTORY + int((T_aligned - MIN_HISTORY) * 0.6)
    oos_date = Y_aligned.index[oos_start]
    nav_original_oos = nav_original.iloc[oos_start:]
    metrics_original = compute_metrics(nav_original_oos)

    # 4. Backtest: 周一开盘执行 (不含周末隔夜)
    #    关键: Y_aligned 仍然是 Fri-to-Fri 收益, 但 construct_portfolio 内部
    #    用 monday_open_returns 替代 Y_aligned 计算 weekly_ret
    logging.info("Backtest: 周一开盘执行 (不含周末隔夜)...")
    nav_monday, weights_monday = construct_portfolio(
        Y_aligned, X_aligned, beta, cfg, return_weights=True,
        monday_open_returns=monday_open_returns,
    )
    nav_monday_oos = nav_monday.iloc[oos_start:]
    metrics_monday = compute_metrics(nav_monday_oos)

    # 5. Daily NAV
    logging.info("计算日频 NAV...")
    nav_daily_original = calculate_daily_nav(weights_original, daily_returns, cfg)
    nav_daily_monday = calculate_daily_nav(weights_monday, daily_returns, cfg)

    nav_daily_original_oos = nav_daily_original[nav_daily_original.index >= oos_date]
    nav_daily_monday_oos = nav_daily_monday[nav_daily_monday.index >= oos_date]

    daily_metrics_original = compute_daily_metrics(nav_daily_original_oos)
    daily_metrics_monday = compute_daily_metrics(nav_daily_monday_oos)

    # 6. Print summary
    logging.info("=" * 60)
    logging.info("OOS 指标对比:")
    logging.info("  %-12s  Sharpe=%6.4f  MaxDD=%7.4f  Calmar=%6.4f  Daily_SR=%6.4f  DD_days=%d",
                 "原始", metrics_original["sharpe"], metrics_original["max_dd"],
                 metrics_original["calmar"], daily_metrics_original["sharpe"],
                 daily_metrics_original["max_dd_duration_days"])
    logging.info("  %-12s  Sharpe=%6.4f  MaxDD=%7.4f  Calmar=%6.4f  Daily_SR=%6.4f  DD_days=%d",
                 "周一开盘", metrics_monday["sharpe"], metrics_monday["max_dd"],
                 metrics_monday["calmar"], daily_metrics_monday["sharpe"],
                 daily_metrics_monday["max_dd_duration_days"])

    # 7. Overnight return estimation
    logging.info("=" * 60)
    sr_diff = metrics_monday["sharpe"] - metrics_original["sharpe"]
    logging.info("  Sharpe 差异 (周一开盘 - 原始): %+.4f", sr_diff)
    if metrics_original["sharpe"] > 0:
        pct_change = sr_diff / metrics_original["sharpe"] * 100
        logging.info("  Sharpe 变化率: %+.1f%%", pct_change)

    # 8. Generate charts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # NAV comparison
    fig, ax = plt.subplots(figsize=(16, 8))
    # Normalize at OOS start
    nav_before_orig = nav_daily_original[nav_daily_original.index < oos_date]
    nav_before_mon = nav_daily_monday[nav_daily_monday.index < oos_date]
    norm_orig = nav_before_orig.iloc[-1] if len(nav_before_orig) > 0 else 1.0
    norm_mon = nav_before_mon.iloc[-1] if len(nav_before_mon) > 0 else 1.0

    nav_norm_orig = nav_daily_original / norm_orig
    nav_norm_mon = nav_daily_monday / norm_mon

    # IS (dashed)
    ax.plot(nav_norm_orig[nav_norm_orig.index < oos_date].index,
            nav_norm_orig[nav_norm_orig.index < oos_date].values,
            linestyle="--", color="#2196F3", alpha=0.5, linewidth=0.8)
    ax.plot(nav_norm_mon[nav_norm_mon.index < oos_date].index,
            nav_norm_mon[nav_norm_mon.index < oos_date].values,
            linestyle="--", color="#FF9800", alpha=0.5, linewidth=0.8)
    # OOS (solid)
    ax.plot(nav_norm_orig[nav_norm_orig.index >= oos_date].index,
            nav_norm_orig[nav_norm_orig.index >= oos_date].values,
            linestyle="-", color="#2196F3", linewidth=1.2, label="原始 (含周末隔夜)")
    ax.plot(nav_norm_mon[nav_norm_mon.index >= oos_date].index,
            nav_norm_mon[nav_norm_mon.index >= oos_date].values,
            linestyle="-", color="#FF9800", linewidth=1.2, label="周一开盘 (不含周末隔夜)")

    ax.axvline(oos_date, color="gray", linestyle="--", alpha=0.7)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.3)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.set_title("v7.10 NAV: 原始 vs 周一开盘执行 (IS+OOS, OOS 起点归一化)")
    ax.set_ylabel("NAV (log scale)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "v7_10_monday_open_nav.png", dpi=150)
    plt.close(fig)
    logging.info("  Saved: %s", out_dir / "v7_10_monday_open_nav.png")

    # Drawdown comparison
    fig, ax = plt.subplots(figsize=(16, 5))
    dd_orig = nav_norm_orig / nav_norm_orig.cummax() - 1
    dd_mon = nav_norm_mon / nav_norm_mon.cummax() - 1

    dd_orig_is = dd_orig[dd_orig.index < oos_date]
    dd_orig_oos = dd_orig[dd_orig.index >= oos_date]
    dd_mon_is = dd_mon[dd_mon.index < oos_date]
    dd_mon_oos = dd_mon[dd_mon.index >= oos_date]

    ax.fill_between(dd_orig_is.index, dd_orig_is.values, 0, alpha=0.1, color="#2196F3")
    ax.plot(dd_orig_is.index, dd_orig_is.values, linewidth=0.3, color="#2196F3", alpha=0.5)
    ax.fill_between(dd_orig_oos.index, dd_orig_oos.values, 0, alpha=0.25, color="#2196F3", label="原始")
    ax.plot(dd_orig_oos.index, dd_orig_oos.values, linewidth=0.5, color="#2196F3")

    ax.fill_between(dd_mon_is.index, dd_mon_is.values, 0, alpha=0.1, color="#FF9800")
    ax.plot(dd_mon_is.index, dd_mon_is.values, linewidth=0.3, color="#FF9800", alpha=0.5)
    ax.fill_between(dd_mon_oos.index, dd_mon_oos.values, 0, alpha=0.25, color="#FF9800", label="周一开盘")
    ax.plot(dd_mon_oos.index, dd_mon_oos.values, linewidth=0.5, color="#FF9800")

    ax.axvline(oos_date, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("v7.10 Drawdown: 原始 vs 周一开盘执行")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "v7_10_monday_open_drawdown.png", dpi=150)
    plt.close(fig)
    logging.info("  Saved: %s", out_dir / "v7_10_monday_open_drawdown.png")

    # 9. Write report
    lines = [
        "# v7.10 周一开盘执行测试报告", "",
        "## 实验设置", "",
        f"- 数据: v7.10 (T={T_aligned}, N={N}, K={K})",
        f"- λ_tv={BEST_LAMBDA_TV}, λ_l1={BEST_LAMBDA_L1}",
        f"- IS/OOS: [{MIN_HISTORY}, {oos_start}) / [{oos_start}, {T_aligned})",
        f"- OOS 日期: {oos_date.date()} ~ {Y_aligned.index[T_aligned-1].date()}", "",
        "## 执行模式", "",
        "| 模式 | 信号生成 | 执行时点 | 收益计算 |",
        "|------|----------|----------|----------|",
        "| 原始 | 周五 t 收盘 (β_{t-1}) | 周五 t+1 收盘 | NAV_fri[t+1]/NAV_fri[t] - 1 (含周末隔夜) |",
        "| 周一开盘 | 周五 t 收盘 (β_{t-1}) | 周一 t+1 开盘 | NAV_fri[t+1]/NAV_mon_open[t+1] - 1 (不含周末隔夜) |",
        "",
        "## OOS 指标对比", "",
        "| 指标 | 原始 | 周一开盘 | 差异 |",
        "|------|------|----------|------|",
        f"| 周频 Sharpe | {metrics_original['sharpe']:.4f} | {metrics_monday['sharpe']:.4f} | {metrics_monday['sharpe'] - metrics_original['sharpe']:+.4f} |",
        f"| 周频 MaxDD | {metrics_original['max_dd']:.4f} | {metrics_monday['max_dd']:.4f} | {metrics_monday['max_dd'] - metrics_original['max_dd']:+.4f} |",
        f"| 周频 Calmar | {metrics_original['calmar']:.4f} | {metrics_monday['calmar']:.4f} | {metrics_monday['calmar'] - metrics_original['calmar']:+.4f} |",
        f"| 日频 Sharpe | {daily_metrics_original['sharpe']:.4f} | {daily_metrics_monday['sharpe']:.4f} | {daily_metrics_monday['sharpe'] - daily_metrics_original['sharpe']:+.4f} |",
        f"| 日频 MaxDD | {daily_metrics_original['max_dd']:.4f} | {daily_metrics_monday['max_dd']:.4f} | {daily_metrics_monday['max_dd'] - daily_metrics_original['max_dd']:+.4f} |",
        f"| 回撤持续期 | {daily_metrics_original['max_dd_duration_days']}天 | {daily_metrics_monday['max_dd_duration_days']}天 | {daily_metrics_monday['max_dd_duration_days'] - daily_metrics_original['max_dd_duration_days']:+d}天 |",
        "",
        "## 隔夜收益分析", "",
        f"- Sharpe 变化: {metrics_original['sharpe']:.4f} → {metrics_monday['sharpe']:.4f} ({sr_diff:+.4f})",
        "",
        "## 图表", "",
        "- `v7_10_monday_open_nav.png` — NAV 对比 (IS+OOS)",
        "- `v7_10_monday_open_drawdown.png` — 水下图对比",
    ]

    (out_dir / "v7_10_monday_open_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    logging.info("  报告: %s", out_dir / "v7_10_monday_open_comparison.md")

    logging.info("=" * 60)
    logging.info("完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
