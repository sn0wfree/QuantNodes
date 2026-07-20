#!/usr/bin/env python3
"""v7.10 β 时点测试: 对比 β[t-1] vs β[t].

简化测试: 收益仍用 Fri-to-Fri (Y), 只改 β 时点.

用法:
  python3.10 scripts/v7_10_beta_timing_test.py

输出:
  reports/momentum_etf_rotation/v7_10_beta_timing.md
  reports/momentum_etf_rotation/v7_10_beta_timing_nav.png
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
    logging.info("v7.10 β 时点测试: β[t-1] vs β[t]")
    logging.info("=" * 60)

    out_dir = REPO / "reports/momentum_etf_rotation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    t0 = time.time()
    X_panel, Y_df, codes = load_v7_10_data()
    T, N, K = X_panel.shape
    daily_returns = load_daily_etf_returns()
    logging.info("  数据加载: X=%s, K=%d, 耗时=%.1fs", X_panel.shape, K, time.time() - t0)

    # 2. Beta estimation
    logging.info("=" * 60)
    logging.info("Beta 估计 (expanding window)...")
    t0 = time.time()
    beta = expanding_window_tvpr(
        Y_df, X_panel,
        BEST_LAMBDA_TV, BEST_LAMBDA_L1,
        min_history=MIN_HISTORY, max_iter=200, tol=1e-5, step=4,
    )
    logging.info("  Beta 估计耗时: %.1fs", time.time() - t0)

    # 3. OOS split
    oos_start = MIN_HISTORY + int((T - MIN_HISTORY) * 0.6)
    oos_date = Y_df.index[oos_start]
    logging.info("  OOS: [%d, %d) = %s ~ %s", oos_start, T,
                 oos_date.date(), Y_df.index[T-1].date())

    # 4. Backtest: A = β[t-1] (baseline)
    logging.info("=" * 60)
    logging.info("A: β[t-1] (baseline)...")
    cfg = V7_6Config()
    nav_A, weights_A = construct_portfolio(
        Y_df, X_panel, beta, cfg, return_weights=True, use_latest_beta=False,
    )
    nav_A_oos = nav_A.iloc[oos_start:]
    metrics_A = compute_metrics(nav_A_oos)

    # 5. Backtest: B = β[t] (latest)
    logging.info("B: β[t] (latest)...")
    nav_B, weights_B = construct_portfolio(
        Y_df, X_panel, beta, cfg, return_weights=True, use_latest_beta=True,
    )
    nav_B_oos = nav_B.iloc[oos_start:]
    metrics_B = compute_metrics(nav_B_oos)

    # 6. Daily NAV
    logging.info("计算日频 NAV...")
    nav_daily_A = calculate_daily_nav(weights_A, daily_returns, cfg)
    nav_daily_B = calculate_daily_nav(weights_B, daily_returns, cfg)
    nav_daily_A_oos = nav_daily_A[nav_daily_A.index >= oos_date]
    nav_daily_B_oos = nav_daily_B[nav_daily_B.index >= oos_date]
    daily_A = compute_daily_metrics(nav_daily_A_oos)
    daily_B = compute_daily_metrics(nav_daily_B_oos)

    # 7. IS metrics
    nav_A_is = nav_A.iloc[MIN_HISTORY:oos_start]
    nav_B_is = nav_B.iloc[MIN_HISTORY:oos_start]
    is_A = compute_metrics(nav_A_is)
    is_B = compute_metrics(nav_B_is)

    # 8. Print summary
    logging.info("=" * 60)
    logging.info("OOS 指标对比:")
    logging.info("  %-12s  周频SR=%6.4f  MaxDD=%7.4f  Calmar=%6.4f  日频SR=%6.4f  DD_days=%d",
                 "A:β[t-1]", metrics_A["sharpe"], metrics_A["max_dd"],
                 metrics_A["calmar"], daily_A["sharpe"], daily_A["max_dd_duration_days"])
    logging.info("  %-12s  周频SR=%6.4f  MaxDD=%7.4f  Calmar=%6.4f  日频SR=%6.4f  DD_days=%d",
                 "B:β[t]", metrics_B["sharpe"], metrics_B["max_dd"],
                 metrics_B["calmar"], daily_B["sharpe"], daily_B["max_dd_duration_days"])

    sr_diff = metrics_B["sharpe"] - metrics_A["sharpe"]
    logging.info("  Sharpe 差异 (B-A): %+.4f (%.1f%%)", sr_diff,
                 sr_diff / max(abs(metrics_A["sharpe"]), 1e-9) * 100)

    # 9. Chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(16, 8))
    norm_A = nav_daily_A[nav_daily_A.index < oos_date].iloc[-1]
    norm_B = nav_daily_B[nav_daily_B.index < oos_date].iloc[-1]
    nav_norm_A = nav_daily_A / norm_A
    nav_norm_B = nav_daily_B / norm_B

    # IS (dashed)
    is_A_dates = nav_norm_A[nav_norm_A.index < oos_date]
    is_B_dates = nav_norm_B[nav_norm_B.index < oos_date]
    ax.plot(is_A_dates.index, is_A_dates.values, "--", color="#2196F3", alpha=0.5, linewidth=0.8)
    ax.plot(is_B_dates.index, is_B_dates.values, "--", color="#FF9800", alpha=0.5, linewidth=0.8)
    # OOS (solid)
    oos_A_dates = nav_norm_A[nav_norm_A.index >= oos_date]
    oos_B_dates = nav_norm_B[nav_norm_B.index >= oos_date]
    ax.plot(oos_A_dates.index, oos_A_dates.values, "-", color="#2196F3", linewidth=1.2,
            label=f"A: β[t-1] (SR={metrics_A['sharpe']:.2f})")
    ax.plot(oos_B_dates.index, oos_B_dates.values, "-", color="#FF9800", linewidth=1.2,
            label=f"B: β[t] (SR={metrics_B['sharpe']:.2f})")

    ax.axvline(oos_date, color="gray", linestyle="--", alpha=0.7)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.3)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.set_title("v7.10 β Timing: β[t-1] vs β[t] (IS+OOS, OOS start normalized)")
    ax.set_ylabel("NAV (log scale)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "v7_10_beta_timing_nav.png", dpi=150)
    plt.close(fig)
    logging.info("  Saved: %s", out_dir / "v7_10_beta_timing_nav.png")

    # 10. Report
    lines = [
        "# v7.10 β 时点测试报告", "",
        "## 实验设置", "",
        f"- 数据: v7.10 (T={T}, N={N}, K={K})",
        f"- λ_tv={BEST_LAMBDA_TV}, λ_l1={BEST_LAMBDA_L1}",
        f"- 收益: Fri-to-Fri (Y, NAV 收益)",
        f"- IS/OOS: [{MIN_HISTORY}, {oos_start}) / [{oos_start}, {T})",
        f"- OOS 日期: {oos_date.date()} ~ {Y_df.index[T-1].date()}", "",
        "## 时点对比", "",
        "| 模式 | β 来源 | 信号生成 | 合法性 |",
        "|------|--------|----------|--------|",
        "| A: β[t-1] | Y[:t-1] | X[t] · β[t-1] | ✅ 保守, 信息滞后 1 周 |",
        "| B: β[t] | Y[:t] | X[t] · β[t] | ✅ 更及时, 用了本周收益 |",
        "",
        "## IS vs OOS 指标对比", "",
        "| 模式 | IS Sharpe | OOS Sharpe | SR 衰减% | IS MaxDD | OOS MaxDD |",
        "|------|-----------|------------|----------|----------|-----------|",
        f"| A: β[t-1] | {is_A['sharpe']:.4f} | {metrics_A['sharpe']:.4f} | "
        f"{(metrics_A['sharpe'] - is_A['sharpe']) / max(abs(is_A['sharpe']), 1e-9) * 100:+.1f}% | "
        f"{is_A['max_dd']:.4f} | {metrics_A['max_dd']:.4f} |",
        f"| B: β[t] | {is_B['sharpe']:.4f} | {metrics_B['sharpe']:.4f} | "
        f"{(metrics_B['sharpe'] - is_B['sharpe']) / max(abs(is_B['sharpe']), 1e-9) * 100:+.1f}% | "
        f"{is_B['max_dd']:.4f} | {metrics_B['max_dd']:.4f} |",
        "",
        "## OOS 指标对比", "",
        "| 指标 | A: β[t-1] | B: β[t] | 差异 |",
        "|------|-----------|---------|------|",
        f"| 周频 Sharpe | {metrics_A['sharpe']:.4f} | {metrics_B['sharpe']:.4f} | {sr_diff:+.4f} |",
        f"| 周频 MaxDD | {metrics_A['max_dd']:.4f} | {metrics_B['max_dd']:.4f} | {metrics_B['max_dd'] - metrics_A['max_dd']:+.4f} |",
        f"| 周频 Calmar | {metrics_A['calmar']:.4f} | {metrics_B['calmar']:.4f} | {metrics_B['calmar'] - metrics_A['calmar']:+.4f} |",
        f"| 日频 Sharpe | {daily_A['sharpe']:.4f} | {daily_B['sharpe']:.4f} | {daily_B['sharpe'] - daily_A['sharpe']:+.4f} |",
        f"| 日频 MaxDD | {daily_A['max_dd']:.4f} | {daily_B['max_dd']:.4f} | {daily_B['max_dd'] - daily_A['max_dd']:+.4f} |",
        f"| 回撤持续期 | {daily_A['max_dd_duration_days']}天 | {daily_B['max_dd_duration_days']}天 | {daily_B['max_dd_duration_days'] - daily_A['max_dd_duration_days']:+d}天 |",
        "",
        "## 结论", "",
    ]
    if sr_diff > 0.05:
        lines.append(f"β[t] 显著优于 β[t-1] (Sharpe +{sr_diff:.4f}), 建议采用.")
    elif sr_diff < -0.05:
        lines.append(f"β[t] 显著劣于 β[t-1] (Sharpe {sr_diff:.4f}), 保持原方案.")
    else:
        lines.append(f"两者差异不大 (Sharpe {sr_diff:+.4f}), 保持 β[t-1] 更保守.")

    lines += ["", "## 图表", "", "- `v7_10_beta_timing_nav.png` — NAV 对比 (IS+OOS)"]
    (out_dir / "v7_10_beta_timing.md").write_text("\n".join(lines), encoding="utf-8")
    logging.info("  报告: %s", out_dir / "v7_10_beta_timing.md")

    logging.info("=" * 60)
    logging.info("完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
