#!/usr/bin/env python3
"""v7.10 step 测试: step=4 vs step=1, 无平滑 vs EMA vs MA.

4 组实验:
  A: step=4, β[t-1], 无平滑 (baseline)
  B: step=1, β[t-1], 无平滑
  C: step=1, β[t-1], EMA α=0.3
  D: step=1, β[t-1], MA w=4

用法:
  python3.10 scripts/v7_10_step_test.py
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
EMA_ALPHA = 0.3
MA_WINDOW = 4


# ============================================================
# Smoothing
# ============================================================
def ema_smooth(beta_arr: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    T, K = beta_arr.shape
    result = np.zeros_like(beta_arr)
    result[0] = beta_arr[0]
    for t in range(1, T):
        result[t] = alpha * beta_arr[t] + (1 - alpha) * result[t - 1]
    return result


def ma_smooth(beta_arr: np.ndarray, window: int = 4) -> np.ndarray:
    T, K = beta_arr.shape
    result = np.zeros_like(beta_arr)
    for t in range(T):
        start = max(0, t - window + 1)
        result[t] = np.mean(beta_arr[start:t + 1], axis=0)
    return result


# ============================================================
# Metrics
# ============================================================
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


def compute_beta_stability(beta: pd.DataFrame) -> float:
    diff = beta.diff().iloc[1:]
    return float(np.sqrt((diff ** 2).sum(axis=1)).mean())


def compute_tv_norm(beta_arr: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(beta_arr, axis=0))))


# ============================================================
# Main
# ============================================================
def main() -> int:
    logging.info("=" * 60)
    logging.info("v7.10 step 测试: step=4 vs step=1 + 平滑")
    logging.info("=" * 60)

    out_dir = REPO / "reports/momentum_etf_rotation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    t0 = time.time()
    X_panel, Y_df, codes = load_v7_10_data()
    T, N, K = X_panel.shape
    daily_returns = load_daily_etf_returns()
    logging.info("  数据加载: X=%s, K=%d, 耗时=%.1fs", X_panel.shape, K, time.time() - t0)

    # OOS split
    oos_start = MIN_HISTORY + int((T - MIN_HISTORY) * 0.6)
    oos_date = Y_df.index[oos_start]
    logging.info("  OOS: [%d, %d) = %s ~ %s", oos_start, T,
                 oos_date.date(), Y_df.index[T - 1].date())

    # 2. Beta estimation
    logging.info("=" * 60)
    logging.info("Beta 估计...")

    # A: step=4 (baseline)
    logging.info("  A: step=4, β[t-1]...")
    t0 = time.time()
    beta_A = expanding_window_tvpr(
        Y_df, X_panel, BEST_LAMBDA_TV, BEST_LAMBDA_L1,
        min_history=MIN_HISTORY, max_iter=200, tol=1e-5, step=4,
    )
    beta_A_arr = beta_A.values
    logging.info("    耗时: %.1fs", time.time() - t0)

    # B/C/D: step=1 (共享 beta_raw)
    logging.info("  B/C/D: step=1, β[t-1]...")
    t0 = time.time()
    beta_B = expanding_window_tvpr(
        Y_df, X_panel, BEST_LAMBDA_TV, BEST_LAMBDA_L1,
        min_history=MIN_HISTORY, max_iter=200, tol=1e-5, step=1,
    )
    beta_B_arr = beta_B.values
    logging.info("    耗时: %.1fs", time.time() - t0)

    # C: EMA 平滑
    beta_C_arr = ema_smooth(beta_B_arr, alpha=EMA_ALPHA)
    # D: MA 平滑
    beta_D_arr = ma_smooth(beta_B_arr, window=MA_WINDOW)

    # 3. Backtest
    logging.info("=" * 60)
    logging.info("OOS backtest + 日频 NAV...")

    cfg = V7_6Config()
    experiments = [
        ("A_step4_raw", beta_A_arr),
        ("B_step1_raw", beta_B_arr),
        ("C_step1_ema", beta_C_arr),
        ("D_step1_ma", beta_D_arr),
    ]

    all_results = {}
    for name, beta_arr in experiments:
        logging.info("  %s ...", name)
        beta_path = pd.DataFrame(beta_arr, index=Y_df.index,
                                  columns=[f"f{i}" for i in range(K)])
        nav_w, weights_df = construct_portfolio(
            Y_df, X_panel, beta_path, cfg, return_weights=True,
            use_latest_beta=False,
        )
        nav_d = calculate_daily_nav(weights_df, daily_returns, cfg)

        nav_w_oos = nav_w.iloc[oos_start:]
        nav_d_oos = nav_d[nav_d.index >= oos_date]
        beta_oos = beta_path.iloc[oos_start:]

        wm = compute_metrics(nav_w_oos)
        dm = compute_daily_metrics(nav_d_oos)
        stab = compute_beta_stability(beta_oos)
        tv = compute_tv_norm(beta_arr)

        # IS metrics
        nav_w_is = nav_w.iloc[MIN_HISTORY:oos_start]
        is_m = compute_metrics(nav_w_is)

        all_results[name] = {
            "nav_weekly": nav_w_oos,
            "nav_daily": nav_d_oos,
            "nav_daily_full": nav_d,
            "nav_weekly_full": nav_w,
            "weights_df": weights_df,
            "beta_arr": beta_arr,
            "metrics": wm,
            "daily_metrics": dm,
            "beta_stability": stab,
            "tv_norm": tv,
            "is_metrics": is_m,
        }

    # 4. Print summary
    logging.info("=" * 60)
    logging.info("OOS 指标对比:")
    for name, r in all_results.items():
        m = r["metrics"]
        dm = r["daily_metrics"]
        is_m = r["is_metrics"]
        sr_decay = (m["sharpe"] - is_m["sharpe"]) / max(abs(is_m["sharpe"]), 1e-9) * 100
        logging.info("  %-16s  周频SR=%6.4f  MaxDD=%7.4f  Calmar=%6.4f  日频SR=%6.4f  DD_days=%3d  "
                     "IS_SR=%6.4f  衰减=%+5.1f%%  TV=%7.1f  稳定=%6.4f",
                     name, m["sharpe"], m["max_dd"], m["calmar"],
                     dm["sharpe"], dm["max_dd_duration_days"],
                     is_m["sharpe"], sr_decay, r["tv_norm"], r["beta_stability"])

    # 5. Charts
    logging.info("=" * 60)
    logging.info("生成图表...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    # NAV comparison
    fig, ax = plt.subplots(figsize=(16, 8))
    for i, (name, r) in enumerate(all_results.items()):
        nav_full = r["nav_daily_full"]
        nav_before = nav_full[nav_full.index < oos_date]
        norm = nav_before.iloc[-1] if len(nav_before) > 0 else 1.0
        nav_norm = nav_full / norm

        is_part = nav_norm[nav_norm.index < oos_date]
        oos_part = nav_norm[nav_norm.index >= oos_date]
        m = r["metrics"]
        ax.plot(is_part.index, is_part.values, "--", color=colors[i], alpha=0.4, linewidth=0.7)
        ax.plot(oos_part.index, oos_part.values, "-", color=colors[i], linewidth=1.2,
                label=f"{name} (SR={m['sharpe']:.2f})")

    ax.axvline(oos_date, color="gray", linestyle="--", alpha=0.7)
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.3)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.set_title("v7.10 Step Test: step=4 vs step=1 + smoothing (IS+OOS)")
    ax.set_ylabel("NAV (log scale, OOS start = 1.0)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "v7_10_step_test_nav.png", dpi=150)
    plt.close(fig)
    logging.info("  Saved: %s", out_dir / "v7_10_step_test_nav.png")

    # Beta path comparison (3 representative factors)
    factor_names = (REPO / "data/high_freq_macro/v7_10_factor_names.csv").read_text().strip().split("\n")[1:]
    # Pick: real_rate (macro), mom_long (PV), rsi (PV)
    pick_names = ["real_rate", "f20_mom_long", "f22_rsi"]
    pick_idx = []
    for pn in pick_names:
        for idx, fn in enumerate(factor_names):
            if pn in fn:
                pick_idx.append(idx)
                break
    if not pick_idx:
        pick_idx = [0, 19, 35]

    fig, axes = plt.subplots(len(pick_idx), 1, figsize=(16, 4 * len(pick_idx)))
    for j, k_idx in enumerate(pick_idx):
        ax = axes[j] if len(pick_idx) > 1 else axes
        fname = factor_names[k_idx] if k_idx < len(factor_names) else f"f{k_idx}"
        dates = Y_df.index

        beta_raw = beta_B_arr[:, k_idx]
        beta_ema = beta_C_arr[:, k_idx]
        beta_ma = beta_D_arr[:, k_idx]
        beta_s4 = beta_A_arr[:, k_idx]

        ax.plot(dates, beta_s4, "-", color="#2196F3", alpha=0.5, linewidth=0.8, label="step=4 raw")
        ax.plot(dates, beta_raw, "-", color="#FF9800", alpha=0.4, linewidth=0.6, label="step=1 raw")
        ax.plot(dates, beta_ema, "-", color="#4CAF50", linewidth=1.2, label=f"step=1 EMA(α={EMA_ALPHA})")
        ax.plot(dates, beta_ma, "-", color="#E91E63", linewidth=1.2, label=f"step=1 MA(w={MA_WINDOW})")

        ax.axvline(oos_date, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"β path: {fname}")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "v7_10_step_test_beta_paths.png", dpi=150)
    plt.close(fig)
    logging.info("  Saved: %s", out_dir / "v7_10_step_test_beta_paths.png")

    # Drawdown
    fig, ax = plt.subplots(figsize=(16, 5))
    for i, (name, r) in enumerate(all_results.items()):
        nav_full = r["nav_daily_full"]
        nav_before = nav_full[nav_full.index < oos_date]
        norm = nav_before.iloc[-1] if len(nav_before) > 0 else 1.0
        nav_norm = nav_full / norm
        dd = nav_norm / nav_norm.cummax() - 1

        is_part = dd[dd.index < oos_date]
        oos_part = dd[dd.index >= oos_date]
        ax.fill_between(is_part.index, is_part.values, 0, alpha=0.08, color=colors[i])
        ax.plot(is_part.index, is_part.values, linewidth=0.3, color=colors[i], alpha=0.4)
        ax.fill_between(oos_part.index, oos_part.values, 0, alpha=0.2, color=colors[i], label=name)
        ax.plot(oos_part.index, oos_part.values, linewidth=0.5, color=colors[i])

    ax.axvline(oos_date, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("v7.10 Step Test: Drawdown (IS+OOS)")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "v7_10_step_test_drawdown.png", dpi=150)
    plt.close(fig)
    logging.info("  Saved: %s", out_dir / "v7_10_step_test_drawdown.png")

    # 6. Report
    lines = [
        "# v7.10 Step 测试报告", "",
        "## 实验设置", "",
        f"- 数据: v7.10 (T={T}, N={N}, K={K})",
        f"- λ_tv={BEST_LAMBDA_TV}, λ_l1={BEST_LAMBDA_L1}",
        f"- IS/OOS: [{MIN_HISTORY}, {oos_start}) / [{oos_start}, {T})",
        f"- OOS 日期: {oos_date.date()} ~ {Y_df.index[T - 1].date()}", "",
        "## 实验配置", "",
        "| 实验 | step | β 时点 | 平滑 | 说明 |",
        "|------|------|--------|------|------|",
        "| A: step=4 raw | 4 | β[t-1] | 无 | 当前 v7.10 baseline |",
        "| B: step=1 raw | 1 | β[t-1] | 无 | 每周重估, 最干净 |",
        f"| C: step=1 EMA | 1 | β[t-1] | EMA α={EMA_ALPHA} | 指数平滑 |",
        f"| D: step=1 MA | 1 | β[t-1] | MA w={MA_WINDOW} | 简单移动平均 |",
        "",
        "## OOS 指标对比", "",
        "| 实验 | 周频SR | MaxDD | Calmar | 日频SR | DD天数 | IS_SR | 衰减% | TV Norm | β稳定性 |",
        "|------|--------|-------|--------|--------|--------|-------|-------|---------|---------|",
    ]

    for name, r in all_results.items():
        m = r["metrics"]
        dm = r["daily_metrics"]
        is_m = r["is_metrics"]
        sr_decay = (m["sharpe"] - is_m["sharpe"]) / max(abs(is_m["sharpe"]), 1e-9) * 100
        lines.append(
            f"| {name} | {m['sharpe']:.4f} | {m['max_dd']:.4f} | {m['calmar']:.4f} | "
            f"{dm['sharpe']:.4f} | {dm['max_dd_duration_days']} | "
            f"{is_m['sharpe']:.4f} | {sr_decay:+.1f}% | {r['tv_norm']:.1f} | {r['beta_stability']:.4f} |"
        )

    lines += [
        "", "## 分析", "",
        "### step 效应 (B vs A)",
    ]
    sr_B = all_results["B_step1_raw"]["metrics"]["sharpe"]
    sr_A = all_results["A_step4_raw"]["metrics"]["sharpe"]
    lines.append(f"- Sharpe: {sr_A:.4f} → {sr_B:.4f} ({sr_B - sr_A:+.4f})")

    lines += ["", "### 平滑效应 (C/D vs B)"]
    sr_C = all_results["C_step1_ema"]["metrics"]["sharpe"]
    sr_D = all_results["D_step1_ma"]["metrics"]["sharpe"]
    lines.append(f"- EMA: {sr_B:.4f} → {sr_C:.4f} ({sr_C - sr_B:+.4f})")
    lines.append(f"- MA:  {sr_B:.4f} → {sr_D:.4f} ({sr_D - sr_B:+.4f})")

    lines += [
        "", "## 图表", "",
        "- `v7_10_step_test_nav.png` — 4 组 NAV 对比 (IS+OOS)",
        "- `v7_10_step_test_beta_paths.png` — β 路径对比 (原始 vs EMA vs MA)",
        "- `v7_10_step_test_drawdown.png` — 水下图",
    ]

    (out_dir / "v7_10_step_test.md").write_text("\n".join(lines), encoding="utf-8")
    logging.info("  报告: %s", out_dir / "v7_10_step_test.md")

    logging.info("=" * 60)
    logging.info("完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
