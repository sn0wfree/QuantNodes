#!/usr/bin/env python3
"""v7.10 OOS 验证: 混合标准化 + 超参数重置.

对比 3 组实验:
  A: v7.9 raw (baseline, 复现)
  B: v7.10 标准化 + 旧 λ (0.05, 0.01)
  C: v7.10 标准化 + CV 新网格 (200 组合)

用法:
  python3.10 scripts/v7_10_oos_validation.py

输出:
  reports/momentum_etf_rotation/v7_10_oos_validation.md
  reports/momentum_etf_rotation/v7_10_oos_nav_curves.png
  reports/momentum_etf_rotation/v7_10_oos_drawdown.png
  reports/momentum_etf_rotation/v7_10_oos_rolling_sharpe.png
  reports/momentum_etf_rotation/v7_10_condition_number.png
  reports/momentum_etf_rotation/v7_10_factor_scales.png
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_v7_9_data,
    load_v7_10_data,
    load_daily_etf_returns,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import (
    tvpr_admm,
    full_sample_tvpr,
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
MACRO_K = 17

# Stage 1: 粗搜 10 组合
LAMBDA_GRID_COARSE = [
    (0.01, 0.01), (0.01, 0.03), (0.01, 0.05),
    (0.03, 0.01), (0.03, 0.03), (0.03, 0.05),
    (0.05, 0.01), (0.05, 0.03), (0.05, 0.05), (0.05, 0.10),
]

# v7.9 baseline 网格 (仅 6 个组合)
LAMBDA_GRID_OLD = [
    (0.01, 0.01), (0.01, 0.05), (0.05, 0.01),
    (0.05, 0.05), (0.1, 0.05), (0.5, 0.1),
]

CV_N_SPLITS = 3
CV_ADMM_MAX_ITER = 50
CV_STEP = 4


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


def compute_condition_number(X_panel: np.ndarray) -> float:
    """计算设计矩阵条件数."""
    T, N, K = X_panel.shape
    X_flat = X_panel.reshape(T * N, K)
    valid_rows = ~np.any(np.isnan(X_flat), axis=1)
    X_clean = X_flat[valid_rows]
    return float(np.linalg.cond(X_clean.T @ X_clean))


def count_effective_factors(beta_arr: np.ndarray, threshold: float = 0.001) -> float:
    """统计 |β| > threshold 的因子平均个数."""
    return float(np.mean(np.sum(np.abs(beta_arr) > threshold, axis=1)))


def compute_tv_norm(beta_arr: np.ndarray) -> float:
    """Σ|Δβ| 总变差."""
    return float(np.sum(np.abs(np.diff(beta_arr, axis=0))))


# ============================================================
# Lambda CV
# ============================================================
def _expanding_sparse_last_beta(Y_arr, X_arr, lambda_tv, lambda_l1,
                                min_history, step, max_iter, tol):
    T_train = Y_arr.shape[0]
    K = X_arr.shape[2]
    beta_warm = None
    beta_last = np.zeros(K)
    for t in range(min_history, T_train, step):
        if beta_warm is not None:
            beta_init = np.zeros((t, K))
            beta_init[:beta_warm.shape[0]] = beta_warm
        else:
            beta_init = None
        beta_path = tvpr_admm(Y_arr[:t], X_arr[:t], lambda_tv, lambda_l1,
                              rho=1.0, max_iter=max_iter, tol=tol, beta_init=beta_init)
        beta_last = beta_path[-1]
        beta_warm = beta_path
    return beta_last


def _cv_fold_task(args):
    lt, ll, train_end, val_end, Y_arr, X_arr, min_h, step, max_iter, tol = args
    beta_last = _expanding_sparse_last_beta(
        Y_arr[:train_end], X_arr[:train_end], lt, ll, min_h, step, max_iter, tol)
    Y_val = Y_arr[train_end:val_end]
    X_val = X_arr[train_end:val_end]
    nav = 1.0
    for t in range(1, len(Y_val)):
        scores = X_val[t] @ beta_last
        valid = ~np.isnan(scores) & ~np.isnan(Y_val[t])
        if valid.sum() == 0:
            continue
        sv = scores[valid]
        top_idx = np.argsort(sv)[-10:]
        chosen = np.where(valid)[0][top_idx]
        ret = float(np.nanmean(Y_val[t][chosen]))
        nav *= (1 + (ret if not np.isnan(ret) else 0.0))
    return {"lambda_tv": lt, "lambda_l1": ll, "train_end": train_end, "nav": nav}


def _make_fine_grid(best_lt: float, best_ll: float) -> list[tuple[float, float]]:
    """Stage 2: 在最优附近 ±0.01 范围生成细网格."""
    tv_range = np.arange(max(0.005, best_lt - 0.01), best_lt + 0.015, 0.005)
    ll_range = np.arange(max(0.005, best_ll - 0.01), best_ll + 0.015, 0.005)
    grid = [(round(lt, 4), round(ll, 4)) for lt in tv_range for ll in ll_range]
    logging.info("  细网格: TV=[%.3f~%.3f], L1=[%.3f~%.3f], %d 组合",
                 tv_range[0], tv_range[-1], ll_range[0], ll_range[-1], len(grid))
    return grid


def select_lambda_mp(Y_arr, X_arr, is_end_idx, lambda_grid, label=""):
    fold_size = (is_end_idx - MIN_HISTORY) // (CV_N_SPLITS + 1)
    if fold_size < 3:
        return 0.05, 0.01, pd.DataFrame()
    tasks = []
    for lt, ll in lambda_grid:
        for i in range(CV_N_SPLITS):
            train_end = MIN_HISTORY + (i + 1) * fold_size
            val_end = min(train_end + fold_size, is_end_idx)
            tasks.append((lt, ll, train_end, val_end,
                          Y_arr, X_arr, MIN_HISTORY, CV_STEP, CV_ADMM_MAX_ITER, 1e-4))
    t0 = time.time()
    with mp.Pool(N_CORES) as pool:
        results = pool.map(_cv_fold_task, tasks)
    t1 = time.time()
    df = pd.DataFrame(results)
    agg = df.groupby(["lambda_tv", "lambda_l1"])["nav"].mean().reset_index()
    agg.columns = ["lambda_tv", "lambda_l1", "mean_nav"]
    best = agg.loc[agg["mean_nav"].idxmax()]
    logging.info("  CV [%s] 耗时: %.1fs, 最优: (%.3f, %.3f), NAV=%.4f",
                 label, t1 - t0, best["lambda_tv"], best["lambda_l1"], best["mean_nav"])
    return float(best["lambda_tv"]), float(best["lambda_l1"]), agg


def select_lambda_two_stage(Y_arr, X_arr, is_end_idx, label="C"):
    """两阶段 CV: Stage1 粗搜 10 组合 → Stage2 细搜 ~25 组合."""
    logging.info("  Stage 1: 粗搜 %d 组合...", len(LAMBDA_GRID_COARSE))
    best_lt, best_ll, agg_coarse = select_lambda_mp(
        Y_arr, X_arr, is_end_idx, LAMBDA_GRID_COARSE, label=f"{label} Stage1")

    fine_grid = _make_fine_grid(best_lt, best_ll)
    logging.info("  Stage 2: 细搜 %d 组合...", len(fine_grid))
    best_lt2, best_ll2, agg_fine = select_lambda_mp(
        Y_arr, X_arr, is_end_idx, fine_grid, label=f"{label} Stage2")

    agg_all = pd.concat([agg_coarse, agg_fine], ignore_index=True)
    agg_all = agg_all.groupby(["lambda_tv", "lambda_l1"])["mean_nav"].mean().reset_index()
    return best_lt2, best_ll2, agg_all


# ============================================================
# Beta Estimation
# ============================================================
def _estimate_beta_task(args):
    method, Y_arr, X_arr, lambda_tv, lambda_l1 = args
    Y = pd.DataFrame(Y_arr)
    if method == "full_sample":
        beta = full_sample_tvpr(Y, X_arr, lambda_tv, lambda_l1,
                                min_history=MIN_HISTORY, max_iter=200, tol=1e-5)
    elif method == "expanding":
        beta = expanding_window_tvpr(Y, X_arr, lambda_tv, lambda_l1,
                                     min_history=MIN_HISTORY, max_iter=200, tol=1e-5, step=4)
    else:
        raise ValueError(f"Unknown method: {method}")
    return method, beta.values


def estimate_betas(Y_arr, X_panel, lambda_tv, lambda_l1, methods):
    tasks = [(m, Y_arr, X_panel, lambda_tv, lambda_l1) for m in methods]
    t0 = time.time()
    with mp.Pool(min(len(tasks), N_CORES)) as pool:
        raw_results = pool.map(_estimate_beta_task, tasks)
    logging.info("  Beta 估计耗时: %.1fs", time.time() - t0)
    return dict(raw_results)


# ============================================================
# Backtest
# ============================================================
def run_backtest(Y, X_panel, beta_arr, start_idx, daily_returns=None):
    cfg = V7_6Config()
    beta_path = pd.DataFrame(beta_arr, index=Y.index,
                              columns=[f"factor_{i}" for i in range(beta_arr.shape[1])])
    nav, weights_df = construct_portfolio(Y, X_panel, beta_path, cfg, return_weights=True)

    # IS/OOS 分别计算
    nav_is = nav.iloc[MIN_HISTORY:start_idx]
    nav_oos = nav.iloc[start_idx:]
    beta_oos = beta_path.iloc[start_idx:]

    metrics = compute_metrics(nav_oos)
    metrics["is_metrics"] = compute_metrics(nav_is)
    metrics["beta_stability"] = compute_beta_stability(beta_oos)
    metrics["nav_weekly"] = nav_oos
    metrics["nav_weekly_full"] = nav  # 完整 NAV (用于 IS+OOS 图表)
    metrics["weights_df"] = weights_df
    metrics["beta_arr"] = beta_arr

    if daily_returns is not None:
        nav_daily = calculate_daily_nav(weights_df, daily_returns, cfg)
        oos_date = Y.index[start_idx]
        nav_daily_oos = nav_daily[nav_daily.index >= oos_date]
        metrics["nav_daily"] = nav_daily_oos
        metrics["nav_daily_full"] = nav_daily  # 完整日频 NAV
        metrics["daily_metrics"] = compute_daily_metrics(nav_daily_oos)

    return metrics


# ============================================================
# Plotting
# ============================================================
def plot_condition_number(cond_raw, cond_std, out_path):
    """图1: 条件数对比."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(["v7.9 Raw", "v7.10 Standardized"], [cond_raw, cond_std],
                  color=["#e74c3c", "#2ecc71"], edgecolor="black")
    ax.set_yscale("log")
    ax.set_ylabel("Condition Number (log scale)")
    ax.set_title("Design Matrix Condition Number: Raw vs Standardized")
    for bar, val in zip(bars, [cond_raw, cond_std]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.2e}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info("  Saved: %s", out_path)


def plot_factor_scales(X_raw, X_std, factor_names, out_path):
    """图2: 因子尺度分布对比 (箱线图)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, N, K = X_raw.shape
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Raw
    ax = axes[0]
    data_raw = []
    for k in range(K):
        vals = X_raw[:, :, k].ravel()
        valid = vals[~np.isnan(vals)]
        data_raw.append(valid)
    ax.boxplot(data_raw, labels=[f"k{k}" for k in range(K)], showfliers=False)
    ax.set_title("v7.9 Raw Factor Scales")
    ax.set_ylabel("Value")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.grid(True, alpha=0.3)

    # Standardized
    ax = axes[1]
    data_std = []
    for k in range(K):
        vals = X_std[:, :, k].ravel()
        valid = vals[~np.isnan(vals)]
        data_std.append(valid)
    ax.boxplot(data_std, labels=[f"k{k}" for k in range(K)], showfliers=False)
    ax.set_title("v7.10 Standardized Factor Scales")
    ax.set_ylabel("Value (Z-score)")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info("  Saved: %s", out_path)


def _get_full_daily_nav(metrics, Y_df, start_idx, daily_returns, cfg):
    """获取完整区间 (IS+OOS) 的日频 NAV，OOS 起点归一化为 1.0."""
    nav_daily = metrics.get("nav_daily_full")
    if nav_daily is None or len(nav_daily) < 10:
        return None
    # 归一化: OOS 起点 = 1.0
    oos_date = Y_df.index[start_idx]
    nav_before = nav_daily[nav_daily.index < oos_date]
    if len(nav_before) > 0:
        norm_val = nav_before.iloc[-1]
    else:
        norm_val = nav_daily.iloc[0]
    return nav_daily / norm_val


def plot_nav_curves(results_dict, oos_date, out_path, Y_df, start_idx, daily_returns):
    """日频 NAV 曲线对比 (IS+OOS，OOS 起点归一化为 1.0)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = V7_6Config()
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    for i, (name, metrics) in enumerate(results_dict.items()):
        nav_full = _get_full_daily_nav(metrics, Y_df, start_idx, daily_returns, cfg)
        if nav_full is None or len(nav_full) < 10:
            continue
        color = colors[i % len(colors)]
        # IS 区间: 虚线
        nav_is = nav_full[nav_full.index < oos_date]
        if len(nav_is) > 1:
            ax.plot(nav_is.index, nav_is.values, linestyle="--", color=color,
                    alpha=0.5, linewidth=0.8)
        # OOS 区间: 实线
        nav_oos = nav_full[nav_full.index >= oos_date]
        if len(nav_oos) > 1:
            ax.plot(nav_oos.index, nav_oos.values, linestyle="-", color=color,
                    linewidth=1.2, label=name)

    ax.axvline(oos_date, color="gray", linestyle="--", alpha=0.7, label="OOS start")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.3)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.set_title("v7.10 Daily NAV (IS + OOS, normalized at OOS start = 1.0)")
    ax.set_ylabel("NAV (log scale, OOS start = 1.0)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info("  Saved: %s", out_path)


def plot_drawdown(results_dict, oos_date, out_path, Y_df, start_idx, daily_returns):
    """水下图 (IS+OOS，OOS 起点归一化)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = V7_6Config()
    fig, ax = plt.subplots(figsize=(16, 5))
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    for i, (name, metrics) in enumerate(results_dict.items()):
        nav_full = _get_full_daily_nav(metrics, Y_df, start_idx, daily_returns, cfg)
        if nav_full is None or len(nav_full) < 10:
            continue
        color = colors[i % len(colors)]
        dd = nav_full / nav_full.cummax() - 1
        # IS 区间: 浅色填充
        dd_is = dd[dd.index < oos_date]
        if len(dd_is) > 1:
            ax.fill_between(dd_is.index, dd_is.values, 0, alpha=0.1, color=color)
            ax.plot(dd_is.index, dd_is.values, linewidth=0.3, color=color, alpha=0.5)
        # OOS 区间: 实线填充
        dd_oos = dd[dd.index >= oos_date]
        if len(dd_oos) > 1:
            ax.fill_between(dd_oos.index, dd_oos.values, 0, alpha=0.25, color=color, label=name)
            ax.plot(dd_oos.index, dd_oos.values, linewidth=0.5, color=color)

    ax.axvline(oos_date, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("v7.10 Drawdown (IS + OOS, normalized at OOS start)")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info("  Saved: %s", out_path)


def plot_rolling_sharpe(results_dict, oos_date, out_path):
    """图5: 滚动 1 年 Sharpe."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    for name, metrics in results_dict.items():
        nav_d = metrics.get("nav_daily")
        if nav_d is not None and len(nav_d) > 252:
            rets = nav_d.pct_change()
            rolling_sr = rets.rolling(252).mean() / rets.rolling(252).std() * np.sqrt(252)
            ax.plot(rolling_sr.index, rolling_sr.values, label=name, linewidth=0.8)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(oos_date, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("v7.10 OOS Rolling 1Y Sharpe Ratio")
    ax.set_ylabel("Sharpe")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info("  Saved: %s", out_path)


# ============================================================
# Report
# ============================================================
def _compute_is_oos_metrics(metrics):
    """从 metrics 中获取 IS 和 OOS 指标."""
    is_m = metrics.get("is_metrics", {})
    oos_m = compute_metrics(metrics["nav_weekly"])
    return is_m, oos_m


def write_report(all_results, diag, out_dir, Y_df, oos_start, T, K, factor_names):
    lines = [
        "# v7.10 OOS 验证报告", "",
        "## 实验设置", "",
        f"- 数据: v7.10 (T={T}, N={Y_df.shape[1]}, K={K})",
        f"- 标准化: 宏观=时间序列Z-score, PV=截面Z-score + Winsorize",
        f"- 条件数: {diag['cond_raw']:.2e} → {diag['cond_std']:.2e}",
        f"- IS 区间: [{MIN_HISTORY}, {oos_start}) ({oos_start - MIN_HISTORY} 周)",
        f"- OOS 区间: [{oos_start}, {T}) ({T - oos_start} 周)",
        f"- OOS 日期: {Y_df.index[oos_start].date()} ~ {Y_df.index[T - 1].date()}", "",
    ]

    # Diagnostic metrics
    lines += ["## 诊断指标", "",
              "| 指标 | v7.9 raw | v7.10 标准化 |",
              "|------|----------|-------------|",
              f"| 条件数 | {diag['cond_raw']:.2e} | {diag['cond_std']:.2e} |",
              f"| 有效因子数 (expanding) | {diag['eff_factors_raw']:.1f} | {diag['eff_factors_std']:.1f} |",
              f"| Beta TV Norm (expanding) | {diag['tv_norm_raw']:.2f} | {diag['tv_norm_std']:.2f} |",
              f"| Beta 稳定性 (expanding) | {diag['beta_stab_raw']:.4f} | {diag['beta_stab_std']:.4f} |",
              ""]

    # IS vs OOS 指标对比
    lines += ["## IS vs OOS 指标对比 (过拟合诊断)", ""]
    is_oos_rows = []
    for name, m in all_results.items():
        is_m, oos_m = _compute_is_oos_metrics(m)
        sr_decay = (oos_m["sharpe"] - is_m.get("sharpe", 0)) / max(abs(is_m.get("sharpe", 1e-9)), 1e-9) * 100
        is_oos_rows.append({
            "method": name,
            "IS_sharpe": is_m.get("sharpe", 0),
            "OOS_sharpe": oos_m["sharpe"],
            "SR_decay_%": round(sr_decay, 1),
            "IS_maxdd": is_m.get("max_dd", 0),
            "OOS_maxdd": oos_m["max_dd"],
            "IS_calmar": is_m.get("calmar", 0),
            "OOS_calmar": oos_m["calmar"],
        })
    is_oos_df = pd.DataFrame(is_oos_rows)
    lines += [is_oos_df.to_markdown(index=False), "",
              "> SR_decay_% = (OOS - IS) / |IS| × 100, 越接近 0 越好（负值表示衰减）", ""]

    # Weekly metrics table
    lines += ["## 策略指标对比 (OOS)", ""]
    weekly_rows = []
    for name, m in all_results.items():
        wm = compute_metrics(m["nav_weekly"])
        wm["method"] = name
        weekly_rows.append(wm)
    weekly_df = pd.DataFrame(weekly_rows)
    weekly_df = weekly_df[["method", "ann_return", "ann_vol", "sharpe", "max_dd", "calmar"]]
    lines += [weekly_df.to_markdown(index=False), "", "## 日频指标对比 (OOS)", ""]

    # Daily metrics table
    daily_rows = []
    for name, m in all_results.items():
        dm = m.get("daily_metrics", {})
        dm["method"] = name
        daily_rows.append(dm)
    daily_df = pd.DataFrame(daily_rows)
    if not daily_df.empty and "method" in daily_df.columns:
        cols = ["method"] + [c for c in daily_df.columns if c != "method"]
        daily_df = daily_df[cols]
        lines += [daily_df.to_markdown(index=False), ""]

    # Lambda CV results
    lines += ["## Lambda CV 结果", ""]
    if "C_std_cv" in diag:
        cv_df = diag["C_std_cv"]
        lines += ["### v7.10 标准化 + 两阶段 CV (Top 10)", ""]
        cv_top = cv_df.sort_values("mean_nav", ascending=False).head(10)
        lines += [cv_top.to_markdown(index=False), ""]

    # Per-factor IC
    best_name = [n for n in all_results if "C" in n]
    if best_name:
        ic_df = all_results[best_name[0]].get("ic_df")
        if ic_df is not None:
            ic_pv = ic_df[(ic_df["factor_name"] != "TOTAL") & (ic_df["IC_type"] == "截面")]
            ic_pv = ic_pv.sort_values("IC_mean", ascending=False)
            ic_macro = ic_df[(ic_df["factor_name"] != "TOTAL") & (ic_df["IC_type"] == "时间序列")]
            ic_macro = ic_macro.sort_values("IC_mean", ascending=False, key=abs)
            lines += [f"## Per-Factor IC ({best_name[0]})", "",
                      "### PV 因子 (截面 IC)", "",
                      ic_pv[["factor_name", "IC_mean", "IC_std", "ICIR", "pct_positive"]].to_markdown(index=False), "",
                      "### 宏观因子 (时间序列 IC)", "",
                      ic_macro[["factor_name", "IC_mean", "n_obs"]].to_markdown(index=False), ""]

    # Charts
    lines += ["## 图表", "",
              "- `v7_10_condition_number.png` — 条件数对比",
              "- `v7_10_factor_scales.png` — 因子尺度分布",
              "- `v7_10_oos_nav_curves.png` — 日频 NAV 曲线 (IS+OOS, OOS 起点归一化)",
              "- `v7_10_oos_drawdown.png` — 水下图 (IS+OOS)",
              "- `v7_10_oos_rolling_sharpe.png` — 滚动 Sharpe"]

    (out_dir / "v7_10_oos_validation.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# IC Calculation
# ============================================================
def calc_factor_ic(Y_arr, X_arr, beta_arr, start_idx, factor_names):
    T, N, K = X_arr.shape
    ref_t = max(start_idx, 5)
    macro_k_set = set()
    for k in range(K):
        cs = X_arr[ref_t, :, k]
        valid_cs = cs[~np.isnan(cs)]
        if len(valid_cs) > 1 and np.std(valid_cs) < 1e-10:
            macro_k_set.add(k)

    ic_by_factor = {k: [] for k in range(K)}
    ic_total_list = []
    macro_ts = {k: [] for k in macro_k_set}

    for t in range(max(start_idx, 1), T - 1):
        beta_prev = beta_arr[t - 1]
        scores = X_arr[t] @ beta_prev
        y_next = Y_arr[t + 1]
        valid = ~np.isnan(scores) & ~np.isnan(y_next)
        if valid.sum() < 10:
            continue
        ic_total, _ = spearmanr(scores[valid], y_next[valid])
        ic_total_list.append(ic_total)

        y_valid = y_next[~np.isnan(y_next)]
        market_ret = float(np.mean(y_valid)) if len(y_valid) > 0 else np.nan

        for k in range(K):
            if k in macro_k_set:
                contrib_scalar = beta_prev[k] * X_arr[t, 0, k]
                if not np.isnan(contrib_scalar) and not np.isnan(market_ret):
                    macro_ts[k].append((contrib_scalar, market_ret))
            else:
                contrib = beta_prev[k] * X_arr[t, :, k]
                valid_k = ~np.isnan(contrib) & ~np.isnan(y_next)
                if valid_k.sum() >= 10:
                    ic, _ = spearmanr(contrib[valid_k], y_next[valid_k])
                    ic_by_factor[k].append(ic)

    rows = []
    for k in range(K):
        fname = factor_names[k] if k < len(factor_names) else f"f{k}"
        if k in macro_k_set:
            pairs = macro_ts[k]
            if len(pairs) >= 10:
                contribs, rets = zip(*pairs)
                ic_ts, _ = spearmanr(contribs, rets)
                rows.append(dict(
                    factor_idx=k, factor_name=fname, IC_type="时间序列",
                    IC_mean=round(float(ic_ts), 4), IC_std=0.0, ICIR=0.0,
                    pct_positive=round(float(1 if ic_ts > 0 else 0), 4),
                    n_obs=len(pairs),
                ))
        else:
            ics = ic_by_factor[k]
            if len(ics) > 0:
                rows.append(dict(
                    factor_idx=k, factor_name=fname, IC_type="截面",
                    IC_mean=round(float(np.mean(ics)), 4),
                    IC_std=round(float(np.std(ics)), 4),
                    ICIR=round(float(np.mean(ics) / max(np.std(ics), 1e-9)), 4),
                    pct_positive=round(float(np.mean(np.array(ics) > 0)), 4),
                    n_obs=len(ics),
                ))
    if ic_total_list:
        rows.append(dict(
            factor_idx=-1, factor_name="TOTAL", IC_type="截面",
            IC_mean=round(float(np.mean(ic_total_list)), 4),
            IC_std=round(float(np.std(ic_total_list)), 4),
            ICIR=round(float(np.mean(ic_total_list) / max(np.std(ic_total_list), 1e-9)), 4),
            pct_positive=round(float(np.mean(np.array(ic_total_list) > 0)), 4),
            n_obs=len(ic_total_list),
        ))
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================
def main() -> int:
    logging.info("=" * 60)
    logging.info("v7.10 OOS 验证: 混合标准化 + 超参数重置")
    logging.info("=" * 60)

    out_dir = REPO / "reports/momentum_etf_rotation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    t0 = time.time()
    X_raw, Y_df, codes = load_v7_9_data()
    X_std, _, _ = load_v7_10_data()
    T, N, K = X_raw.shape
    Y_arr = Y_df.values
    factor_names = (REPO / "data/high_freq_macro/v7_9_factor_names.csv").read_text().strip().split("\n")[1:]
    daily_returns = load_daily_etf_returns()
    logging.info("  数据加载: X=%s, K=%d, 耗时=%.1fs", X_raw.shape, K, time.time() - t0)

    # 2. Diagnostics
    logging.info("=" * 60)
    logging.info("诊断指标...")
    cond_raw = compute_condition_number(X_raw)
    cond_std = compute_condition_number(X_std)
    logging.info("  条件数: %.2e → %.2e", cond_raw, cond_std)

    # IS/OOS split
    is_end = int((T - MIN_HISTORY) * 0.6) + MIN_HISTORY
    oos_start = is_end
    oos_date = Y_df.index[oos_start]
    logging.info("  IS: [%d, %d), OOS: [%d, %d) = %s ~ %s",
                 MIN_HISTORY, is_end, oos_start, T,
                 Y_df.index[MIN_HISTORY].date(), Y_df.index[T-1].date())

    # 3. Lambda CV
    logging.info("=" * 60)
    logging.info("Lambda CV...")

    # A: v7.9 raw + old grid (baseline)
    best_lt_A, best_ll_A, _ = select_lambda_mp(Y_arr, X_raw, is_end, LAMBDA_GRID_OLD, label="A: v7.9 raw")

    # B: v7.10 std + old grid
    best_lt_B, best_ll_B, _ = select_lambda_mp(Y_arr, X_std, is_end, LAMBDA_GRID_OLD, label="B: v7.10 std+old")

    # C: v7.10 std + 两阶段 CV (粗搜10 → 细搜~25)
    best_lt_C, best_ll_C, cv_df_C = select_lambda_two_stage(Y_arr, X_std, is_end, label="C")

    # 4. Beta estimation
    logging.info("=" * 60)
    logging.info("Beta 估计 (3 组实验)...")

    betas_A = estimate_betas(Y_arr, X_raw, best_lt_A, best_ll_A, ["expanding"])
    betas_B = estimate_betas(Y_arr, X_std, best_lt_B, best_ll_B, ["expanding"])
    betas_C = estimate_betas(Y_arr, X_std, best_lt_C, best_ll_C, ["expanding"])

    # 5. OOS backtest
    logging.info("=" * 60)
    logging.info("OOS backtest + 日频 NAV + IC...")

    all_results = {}
    experiments = [
        ("A_v79_raw", X_raw, betas_A["expanding"]),
        ("B_v710_std_oldλ", X_std, betas_B["expanding"]),
        ("C_v710_std_newλ", X_std, betas_C["expanding"]),
    ]

    for name, X_used, beta_arr in experiments:
        logging.info("  %s ...", name)
        metrics = run_backtest(Y_df, X_used, beta_arr, oos_start, daily_returns=daily_returns)
        ic_df = calc_factor_ic(Y_arr, X_used, beta_arr, oos_start, factor_names)
        metrics["ic_df"] = ic_df
        all_results[name] = metrics

    # 6. Compute diagnostics for best experiment
    beta_raw = betas_A["expanding"]
    beta_std = betas_C["expanding"]
    diag = {
        "cond_raw": cond_raw,
        "cond_std": cond_std,
        "eff_factors_raw": count_effective_factors(beta_raw),
        "eff_factors_std": count_effective_factors(beta_std),
        "tv_norm_raw": compute_tv_norm(beta_raw),
        "tv_norm_std": compute_tv_norm(beta_std),
        "beta_stab_raw": compute_beta_stability(pd.DataFrame(beta_raw)),
        "beta_stab_std": compute_beta_stability(pd.DataFrame(beta_std)),
        "C_std_cv": cv_df_C,
    }

    # 7. Print summary
    logging.info("=" * 60)
    summary_rows = []
    for name, m in all_results.items():
        wm = compute_metrics(m["nav_weekly"])
        dm = m.get("daily_metrics", {})
        summary_rows.append({
            "method": name,
            "weekly_sharpe": wm["sharpe"],
            "weekly_calmar": wm["calmar"],
            "weekly_maxdd": wm["max_dd"],
            "daily_sharpe": dm.get("sharpe", 0),
            "daily_maxdd": dm.get("max_dd", 0),
            "daily_dd_days": dm.get("max_dd_duration_days", 0),
            "beta_stab": m["beta_stability"],
            "eff_factors": count_effective_factors(m["beta_arr"]),
            "tv_norm": compute_tv_norm(m["beta_arr"]),
        })
    summary = pd.DataFrame(summary_rows)
    logging.info("\n%s", summary.to_string(index=False))

    # 8. Generate charts
    logging.info("=" * 60)
    logging.info("生成图表...")
    plot_condition_number(cond_raw, cond_std, out_dir / "v7_10_condition_number.png")
    plot_factor_scales(X_raw, X_std, factor_names, out_dir / "v7_10_factor_scales.png")
    plot_nav_curves(all_results, oos_date, out_dir / "v7_10_oos_nav_curves.png",
                    Y_df, oos_start, daily_returns)
    plot_drawdown(all_results, oos_date, out_dir / "v7_10_oos_drawdown.png",
                  Y_df, oos_start, daily_returns)
    plot_rolling_sharpe(all_results, oos_date, out_dir / "v7_10_oos_rolling_sharpe.png")

    # 9. Write report
    write_report(all_results, diag, out_dir, Y_df, oos_start, T, K, factor_names)

    logging.info("=" * 60)
    logging.info("完成! 报告: %s", out_dir / "v7_10_oos_validation.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
