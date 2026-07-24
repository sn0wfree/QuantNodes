#!/usr/bin/env python3
"""v7.9 OOS 验证: 因子去重 + RollingSymmetry 正交化 + 日频 NAV + 图表.

对比:
  A: v7.9 原始 (36 因子, 无正交化)
  B: v7.9 + RollingSymmetry 正交化
  每组各跑 expanding + full_sample

用法:
  python3.10 scripts/v7_9_oos_validation.py

输出:
  reports/momentum_etf_rotation/v7_9_oos_validation.md
  reports/momentum_etf_rotation/v7_9_oos_nav_curves.png
  reports/momentum_etf_rotation/v7_9_oos_drawdown.png
  reports/momentum_etf_rotation/v7_9_oos_rolling_sharpe.png
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
MACRO_K = 17  # 宏观因子数量 (k=0..16)

LAMBDA_GRID = [
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
    """日频指标: 年化收益/波动/Sharpe/MaxDD/回撤持续期."""
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
    # 最大回撤持续期 (交易日)
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


# ============================================================
# RollingSymmetry 正交化
# ============================================================
def orthogonalize_pv_factors(X_panel: np.ndarray) -> np.ndarray:
    """对 PV 因子做截面 Symmetry 正交 (每个时间点独立).

    宏观因子 (k < MACRO_K) 不变.
    PV 因子 (k >= MACRO_K): 在每个截面上做 Symmetry 白化, 使截面 cov = I.
    """
    T, N, K = X_panel.shape
    K_pv = K - MACRO_K
    X_out = X_panel.copy()

    for t in range(T):
        X_pv = X_panel[t, :, MACRO_K:]  # (N, K_pv)
        valid = ~np.isnan(X_pv).any(axis=1)
        if valid.sum() < K_pv + 1:
            continue

        X_pv_valid = X_pv[valid]  # (N_valid, K_pv)
        cov = np.cov(X_pv_valid, rowvar=False)  # (K_pv, K_pv)
        if cov.ndim < 2:
            continue

        D, U = np.linalg.eigh(cov)
        D = np.maximum(D, 1e-8)
        S = U @ np.diag(D ** -0.5) @ U.T  # (K_pv, K_pv)

        X_pv_orth = X_pv_valid @ S  # (N_valid, K_pv)
        X_out[t, :, MACRO_K:] = np.nan
        X_out[t, valid, MACRO_K:] = X_pv_orth

    return X_out


# ============================================================
# IC Calculation
# ============================================================
def calc_factor_ic(Y_arr, X_arr, beta_arr, start_idx, factor_names):
    """计算 per-factor IC (PV=截面, 宏观=时间序列)."""
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


def select_lambda_mp(Y_arr, X_arr, is_end_idx, label=""):
    fold_size = (is_end_idx - MIN_HISTORY) // (CV_N_SPLITS + 1)
    if fold_size < 3:
        return 0.05, 0.01, pd.DataFrame()
    tasks = []
    for lt, ll in LAMBDA_GRID:
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
    """并行估计 beta (expanding + full_sample)."""
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
    """运行回测, 返回指标 + NAV + weights."""
    cfg = V7_6Config()
    beta_path = pd.DataFrame(beta_arr, index=Y.index,
                              columns=[f"factor_{i}" for i in range(beta_arr.shape[1])])
    nav, weights_df = construct_portfolio(Y, X_panel, beta_path, cfg, return_weights=True)
    nav_oos = nav.iloc[start_idx:]
    beta_oos = beta_path.iloc[start_idx:]

    metrics = compute_metrics(nav_oos)
    metrics["beta_stability"] = compute_beta_stability(beta_oos)
    metrics["nav_weekly"] = nav_oos
    metrics["weights_df"] = weights_df

    if daily_returns is not None:
        nav_daily = calculate_daily_nav(weights_df, daily_returns, cfg)
        # 找到 OOS 起点对应的日频日期
        oos_date = Y.index[start_idx]
        nav_daily_oos = nav_daily[nav_daily.index >= oos_date]
        metrics["nav_daily"] = nav_daily_oos
        metrics["daily_metrics"] = compute_daily_metrics(nav_daily_oos)

    return metrics


# ============================================================
# Plotting
# ============================================================
def plot_nav_curves(results_dict, oos_date, out_path):
    """图1: 日频 NAV 曲线对比 (对数坐标)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 7))
    for name, metrics in results_dict.items():
        nav_d = metrics.get("nav_daily")
        if nav_d is not None and len(nav_d) > 10:
            ax.plot(nav_d.index, nav_d.values, label=name, linewidth=1)
    ax.set_yscale("log")
    ax.axvline(oos_date, color="gray", linestyle="--", alpha=0.7, label="OOS start")
    ax.legend(fontsize=9)
    ax.set_title("v7.9 OOS Daily NAV Comparison")
    ax.set_ylabel("NAV (log scale)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info("  Saved: %s", out_path)


def plot_drawdown(results_dict, oos_date, out_path):
    """图2: 水下图 (Underwater Plot)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    for name, metrics in results_dict.items():
        nav_d = metrics.get("nav_daily")
        if nav_d is not None and len(nav_d) > 10:
            dd = nav_d / nav_d.cummax() - 1
            ax.fill_between(dd.index, dd.values, 0, alpha=0.25, label=name)
            ax.plot(dd.index, dd.values, linewidth=0.5)
    ax.axvline(oos_date, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("v7.9 OOS Drawdown (Underwater)")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logging.info("  Saved: %s", out_path)


def plot_rolling_sharpe(results_dict, oos_date, out_path):
    """图3: 滚动 1 年 Sharpe."""
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
    ax.set_title("v7.9 OOS Rolling 1Y Sharpe Ratio")
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
def write_report(all_results, out_dir, Y_df, oos_start, T, K, best_lt_raw, best_ll_raw,
                 best_lt_orth, best_ll_orth, factor_names):
    lines = [
        "# v7.9 OOS 验证报告", "",
        "## 实验设置", "",
        f"- 数据: v7.9 (T={T}, N={Y_df.shape[1]}, K={K})",
        f"- 因子去重: f4_vol_vol, f9_pv_corr, f21_reversal (39→36)",
        f"- Log 变换: f3_amt_vol, f6_ls_total, f7_ls_change, f12_amihud, f22_rsi",
        f"- IS 区间: [{MIN_HISTORY}, {oos_start}) ({oos_start - MIN_HISTORY} 周)",
        f"- OOS 区间: [{oos_start}, {T}) ({T - oos_start} 周)",
        f"- OOS 日期: {Y_df.index[oos_start].date()} ~ {Y_df.index[T - 1].date()}",
        f"- Lambda CV (raw): tv={best_lt_raw}, l1={best_ll_raw}",
        f"- Lambda CV (orthogonal): tv={best_lt_orth}, l1={best_ll_orth}", "",
        "## 策略指标对比", "",
    ]

    # Weekly metrics table
    weekly_rows = []
    for name, m in all_results.items():
        wm = compute_metrics(m["nav_weekly"])
        wm["method"] = name
        weekly_rows.append(wm)
    weekly_df = pd.DataFrame(weekly_rows)
    weekly_df = weekly_df[["method", "ann_return", "ann_vol", "sharpe", "max_dd", "calmar"]]
    lines += [weekly_df.to_markdown(index=False), "", "## 日频指标对比", ""]

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

    # Per-factor IC (best expanding method)
    best_expanding = [n for n in all_results if "expanding" in n]
    if best_expanding:
        best_name = best_expanding[0]
        ic_df = all_results[best_name].get("ic_df")
        if ic_df is not None:
            ic_pv = ic_df[(ic_df["factor_name"] != "TOTAL") & (ic_df["IC_type"] == "截面")]
            ic_pv = ic_pv.sort_values("IC_mean", ascending=False)
            ic_macro = ic_df[(ic_df["factor_name"] != "TOTAL") & (ic_df["IC_type"] == "时间序列")]
            ic_macro = ic_macro.sort_values("IC_mean", ascending=False, key=abs)
            lines += [f"## Per-Factor IC ({best_name})", "",
                      "### PV 因子 (截面 IC)", "",
                      ic_pv[["factor_name", "IC_mean", "IC_std", "ICIR", "pct_positive"]].to_markdown(index=False), "",
                      "### 宏观因子 (时间序列 IC)", "",
                      ic_macro[["factor_name", "IC_mean", "n_obs"]].to_markdown(index=False), ""]

    # Orthogonalization effect
    lines += ["## 正交化效果对比", ""]
    for method_type in ["expanding", "full_sample"]:
        raw_name = f"{method_type}_raw"
        orth_name = f"{method_type}_orth"
        if raw_name in all_results and orth_name in all_results:
            raw_m = compute_metrics(all_results[raw_name]["nav_weekly"])
            orth_m = compute_metrics(all_results[orth_name]["nav_weekly"])
            lines += [
                f"### {method_type}", "",
                "| 指标 | 原始 | 正交化 | 变化 |",
                "|------|------|--------|------|",
                f"| Sharpe | {raw_m['sharpe']:.4f} | {orth_m['sharpe']:.4f} | {orth_m['sharpe']-raw_m['sharpe']:+.4f} |",
                f"| Calmar | {raw_m['calmar']:.4f} | {orth_m['calmar']:.4f} | {orth_m['calmar']-raw_m['calmar']:+.4f} |",
                f"| MaxDD | {raw_m['max_dd']:.4f} | {orth_m['max_dd']:.4f} | {orth_m['max_dd']-raw_m['max_dd']:+.4f} |",
                f"| AnnReturn | {raw_m['ann_return']:.4f} | {orth_m['ann_return']:.4f} | {orth_m['ann_return']-raw_m['ann_return']:+.4f} |",
                "",
            ]

    lines += [
        "## 图表", "",
        "- `v7_9_oos_nav_curves.png` — 日频 NAV 曲线对比",
        "- `v7_9_oos_drawdown.png` — 水下图 (回撤)",
        "- `v7_9_oos_rolling_sharpe.png` — 滚动 1 年 Sharpe",
    ]

    (out_dir / "v7_9_oos_validation.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# Main
# ============================================================
def main() -> int:
    logging.info("=" * 60)
    logging.info("v7.9 OOS 验证 (去重 + 正交化 + 日频 NAV)")
    logging.info("=" * 60)

    out_dir = REPO / "reports/momentum_etf_rotation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    t0 = time.time()
    X_panel, Y_df, codes = load_v7_9_data()
    T, N, K = X_panel.shape
    Y_arr = Y_df.values
    factor_names = (REPO / "data/high_freq_macro/v7_9_factor_names.csv").read_text().strip().split("\n")[1:]
    logging.info("  数据加载: X=%s, Y=%s, K=%d, 耗时=%.1fs", X_panel.shape, Y_df.shape, K, time.time() - t0)

    # Load daily returns
    daily_returns = load_daily_etf_returns()

    # IS/OOS split
    is_end = int((T - MIN_HISTORY) * 0.6) + MIN_HISTORY
    oos_start = is_end
    oos_date = Y_df.index[oos_start]
    logging.info("  IS: [%d, %d), OOS: [%d, %d) = %s ~ %s",
                 MIN_HISTORY, is_end, oos_start, T,
                 Y_df.index[MIN_HISTORY].date(), Y_df.index[T-1].date())

    # 2. Create orthogonalized version
    logging.info("=" * 60)
    logging.info("RollingSymmetry 正交化...")
    t0 = time.time()
    X_panel_orth = orthogonalize_pv_factors(X_panel)
    logging.info("  正交化耗时: %.1fs", time.time() - t0)

    # 3. Lambda CV for both raw and orthogonal
    logging.info("=" * 60)
    logging.info("Lambda CV...")
    best_lt_raw, best_ll_raw, _ = select_lambda_mp(Y_arr, X_panel, is_end, label="raw")
    best_lt_orth, best_ll_orth, _ = select_lambda_mp(Y_arr, X_panel_orth, is_end, label="orthogonal")

    # 4. Beta estimation (4 experiments in parallel)
    logging.info("=" * 60)
    logging.info("Beta 估计 (4 组实验)...")

    # Raw: expanding + full_sample
    betas_raw = estimate_betas(Y_arr, X_panel, best_lt_raw, best_ll_raw,
                               ["expanding", "full_sample"])
    # Orthogonal: expanding + full_sample
    betas_orth = estimate_betas(Y_arr, X_panel_orth, best_lt_orth, best_ll_orth,
                                ["expanding", "full_sample"])

    # 5. OOS backtest + IC
    logging.info("=" * 60)
    logging.info("OOS backtest + 日频 NAV + IC...")

    all_results = {}
    experiments = [
        ("expanding_raw", X_panel, betas_raw["expanding"]),
        ("full_sample_raw", X_panel, betas_raw["full_sample"]),
        ("expanding_orth", X_panel_orth, betas_orth["expanding"]),
        ("full_sample_orth", X_panel_orth, betas_orth["full_sample"]),
    ]

    for name, X_used, beta_arr in experiments:
        logging.info("  %s ...", name)
        metrics = run_backtest(Y_df, X_used, beta_arr, oos_start, daily_returns=daily_returns)
        ic_df = calc_factor_ic(Y_arr, X_used, beta_arr, oos_start, factor_names)
        metrics["ic_df"] = ic_df
        all_results[name] = metrics

    # 6. Print summary
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
        })
    summary = pd.DataFrame(summary_rows)
    logging.info("\n%s", summary.to_string(index=False))

    # 7. Generate charts
    logging.info("=" * 60)
    logging.info("生成图表...")
    chart_results = {n: m for n, m in all_results.items()}
    plot_nav_curves(chart_results, oos_date, out_dir / "v7_9_oos_nav_curves.png")
    plot_drawdown(chart_results, oos_date, out_dir / "v7_9_oos_drawdown.png")
    plot_rolling_sharpe(chart_results, oos_date, out_dir / "v7_9_oos_rolling_sharpe.png")

    # 8. Write report
    write_report(all_results, out_dir, Y_df, oos_start, T, K,
                 best_lt_raw, best_ll_raw, best_lt_orth, best_ll_orth, factor_names)

    logging.info("=" * 60)
    logging.info("完成! 报告: %s", out_dir / "v7_9_oos_validation.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
