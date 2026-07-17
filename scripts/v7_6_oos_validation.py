#!/usr/bin/env python3
"""v7.6 OOS 验证实验: Full-sample vs Expanding vs Rolling ADMM.

量化全量样本 ADMM 的前视偏差，对比三种估计方法的 OOS 表现。

用法:
  python3.10 scripts/v7_6_oos_validation.py

输出:
  reports/momentum_etf_rotation/v7_6_oos_validation.csv
  reports/momentum_etf_rotation/v7_6_oos_ic_comparison.csv
  reports/momentum_etf_rotation/v7_6_oos_validation.md
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

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import load_v7_6_data
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import (
    tvpr_admm,
    full_sample_tvpr,
    expanding_window_tvpr,
    rolling_window_tvpr,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config,
    construct_portfolio,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

MIN_HISTORY = 52
FREQ_PER_YEAR = 52
IS_RATIO = 0.6
N_CORES = 20

LAMBDA_GRID = [
    (0.01, 0.01), (0.01, 0.05), (0.05, 0.01),
    (0.05, 0.05), (0.1, 0.05), (0.5, 0.1),
]
ROLLING_WINDOWS = [52, 104, 208]
CV_N_SPLITS = 3
CV_ADMM_MAX_ITER = 50
CV_STEP = 4


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


def compute_beta_stability(beta: pd.DataFrame) -> float:
    diff = beta.diff().iloc[1:]
    return float(np.sqrt((diff ** 2).sum(axis=1)).mean())


def calc_cross_sectional_ic(Y_arr, X_arr, beta_arr, start_idx, factor_names):
    """计算 per-factor IC.

    PV 因子 (截面有变化): 截面 Spearman IC (选股能力)
    宏观因子 (截面无变化): 时间序列 Spearman IC (择时能力)

    对于宏观因子: contrib_k[t] = beta_k[t] × X[t, 0, k] (标量)
    market_ret[t] = mean(Y[t+1, :]) (等权市场收益)
    IC_k = spearmanr(contrib_k, market_ret) (时间序列相关)
    """
    T, N, K = X_arr.shape

    # 检测哪些因子是截面常数 (宏观因子)
    ref_t = max(start_idx, 5)
    macro_k = set()
    for k in range(K):
        cs = X_arr[ref_t, :, k]
        valid_cs = cs[~np.isnan(cs)]
        if len(valid_cs) > 1 and np.std(valid_cs) < 1e-10:
            macro_k.add(k)

    ic_by_factor = {k: [] for k in range(K)}
    ic_total_list = []
    # 宏观因子: 收集时间序列对 (contrib, market_ret)
    macro_ts = {k: [] for k in macro_k}

    for t in range(max(start_idx, 1), T - 1):
        beta_prev = beta_arr[t - 1]
        scores = X_arr[t] @ beta_prev
        y_next = Y_arr[t + 1]
        valid = ~np.isnan(scores) & ~np.isnan(y_next)
        if valid.sum() < 10:
            continue
        ic_total, _ = spearmanr(scores[valid], y_next[valid])
        ic_total_list.append(ic_total)

        # 市场平均收益 (用于宏观因子时间序列 IC)
        y_valid = y_next[~np.isnan(y_next)]
        if len(y_valid) > 0:
            market_ret = float(np.mean(y_valid))
        else:
            market_ret = np.nan

        for k in range(K):
            if k in macro_k:
                # 宏观因子: 时间序列 IC
                contrib_scalar = beta_prev[k] * X_arr[t, 0, k]
                if not np.isnan(contrib_scalar) and not np.isnan(market_ret):
                    macro_ts[k].append((contrib_scalar, market_ret))
            else:
                # PV 因子: 截面 IC
                contrib = beta_prev[k] * X_arr[t, :, k]
                valid_k = ~np.isnan(contrib) & ~np.isnan(y_next)
                if valid_k.sum() >= 10:
                    ic, _ = spearmanr(contrib[valid_k], y_next[valid_k])
                    ic_by_factor[k].append(ic)

    rows = []
    for k in range(K):
        if k in macro_k:
            # 宏观因子: 时间序列 IC
            pairs = macro_ts[k]
            if len(pairs) >= 10:
                contribs, rets = zip(*pairs)
                ic_ts, _ = spearmanr(contribs, rets)
                ics_arr = [ic_ts]  # 单个相关系数
                rows.append(dict(
                    factor_idx=k, factor_name=factor_names[k] if k < len(factor_names) else f"f{k}",
                    IC_type="时间序列",
                    IC_mean=round(float(ic_ts), 4), IC_std=0.0,
                    ICIR=0.0,
                    pct_positive=round(float(1 if ic_ts > 0 else 0), 4),
                    n_obs=len(pairs),
                ))
        else:
            # PV 因子: 截面 IC
            ics = ic_by_factor[k]
            if len(ics) > 0:
                rows.append(dict(
                    factor_idx=k, factor_name=factor_names[k] if k < len(factor_names) else f"f{k}",
                    IC_type="截面",
                    IC_mean=round(float(np.mean(ics)), 4),
                    IC_std=round(float(np.std(ics)), 4),
                    ICIR=round(float(np.mean(ics) / max(np.std(ics), 1e-9)), 4),
                    pct_positive=round(float(np.mean(np.array(ics) > 0)), 4),
                    n_obs=len(ics),
                ))
    if ic_total_list:
        rows.append(dict(
            factor_idx=-1, factor_name="TOTAL",
            IC_type="截面",
            IC_mean=round(float(np.mean(ic_total_list)), 4),
            IC_std=round(float(np.std(ic_total_list)), 4),
            ICIR=round(float(np.mean(ic_total_list) / max(np.std(ic_total_list), 1e-9)), 4),
            pct_positive=round(float(np.mean(np.array(ic_total_list) > 0)), 4),
            n_obs=len(ic_total_list),
        ))
    return pd.DataFrame(rows)


# ============================================================
# Lambda CV (expanding-window, multiprocessing)
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


def select_lambda_mp(Y_arr, X_arr, is_end_idx):
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
    logging.info("  CV 耗时: %.1fs, 最优: (%.3f, %.3f), NAV=%.4f",
                 t1 - t0, best["lambda_tv"], best["lambda_l1"], best["mean_nav"])
    return float(best["lambda_tv"]), float(best["lambda_l1"]), agg


# ============================================================
# OOS beta 估计 (multiprocessing)
# ============================================================
def _estimate_beta_task(args):
    method, Y_arr, X_arr, lambda_tv, lambda_l1, window = args
    Y = pd.DataFrame(Y_arr)
    if method == "full_sample":
        beta = full_sample_tvpr(Y, X_arr, lambda_tv, lambda_l1,
                                min_history=MIN_HISTORY, max_iter=200, tol=1e-5)
    elif method == "expanding":
        beta = expanding_window_tvpr(Y, X_arr, lambda_tv, lambda_l1,
                                     min_history=MIN_HISTORY, max_iter=200, tol=1e-5, step=4)
    elif method == "rolling":
        beta = rolling_window_tvpr(Y, X_arr, lambda_tv, lambda_l1,
                                   window=window, min_history=MIN_HISTORY, max_iter=200, tol=1e-5, step=4)
    else:
        raise ValueError(f"Unknown method: {method}")
    name = f"rolling_{window}" if method == "rolling" else method
    return name, beta.values


def run_backtest(Y, X_panel, beta_path, start_idx):
    cfg = V7_6Config()
    nav, _ = construct_portfolio(Y, X_panel, beta_path, cfg, return_weights=True)
    nav_oos = nav.iloc[start_idx:]
    beta_oos = beta_path.iloc[start_idx:]
    metrics = compute_metrics(nav_oos)
    metrics["beta_stability"] = compute_beta_stability(beta_oos)
    metrics["nav_series"] = nav_oos
    return metrics


def write_report(summary, ic_summary, ic_all, best_lt, best_ll, Y, oos_start, T):
    out_dir = REPO / "reports/momentum_etf_rotation"
    lines = [
        "# v7.6 OOS 验证报告", "",
        "## 实验设置", "",
        f"- 数据: X_panel (T={T}, N={Y.shape[1]}, K=39)",
        f"- IS 区间: [{MIN_HISTORY}, {oos_start}) ({oos_start - MIN_HISTORY} 周)",
        f"- OOS 区间: [{oos_start}, {T}) ({T - oos_start} 周)",
        f"- OOS 日期: {Y.index[oos_start].date()} ~ {Y.index[T - 1].date()}",
        f"- Lambda CV: expanding-window, step=4, max_iter=50, multiprocessing({N_CORES})",
        f"- 最优 lambda: tv={best_lt}, l1={best_ll}", "",
        "## 策略指标对比", "",
        summary.to_markdown(index=False), "",
        "## IC 对比 (整体)", "",
        ic_summary.to_markdown(index=False), "",
    ]
    if not summary.empty:
        best_name = summary.loc[summary["calmar"].idxmax(), "method"]
        best_ic = ic_all[(ic_all["method"] == best_name) & (ic_all["factor_name"] != "TOTAL")]
        best_ic = best_ic.sort_values("IC_mean", ascending=False)
        lines += [f"## Per-Factor IC ({best_name})", "",
                  best_ic[["factor_name", "IC_type", "IC_mean", "IC_std", "ICIR", "pct_positive"]].to_markdown(index=False), ""]
    lines += [
        "## 前视偏差分析", "",
        "| 方法 | 前视? | 说明 |",
        "|------|-------|------|",
        "| full_sample | 是 | 全量 ADMM 平滑器, beta[t] 用到未来数据 |",
        "| expanding | 否 | 递增窗口, beta[t] 只用 [0, t-1], warm-start |",
        "| rolling_52 | 否 | 滑动 1 年窗口 |",
        "| rolling_104 | 否 | 滑动 2 年窗口 |",
        "| rolling_208 | 否 | 滑动 4 年窗口 |", "",
        "## 判断标准", "",
        "| 场景 | Full-sample OOS Sharpe | Expanding OOS Sharpe | 判断 |",
        "|------|----------------------|---------------------|------|",
        "| 前视偏差小 | ~1.0 | ~0.8-1.0 | 策略有效 |",
        "| 前视偏差中 | ~1.0 | ~0.4-0.6 | 策略部分有效 |",
        "| 前视偏差大 | ~1.0 | <0.3 | 策略不可用 |",
    ]
    (out_dir / "v7_6_oos_validation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    logging.info("=" * 60)
    logging.info("v7.6 OOS 验证实验 (multiprocessing, %d cores)", N_CORES)
    logging.info("=" * 60)

    t0 = time.time()
    X_panel, Y_df, codes = load_v7_6_data(macro_use_log_return=True)
    T, N, K = X_panel.shape
    Y_arr = Y_df.values
    logging.info("  X_panel: %s, Y: %s, 耗时: %.1fs", X_panel.shape, Y_df.shape, time.time() - t0)

    is_end = int((T - MIN_HISTORY) * IS_RATIO) + MIN_HISTORY
    oos_start = is_end
    logging.info("  IS: [%d, %d), OOS: [%d, %d)", MIN_HISTORY, is_end, oos_start, T)
    logging.info("  IS 日期: %s ~ %s", Y_df.index[MIN_HISTORY].date(), Y_df.index[is_end - 1].date())
    logging.info("  OOS 日期: %s ~ %s", Y_df.index[oos_start].date(), Y_df.index[T - 1].date())

    # Lambda CV
    logging.info("=" * 60)
    logging.info("Lambda CV (expanding-window, multiprocessing)...")
    best_lt, best_ll, cv_df = select_lambda_mp(Y_arr, X_panel, is_end)
    if not cv_df.empty:
        cv_path = REPO / "reports/momentum_etf_rotation/v7_6_oos_lambda_cv.csv"
        cv_path.parent.mkdir(parents=True, exist_ok=True)
        cv_df.to_csv(cv_path, index=False)

    # Beta estimation (parallel)
    logging.info("=" * 60)
    logging.info("Beta 估计 (并行)...")
    tasks = [
        ("full_sample", Y_arr, X_panel, best_lt, best_ll, 0),
        ("expanding", Y_arr, X_panel, best_lt, best_ll, 0),
    ]
    for W in ROLLING_WINDOWS:
        tasks.append(("rolling", Y_arr, X_panel, best_lt, best_ll, W))

    t0 = time.time()
    with mp.Pool(min(len(tasks), N_CORES)) as pool:
        raw_results = pool.map(_estimate_beta_task, tasks)
    logging.info("  耗时: %.1fs", time.time() - t0)

    results = {}
    for name, beta_arr in raw_results:
        results[name] = pd.DataFrame(beta_arr, index=Y_df.index,
                                     columns=[f"factor_{i}" for i in range(K)])

    # OOS backtest + IC
    logging.info("=" * 60)
    logging.info("OOS backtest + IC...")
    factor_names = pd.read_csv(
        REPO / "data/high_freq_macro/v7_7_factor_names.csv").iloc[:, 0].tolist()

    summary_rows = []
    ic_dict = {}
    for name, beta_path in results.items():
        logging.info("  %s ...", name)
        metrics = run_backtest(Y_df, X_panel, beta_path, oos_start)
        metrics.pop("nav_series")
        metrics["method"] = name
        summary_rows.append(metrics)

        ic_df = calc_cross_sectional_ic(Y_arr, X_panel, beta_path.values, oos_start, factor_names)
        ic_df["method"] = name
        ic_dict[name] = ic_df

    summary = pd.DataFrame(summary_rows)
    summary = summary[["method", "ann_return", "ann_vol", "sharpe", "max_dd", "calmar", "beta_stability"]]
    logging.info("\n策略指标:\n%s", summary.to_string(index=False))

    ic_summary_rows = []
    for name, ic_df in ic_dict.items():
        total_row = ic_df[ic_df["factor_name"] == "TOTAL"]
        if not total_row.empty:
            ic_summary_rows.append({
                "method": name,
                "IC_mean": total_row.iloc[0]["IC_mean"],
                "IC_std": total_row.iloc[0]["IC_std"],
                "ICIR": total_row.iloc[0]["ICIR"],
                "pct_positive": total_row.iloc[0]["pct_positive"],
                "n_significant": int((ic_df[ic_df["factor_name"] != "TOTAL"]["pct_positive"] > 0.55).sum()),
            })
    ic_summary = pd.DataFrame(ic_summary_rows)
    logging.info("\nIC Summary:\n%s", ic_summary.to_string(index=False))

    out_dir = REPO / "reports/momentum_etf_rotation"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "v7_6_oos_validation.csv", index=False)
    ic_all = pd.concat(ic_dict.values(), ignore_index=True)
    ic_all.to_csv(out_dir / "v7_6_oos_ic_comparison.csv", index=False)
    write_report(summary, ic_summary, ic_all, best_lt, best_ll, Y_df, oos_start, T)

    logging.info("=" * 60)
    logging.info("完成! 报告: %s", out_dir / "v7_6_oos_validation.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
