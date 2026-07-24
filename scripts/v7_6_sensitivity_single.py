# coding: utf-8
"""v7.6 Phase 1: 单参数敏感性扫描.

目的: 量化 4 个核心超参数 (lambda_tv, lambda_l1, window_size, rho) 的敏感性.

用法:
   python3.11 scripts/v7_6_sensitivity_single.py

输出:
   reports/momentum_etf_rotation/v7_6_sensitivity_single.csv

总实验数: ~14 (单参数轮换, 减去默认值)
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

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import load_v7_6_data
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import V7_6Config
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import tvpr_estimator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

DAYS_PER_YEAR = 252

START_POINTS = [
    "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01",
]

# 4 个核心参数扫描 (默认值为 lambda_tv=0.05, lambda_l1=0.001, ws=52, rho=1.0)
SCAN = {
    "lambda_tv": [0.005, 0.02, 0.05, 0.07, 0.10, 0.20],
    "lambda_l1": [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.010],
    "window_size": [26, 52, 78, 104, 130, 156],
    "rho": [0.5, 1.0, 2.0, 5.0],
}

# 默认值
DEFAULTS = {
    "lambda_tv": 0.05,
    "lambda_l1": 0.001,
    "window_size": 52,
    "rho": 1.0,
    "min_history": 52,
}

OUTPUT_DIR = REPO / "reports/momentum_etf_rotation"


def compute_metrics(nav: pd.Series, freq: int = DAYS_PER_YEAR) -> dict:
    """计算业绩指标."""
    if nav.empty or len(nav) < 2:
        return {"calmar": 0.0, "ann_return": 0.0, "vol": 0.0, "max_dd": 0.0, "sharpe": 0.0}

    rets = nav.pct_change().dropna()
    if rets.empty:
        return {"calmar": 0.0, "ann_return": 0.0, "vol": 0.0, "max_dd": 0.0, "sharpe": 0.0}

    n_years = len(rets) / freq
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = float((1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1)
    vol = float(rets.std() * np.sqrt(freq))
    cummax = nav.cummax()
    dd = (nav / cummax - 1)
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    sharpe = ann_ret / vol if vol > 0 else 0.0

    return {
        "calmar": round(calmar, 4),
        "ann_return": round(ann_ret, 4),
        "vol": round(vol, 4),
        "max_dd": round(max_dd, 4),
        "sharpe": round(sharpe, 4),
    }


def run_with_overrides(X_panel, Y, valid_codes, **overrides):
    """跑一组参数, 返回完整结果 (含全段, OOS, 起点 CV)."""
    from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import run_v7_6_backtest
    cfg_kwargs = {**DEFAULTS, **overrides}
    cfg = V7_6Config(**cfg_kwargs)

    # 全段
    t0 = time.time()
    nav_weekly, nav_daily = run_v7_6_backtest(X_panel, Y, valid_codes, cfg, return_daily=True)
    full = compute_metrics(nav_daily)

    # OOS
    nav_daily_oos = nav_daily.loc["2022-01-01":]
    oos = compute_metrics(nav_daily_oos)

    # 起点依赖
    start_calmar = []
    for start in START_POINTS:
        mask = Y.index >= start
        Y_start = Y[mask]
        X_start = X_panel[mask]
        if len(Y_start) < cfg.min_history + 12:
            continue
        _, nav_daily_start = run_v7_6_backtest(
            X_start, Y_start, valid_codes, cfg, return_daily=True
        )
        m = compute_metrics(nav_daily_start)
        start_calmar.append(m["calmar"])

    mean_c = float(np.mean(start_calmar)) if start_calmar else 0
    std_c = float(np.std(start_calmar)) if start_calmar else 0
    cv = std_c / mean_c if mean_c > 0 else 0

    return {
        "full_calmar": full["calmar"],
        "full_ann": full["ann_return"],
        "full_vol": full["vol"],
        "full_dd": full["max_dd"],
        "full_sharpe": full["sharpe"],
        "oos_calmar": oos["calmar"],
        "oos_ann": oos["ann_return"],
        "oos_vol": oos["vol"],
        "oos_dd": oos["max_dd"],
        "oos_sharpe": oos["sharpe"],
        "start_mean": round(mean_c, 4),
        "start_std": round(std_c, 4),
        "start_cv": round(cv, 4),
        "start_n": len(start_calmar),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    logging.info("=" * 60)
    logging.info("Phase 1: 单参数敏感性扫描")
    logging.info("=" * 60)

    logging.info("加载数据...")
    t0 = time.time()
    X_panel, Y, valid_codes = load_v7_6_data()
    logging.info("  X_panel: %s, Y: %s, 耗时: %.1fs", X_panel.shape, Y.shape, time.time() - t0)

    rows = []

    # 1. 默认值作为基线
    logging.info("=" * 60)
    logging.info("[默认参数] λ_tv=0.05, λ_l1=0.001, ws=52, rho=1.0")
    base = run_with_overrides(X_panel, Y, valid_codes)
    base_row = {
        "param": "default", "value": "baseline",
        "lambda_tv": DEFAULTS["lambda_tv"], "lambda_l1": DEFAULTS["lambda_l1"],
        "window_size": DEFAULTS["window_size"], "rho": DEFAULTS["rho"],
        **base,
    }
    rows.append(base_row)
    logging.info("  OOS Calmar=%.4f, CV%%=%.1f%%", base["oos_calmar"], base["start_cv"] * 100)

    # 2. 单参数扫描
    for param_name, values in SCAN.items():
        for val in values:
            # 跳过默认值
            if val == DEFAULTS[param_name]:
                continue

            logging.info("=" * 60)
            logging.info("[%s=%s] 其他参数为默认值", param_name, val)

            overrides = {param_name: val}
            try:
                r = run_with_overrides(X_panel, Y, valid_codes, **overrides)
            except Exception as e:
                logging.error("  失败: %s", e)
                r = {"oos_calmar": 0, "start_cv": 999, "oos_sharpe": 0, "oos_dd": 0,
                     "oos_ann": 0, "full_calmar": 0, "full_sharpe": 0, "full_dd": 0,
                     "full_ann": 0, "full_vol": 0, "oos_vol": 0,
                     "start_mean": 0, "start_std": 0, "start_n": 0, "seconds": 0}

            row = {
                "param": param_name, "value": val,
                "lambda_tv": DEFAULTS["lambda_tv"], "lambda_l1": DEFAULTS["lambda_l1"],
                "window_size": DEFAULTS["window_size"], "rho": DEFAULTS["rho"],
                **r,
            }
            rows.append(row)
            logging.info("  OOS Calmar=%.4f, CV%%=%.1f%%, %.1fs",
                         r["oos_calmar"], r["start_cv"] * 100, r["seconds"])

            # 增量保存 (防止超时丢数据)
            try:
                incremental_path = OUTPUT_DIR / "v7_6_sensitivity_single_incremental.csv"
                pd.DataFrame(rows).to_csv(incremental_path, index=False)
            except Exception as e:
                logging.warning("  增量保存失败: %s", e)

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "v7_6_sensitivity_single.csv"
    df.to_csv(out_path, index=False)
    logging.info("=" * 60)
    logging.info("结果已保存: %s", out_path)

    # 输出每个参数的最优
    print("\n" + "=" * 80)
    print("Phase 1 结果汇总 (按参数分组)")
    print("=" * 80)
    for param_name in SCAN:
        sub = df[df["param"].isin([param_name, "default"])].copy()
        print(f"\n## {param_name}:")
        cols = ["value", "oos_calmar", "oos_sharpe", "oos_dd", "start_cv"]
        print(sub[cols].to_string(index=False))

    # 全表
    print("\n" + "=" * 80)
    print("全量结果:")
    print("=" * 80)
    cols = ["param", "value", "oos_calmar", "oos_sharpe", "oos_dd", "oos_ann", "start_cv", "seconds"]
    print(df[cols].to_string(index=False))

    # 分析
    base_oos = base["oos_calmar"]
    base_cv = base["start_cv"]
    print("\n" + "=" * 80)
    print(f"基线: OOS Calmar={base_oos:.4f}, 起点 CV%={base_cv:.1%}")
    print("=" * 80)
    other = df[df["param"] != "default"]
    if len(other) > 0:
        max_degradation = (base_oos - other["oos_calmar"].max()) / base_oos if base_oos > 0 else 0
        min_oos = other["oos_calmar"].min()
        avg_oos = other["oos_calmar"].mean()
        std_oos = other["oos_calmar"].std()
        print(f"其他组 OOS Calmar: min={min_oos:.4f}, max={other['oos_calmar'].max():.4f}, "
              f"mean={avg_oos:.4f}, std={std_oos:.4f}")
        print(f"最低退化: {(base_oos - min_oos) / base_oos * 100:+.1f}%")
        if max_degradation > 0.5:
            print("🔴 严重参数敏感 (>50% 退化)")
        elif max_degradation > 0.3:
            print("🟡 中度参数敏感 (30-50% 退化)")
        else:
            print("🟢 低参数敏感 (<30% 退化)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
