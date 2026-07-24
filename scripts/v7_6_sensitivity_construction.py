# coding: utf-8
"""v7.6 Phase 5: 构造层扰动 (top_n, max_weight, vol_window).

目的: 看 v7.6 对非核心构造参数的敏感性.

用法:
   python3.11 scripts/v7_6_sensitivity_construction.py

输出:
   reports/momentum_etf_rotation/v7_6_sensitivity_construction.csv
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
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config,
    run_v7_6_backtest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

DAYS_PER_YEAR = 252

START_POINTS = [
    "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01",
]

# 3 个构造参数扫描
SCAN = {
    "top_n": [5, 8, 10, 15, 20],
    "max_weight": [0.15, 0.20, 0.25, 0.30],
    "vol_window": [13, 26, 52],
}

DEFAULTS = {
    "lambda_tv": 0.05,
    "lambda_l1": 0.001,
    "window_size": 52,
    "top_n": 10,
    "max_weight": 0.25,
    "vol_window": 26,
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
    return {"calmar": round(calmar, 4), "ann_return": round(ann_ret, 4),
            "vol": round(vol, 4), "max_dd": round(max_dd, 4), "sharpe": round(sharpe, 4)}


def run_with_construction_overrides(X_panel, Y, valid_codes, **overrides):
    """跑一组构造参数."""
    cfg_kwargs = {**DEFAULTS, **overrides}
    cfg = V7_6Config(**cfg_kwargs)

    t0 = time.time()
    nav_weekly, nav_daily = run_v7_6_backtest(X_panel, Y, valid_codes, cfg, return_daily=True)
    full = compute_metrics(nav_daily)
    nav_daily_oos = nav_daily.loc["2022-01-01":]
    oos = compute_metrics(nav_daily_oos)

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
        "oos_calmar": oos["calmar"],
        "oos_sharpe": oos["sharpe"],
        "oos_dd": oos["max_dd"],
        "oos_ann": oos["ann_return"],
        "start_mean": round(mean_c, 4),
        "start_cv": round(cv, 4),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    logging.info("=" * 60)
    logging.info("Phase 5: 构造层扰动")
    logging.info("=" * 60)

    X_panel, Y, valid_codes = load_v7_6_data()
    logging.info("  X_panel: %s, Y: %s", X_panel.shape, Y.shape)

    rows = []

    # baseline
    logging.info("=" * 60)
    logging.info("[baseline] top_n=10, max_weight=0.25, vol_window=26")
    base = run_with_construction_overrides(X_panel, Y, valid_codes)
    base_row = {
        "param": "default", "value": "baseline",
        "top_n": DEFAULTS["top_n"], "max_weight": DEFAULTS["max_weight"],
        "vol_window": DEFAULTS["vol_window"],
        **base,
    }
    rows.append(base_row)
    base_oos = base["oos_calmar"]
    logging.info("  baseline OOS Calmar=%.4f", base_oos)

    # 单参数扫描
    for param_name, values in SCAN.items():
        for val in values:
            if val == DEFAULTS[param_name]:
                continue
            logging.info("=" * 60)
            logging.info("[%s=%s]", param_name, val)

            overrides = {param_name: val}
            try:
                r = run_with_construction_overrides(X_panel, Y, valid_codes, **overrides)
            except Exception as e:
                logging.error("  失败: %s", e)
                r = {"oos_calmar": 0, "oos_sharpe": 0, "oos_dd": 0, "oos_ann": 0,
                     "full_calmar": 0, "start_mean": 0, "start_cv": 999, "seconds": 0}

            row = {
                "param": param_name, "value": val,
                "top_n": DEFAULTS["top_n"], "max_weight": DEFAULTS["max_weight"],
                "vol_window": DEFAULTS["vol_window"],
                **r,
            }
            rows.append(row)
            degradation = (base_oos - r["oos_calmar"]) / base_oos * 100 if base_oos > 0 else 0
            logging.info("  OOS Calmar=%.4f (退化 %.1f%%), CV%%=%.1f%%",
                         r["oos_calmar"], degradation, r["start_cv"] * 100)

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "v7_6_sensitivity_construction.csv"
    df.to_csv(out_path, index=False)
    logging.info("=" * 60)
    logging.info("结果已保存: %s", out_path)

    # 输出
    print("\n" + "=" * 80)
    print("Phase 5 构造层结果")
    print("=" * 80)
    for param_name in SCAN:
        sub = df[df["param"].isin([param_name, "default"])].copy()
        print(f"\n## {param_name}:")
        cols = ["value", "oos_calmar", "oos_sharpe", "oos_dd", "start_cv"]
        print(sub[cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
