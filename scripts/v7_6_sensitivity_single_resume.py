# coding: utf-8
"""v7.6 Phase 1 续跑: rho 参数 + 历史已跑结果合并.

目的: Phase 1 超时, 只跑了 rho=0.5 还需跑 rho=2.0, 5.0.
      这里从历史日志提取已跑结果, 直接跑剩余参数.

用法:
   python3.11 scripts/v7_6_sensitivity_single_resume.py

输出:
   reports/momentum_etf_rotation/v7_6_sensitivity_single.csv (合并)
"""
from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

DAYS_PER_YEAR = 252
START_POINTS = [
    "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01",
]
DEFAULTS = {
    "lambda_tv": 0.05,
    "lambda_l1": 0.001,
    "window_size": 52,
    "rho": 1.0,
    "min_history": 52,
}
OUTPUT_DIR = REPO / "reports/momentum_etf_rotation"

# 历史已跑结果 (从 v7_6_sens1_v2.log 提取)
HISTORICAL = [
    # (param, value, oos_calmar, oos_sharpe, oos_dd, oos_ann, start_cv)
    ("default", "baseline", 1.8913, 1.68, -0.1594, 0.3015, 0.487),  # 来自 Phase 0
    ("lambda_tv", 0.005, 1.4022, None, None, None, 0.530),
    ("lambda_tv", 0.02, 2.1182, None, None, None, 0.486),
    ("lambda_tv", 0.07, 2.1952, None, None, None, 0.488),
    ("lambda_tv", 0.1, 1.9728, None, None, None, 0.495),
    ("lambda_tv", 0.2, 1.4612, None, None, None, 0.513),
    ("lambda_l1", 0.0001, 1.8913, None, None, None, 0.487),
    ("lambda_l1", 0.0005, 1.8913, None, None, None, 0.487),
    ("lambda_l1", 0.002, 1.8913, None, None, None, 0.487),
    ("lambda_l1", 0.005, 1.8913, None, None, None, 0.487),
    ("lambda_l1", 0.01, 1.8913, None, None, None, 0.487),
    ("window_size", 26, 1.7690, None, None, None, 0.488),
    ("window_size", 78, 1.8488, None, None, None, 0.488),
    ("window_size", 104, 1.7965, None, None, None, 0.488),
    ("window_size", 130, 1.8169, None, None, None, 0.488),
    ("window_size", 156, 1.8082, None, None, None, 0.489),
    ("rho", 0.5, 1.8075, None, None, None, 0.496),
]

# 还要跑的参数
REMAINING = [
    ("rho", 2.0),
    ("rho", 5.0),
]


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


def run_one(X_panel, Y, valid_codes, **overrides):
    from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import run_v7_6_backtest
    cfg_kwargs = {**DEFAULTS, **overrides}
    from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import V7_6Config
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
    logging.info("Phase 1 续跑: rho 参数 + 历史结果合并")
    logging.info("=" * 60)

    from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import load_v7_6_data
    X_panel, Y, valid_codes = load_v7_6_data()
    logging.info("  X_panel: %s, Y: %s", X_panel.shape, Y.shape)

    # 1. 构建历史结果 DataFrame
    rows = []
    for param, value, calmar, sharpe, dd, ann, cv in HISTORICAL:
        row = {
            "param": param,
            "value": value,
            "lambda_tv": DEFAULTS["lambda_tv"],
            "lambda_l1": DEFAULTS["lambda_l1"],
            "window_size": DEFAULTS["window_size"],
            "rho": DEFAULTS["rho"],
            "oos_calmar": calmar,
            "oos_sharpe": sharpe if sharpe is not None else 0,
            "oos_dd": dd if dd is not None else 0,
            "oos_ann": ann if ann is not None else 0,
            "start_cv": cv,
            "note": "from_history",
        }
        # 默认 baseline 来自 Phase 0
        if param == "default":
            row["note"] = "from_phase0"
            row["oos_sharpe"] = 1.6776
            row["oos_dd"] = -0.1594
            row["oos_ann"] = 0.3015
        rows.append(row)
    logging.info("已加载 %d 条历史结果", len(rows))

    # 2. 跑剩余参数
    for param, val in REMAINING:
        logging.info("=" * 60)
        logging.info("[%s=%s] (续跑)", param, val)
        try:
            r = run_one(X_panel, Y, valid_codes, **{param: val})
            row = {
                "param": param,
                "value": val,
                "lambda_tv": DEFAULTS["lambda_tv"],
                "lambda_l1": DEFAULTS["lambda_l1"],
                "window_size": DEFAULTS["window_size"],
                "rho": DEFAULTS["rho"],
                **r,
                "note": "newly_run",
            }
            rows.append(row)
            logging.info("  OOS Calmar=%.4f, CV%%=%.1f%%, %.1fs",
                         r["oos_calmar"], r["start_cv"] * 100, r["seconds"])
        except Exception as e:
            logging.error("  失败: %s", e)

    # 3. 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "v7_6_sensitivity_single.csv"
    df.to_csv(out_path, index=False)
    logging.info("=" * 60)
    logging.info("结果已保存: %s", out_path)

    # 4. 输出汇总
    print("\n" + "=" * 80)
    print("Phase 1 最终结果 (含续跑)")
    print("=" * 80)

    base = df[df["param"] == "default"].iloc[0]
    base_oos = base["oos_calmar"]
    others = df[df["param"] != "default"]

    # 按参数分组
    for param_name in ["lambda_tv", "lambda_l1", "window_size", "rho"]:
        sub = df[df["param"].isin([param_name, "default"])].copy()
        print(f"\n## {param_name}:")
        cols = ["value", "oos_calmar", "oos_sharpe", "oos_dd", "start_cv"]
        print(sub[cols].to_string(index=False))

    print("\n" + "=" * 80)
    print(f"基线 OOS Calmar: {base_oos:.4f}")
    print("=" * 80)
    if len(others) > 0:
        max_cal = others["oos_calmar"].max()
        min_cal = others["oos_calmar"].min()
        avg_cal = others["oos_calmar"].mean()
        std_cal = others["oos_calmar"].std()
        print(f"其他组 OOS Calmar: min={min_cal:.4f}, max={max_cal:.4f}, "
              f"mean={avg_cal:.4f}, std={std_cal:.4f}")
        degradation = (base_oos - min_cal) / base_oos * 100 if base_oos > 0 else 0
        print(f"最大退化 (vs min): {degradation:.1f}%")
        if degradation > 50:
            print("🔴 严重参数敏感 (>50% 退化)")
        elif degradation > 30:
            print("🟡 中度参数敏感 (30-50% 退化)")
        else:
            print("🟢 低参数敏感 (<30% 退化)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
