# coding: utf-8
"""v7.6 Phase 0 + Phase 8: 论文默认 λ 参数快速验证.

目的: 用 Cui 2025 论文 Table 4 默认值, 看 v7.6 TV-PR 在不调参下的真实 OOS
       与当前 λ_tv=0.01 CV 调出的结果对比.

用法:
   python3.11 scripts/v7_6_paper_default_check.py

输出:
   reports/momentum_etf_rotation/v7_6_sensitivity_paper_default.csv
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

# 论文 Cui 2025 Table 4 推荐范围中位数 + 当前 CV 调出的值
PARAMS = [
    # (label, lambda_tv, lambda_l1, window_size, 备注)
    ("baseline_current", 0.01, 0.001, 52, "当前 (CV 调出)"),
    ("paper_default", 0.05, 0.001, 52, "Cui 2025 论文默认"),
    ("paper_upper", 0.10, 0.005, 52, "论文上限"),
    ("paper_mid", 0.07, 0.003, 52, "论文中位"),
]

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


def run_one_config(cfg: V7_6Config, X_panel: np.ndarray, Y: pd.DataFrame,
                   valid_codes: list, start_points: list) -> dict:
    """跑一个配置: 全段 + OOS + 起点依赖."""
    result = {"cfg_lambda_tv": cfg.lambda_tv, "cfg_lambda_l1": cfg.lambda_l1,
              "cfg_window_size": cfg.window_size}

    # 全段
    t0 = time.time()
    nav_weekly, nav_daily = run_v7_6_backtest(
        X_panel, Y, valid_codes, cfg, return_daily=True
    )
    t1 = time.time()

    # 全段指标
    full = compute_metrics(nav_daily)
    result["full_calmar"] = full["calmar"]
    result["full_ann"] = full["ann_return"]
    result["full_vol"] = full["vol"]
    result["full_dd"] = full["max_dd"]
    result["full_sharpe"] = full["sharpe"]
    result["full_seconds"] = round(t1 - t0, 1)

    # OOS 指标
    nav_daily_oos = nav_daily.loc["2022-01-01":]
    oos = compute_metrics(nav_daily_oos)
    result["oos_calmar"] = oos["calmar"]
    result["oos_ann"] = oos["ann_return"]
    result["oos_vol"] = oos["vol"]
    result["oos_dd"] = oos["max_dd"]
    result["oos_sharpe"] = oos["sharpe"]

    # 起点依赖
    start_calmar = []
    for start in start_points:
        mask = Y.index >= start
        Y_start = Y[mask]
        X_start = X_panel[mask]
        if len(Y_start) < cfg.min_history + 12:
            continue
        _, nav_daily_start = run_v7_6_backtest(
            X_start, Y_start, valid_codes, cfg, return_daily=True
        )
        metrics_start = compute_metrics(nav_daily_start)
        start_calmar.append(metrics_start["calmar"])

    if len(start_calmar) >= 2:
        mean_c = float(np.mean(start_calmar))
        std_c = float(np.std(start_calmar))
        cv = std_c / mean_c if mean_c > 0 else 0.0
    else:
        mean_c, std_c, cv = 0.0, 0.0, 0.0
    result["start_cv"] = round(cv, 4)
    result["start_mean"] = round(mean_c, 4)
    result["start_std"] = round(std_c, 4)
    result["start_n"] = len(start_calmar)
    return result


def main() -> int:
    logging.info("=" * 60)
    logging.info("Phase 0 + 8: 论文默认 λ 参数快速验证")
    logging.info("=" * 60)

    logging.info("加载数据...")
    t0 = time.time()
    X_panel, Y, valid_codes = load_v7_6_data()
    logging.info("  X_panel: %s, Y: %s, valid_codes: %d, 耗时: %.1fs",
                 X_panel.shape, Y.shape, len(valid_codes), time.time() - t0)

    rows = []
    for label, lambda_tv, lambda_l1, window_size, note in PARAMS:
        logging.info("=" * 60)
        logging.info("配置: %s (λ_tv=%s, λ_l1=%s, ws=%s)",
                     label, lambda_tv, lambda_l1, window_size)
        logging.info("       %s", note)

        cfg = V7_6Config(
            lambda_tv=lambda_tv,
            lambda_l1=lambda_l1,
            min_history=52,
            window_size=window_size,
        )

        result = run_one_config(cfg, X_panel, Y, valid_codes, START_POINTS)
        result["label"] = label
        result["note"] = note
        rows.append(result)
        logging.info("  全段 Calmar=%.4f, OOS Calmar=%.4f, 起点 CV%%=%.1f%%",
                     result["full_calmar"], result["oos_calmar"], result["start_cv"] * 100)

    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    cols = ["label", "cfg_lambda_tv", "cfg_lambda_l1", "cfg_window_size", "note",
            "full_calmar", "full_ann", "full_vol", "full_dd", "full_sharpe",
            "full_seconds", "oos_calmar", "oos_ann", "oos_vol", "oos_dd", "oos_sharpe",
            "start_mean", "start_std", "start_cv", "start_n"]
    df = df[cols]
    out_path = OUTPUT_DIR / "v7_6_sensitivity_paper_default.csv"
    df.to_csv(out_path, index=False)
    logging.info("=" * 60)
    logging.info("结果已保存: %s", out_path)

    # 对比分析
    print("\n" + "=" * 80)
    print("Phase 0 + 8 关键结果对比")
    print("=" * 80)
    print(df.to_string(index=False))

    # 判据
    baseline = df[df["label"] == "baseline_current"].iloc[0]
    paper_default = df[df["label"] == "paper_default"].iloc[0]
    degradation = (baseline["oos_calmar"] - paper_default["oos_calmar"]) / baseline["oos_calmar"] if baseline["oos_calmar"] > 0 else 0

    print("\n" + "=" * 80)
    print("判据")
    print("=" * 80)
    print(f"baseline (当前 CV 调出): OOS Calmar = {baseline['oos_calmar']:.4f}")
    print(f"paper_default (Cui 2025): OOS Calmar = {paper_default['oos_calmar']:.4f}")
    print(f"OOS Calmar 退化: {degradation * 100:+.1f}%")
    print()
    if degradation > 0.5:
        verdict = "🔴 严重过拟合 (>50% 退化)"
    elif degradation > 0.3:
        verdict = "🟡 中度过拟合 (30-50% 退化)"
    else:
        verdict = "🟢 低过拟合 (<30% 退化)"
    print(f"判据: {verdict}")

    if paper_default["oos_calmar"] < 0.5:
        print("\n⚠️ 论文默认参数 OOS Calmar < 0.5 → 强烈过拟合")
    elif paper_default["oos_calmar"] < 1.0:
        print("\n⚠️ 论文默认参数 OOS Calmar 0.5-1.0 → 中度过拟合")

    return 0


if __name__ == "__main__":
    sys.exit(main())
