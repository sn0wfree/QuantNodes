# coding: utf-8
"""v7.6 Phase 2: Hold-out 多段测试.

目的: 冻结当前 λ + window, 测试 3 个独立时间段, 看是否一致.

用法:
   python3.11 scripts/v7_6_holdout_test.py

输出:
   reports/momentum_etf_rotation/v7_6_holdout_test.csv
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

# 3 个独立 hold-out 段
SEGMENTS = [
    ("A", "2022-01-01", "2024-06-30", "近 2.5 年"),
    ("B", "2024-07-01", "2026-06-30", "最近 2 年"),
    ("C", "2023-01-01", "2024-12-31", "中段 2 年"),
    # 额外段
    ("D", "2024-01-01", "2024-12-31", "1 年段"),
    ("E", "2025-01-01", "2026-06-30", "极近期 1.5 年"),
]

# 冻结参数 (基于 Phase 0 + 8 结果)
DEFAULT_PARAMS = {
    "lambda_tv": 0.05,
    "lambda_l1": 0.001,
    "window_size": 52,
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


def main() -> int:
    logging.info("=" * 60)
    logging.info("Phase 2: Hold-out 多段测试")
    logging.info("=" * 60)

    logging.info("加载数据...")
    t0 = time.time()
    X_panel, Y, valid_codes = load_v7_6_data()
    logging.info("  X_panel: %s, Y: %s, 耗时: %.1fs", X_panel.shape, Y.shape, time.time() - t0)

    # 冻结当前最优参数 (paper_default + 全段起点完整)
    cfg = V7_6Config(
        lambda_tv=DEFAULT_PARAMS["lambda_tv"],
        lambda_l1=DEFAULT_PARAMS["lambda_l1"],
        min_history=52,
        window_size=DEFAULT_PARAMS["window_size"],
    )

    rows = []

    # 1. 完整全段回测
    logging.info("=" * 60)
    logging.info("全段回测 2018-01 ~ 2026-06")
    t0 = time.time()
    nav_weekly, nav_daily = run_v7_6_backtest(
        X_panel, Y, valid_codes, cfg, return_daily=True
    )
    full = compute_metrics(nav_daily)
    full["seconds"] = round(time.time() - t0, 1)
    full["segment"] = "FULL"
    full["label"] = "全段 2018-2026"
    full["period"] = "2018-01-01~2026-06-30"
    rows.append(full)
    logging.info("  全段 Calmar=%.4f, Sharpe=%.2f, DD=%.2f%%",
                 full["calmar"], full["sharpe"], full["max_dd"] * 100)

    # 2. 各段测试
    for label, start, end, desc in SEGMENTS:
        logging.info("=" * 60)
        logging.info("段 %s: %s (%s ~ %s)", label, desc, start, end)

        mask = (Y.index >= start) & (Y.index <= end)
        Y_seg = Y[mask]
        X_seg = X_panel[mask]

        if len(Y_seg) < cfg.min_history + 26:
            logging.warning("  数据不足, 跳过")
            continue

        t0 = time.time()
        _, nav_daily_seg = run_v7_6_backtest(
            X_seg, Y_seg, valid_codes, cfg, return_daily=True
        )
        elapsed = round(time.time() - t0, 1)
        m = compute_metrics(nav_daily_seg)
        m["segment"] = label
        m["label"] = desc
        m["period"] = f"{start}~{end}"
        m["seconds"] = elapsed
        rows.append(m)
        logging.info("  段 %s: Calmar=%.4f, Sharpe=%.2f, 年化=%.2f%%, DD=%.2f%%, %.1fs",
                     label, m["calmar"], m["sharpe"], m["ann_return"] * 100,
                     m["max_dd"] * 100, elapsed)

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    cols = ["segment", "label", "period", "calmar", "sharpe", "ann_return",
            "vol", "max_dd", "seconds"]
    df = df[cols]
    out_path = OUTPUT_DIR / "v7_6_holdout_test.csv"
    df.to_csv(out_path, index=False)
    logging.info("=" * 60)
    logging.info("结果已保存: %s", out_path)

    # 分析
    print("\n" + "=" * 80)
    print("Phase 2 Hold-out 段结果")
    print("=" * 80)
    print(df.to_string(index=False))

    # 判据
    segments_only = df[df["segment"] != "FULL"]
    if len(segments_only) >= 2:
        cals = segments_only["calmar"].astype(float)
        max_cal = cals.max()
        min_cal = cals.min()
        ratio = max_cal / min_cal if min_cal > 0 else float('inf')
        full_cal = df[df["segment"] == "FULL"].iloc[0]["calmar"]
        recent_seg = segments_only[segments_only["segment"] == "B"]
        if len(recent_seg) > 0:
            recent_cal = recent_seg.iloc[0]["calmar"]
            degradation = (full_cal - recent_cal) / full_cal if full_cal > 0 else 0
        else:
            recent_cal = None
            degradation = 0

        print("\n" + "=" * 80)
        print("判据")
        print("=" * 80)
        print(f"全段 Calmar: {full_cal:.4f}")
        print(f"段 Calmar min/max: {min_cal:.4f} / {max_cal:.4f}")
        print(f"段最大/最小 ratio: {ratio:.2f}")

        if recent_cal is not None:
            print(f"最近段 (B 2024-2026) Calmar: {recent_cal:.4f}")
            print(f"最近段 vs 全段退化: {degradation * 100:+.1f}%")

        if ratio > 3.0:
            print("🔴 段间差异极大 (>3x)")
        elif ratio > 2.0:
            print("🟡 段间差异大 (2-3x)")
        else:
            print("🟢 段间一致 (<2x)")

        if recent_cal is not None and degradation > 0.5:
            print("🔴 近期段严重退化 (>50%)")
        elif recent_cal is not None and degradation > 0.3:
            print("🟡 近期段退化 (30-50%)")
        elif recent_cal is not None:
            print("🟢 近期段无明显退化")

    return 0


if __name__ == "__main__":
    sys.exit(main())
