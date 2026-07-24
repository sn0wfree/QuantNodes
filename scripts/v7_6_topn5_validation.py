# coding: utf-8
"""v7.6 top_n=5 起点 CV% 验证.

目的: 验证 Phase 5 发现的 top_n=5 是否能改进起点 CV% (50% → ≤25%).

测试组合:
   1. baseline (Phase 0+8): top_n=10, lambda_tv=0.05, ws=52, rho=1.0
   2. top_n=5 only: top_n=5, lambda_tv=0.05, ws=52, rho=1.0
   3. best (Phase 1+5):  top_n=5, lambda_tv=0.07, ws=78, rho=2.0
   4. top_n=5 + tf:  top_n=5, lambda_tv=0.05, ws=52, rho=1.0 + trend filter
   5. top_n=5 + sl:  top_n=5, lambda_tv=0.05, ws=52, rho=1.0 + stop loss
   6. top_n=5 + tf + sl: 综合

用法:
   python3.11 scripts/v7_6_topn5_validation.py

输出:
   reports/momentum_etf_rotation/v7_6_topn5_validation.csv
   reports/momentum_etf_rotation/v7_6_topn5_validation.md
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

DAYS_PER_YEAR = 252
START_POINTS = [
    "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01",
]

# 测试组合
COMBOS = [
    {
        "label": "baseline_topn10",
        "note": "Phase 0+8 默认 (top_n=10)",
        "lambda_tv": 0.05, "lambda_l1": 0.001,
        "window_size": 52, "rho": 1.0,
        "top_n": 10, "max_weight": 0.25,
    },
    {
        "label": "topn5_only",
        "note": "Phase 5 新最优 (top_n=5)",
        "lambda_tv": 0.05, "lambda_l1": 0.001,
        "window_size": 52, "rho": 1.0,
        "top_n": 5, "max_weight": 0.25,
    },
    {
        "label": "best_combo",
        "note": "Phase 1+5 联合最优",
        "lambda_tv": 0.07, "lambda_l1": 0.001,
        "window_size": 78, "rho": 2.0,
        "top_n": 5, "max_weight": 0.30,
    },
    {
        "label": "topn5_tw",
        "note": "top_n=5 + window=78",
        "lambda_tv": 0.05, "lambda_l1": 0.001,
        "window_size": 78, "rho": 1.0,
        "top_n": 5, "max_weight": 0.25,
    },
    {
        "label": "topn5_rho2",
        "note": "top_n=5 + rho=2.0",
        "lambda_tv": 0.05, "lambda_l1": 0.001,
        "window_size": 52, "rho": 2.0,
        "top_n": 5, "max_weight": 0.25,
    },
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
    return {"calmar": round(calmar, 4), "ann_return": round(ann_ret, 4),
            "vol": round(vol, 4), "max_dd": round(max_dd, 4), "sharpe": round(sharpe, 4)}


def run_combo(X_panel, Y, valid_codes, combo: dict) -> dict:
    """跑一个 combo: 全段 + OOS + 起点 CV."""
    from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import run_v7_6_backtest

    cfg = V7_6Config(
        lambda_tv=combo["lambda_tv"],
        lambda_l1=combo["lambda_l1"],
        window_size=combo["window_size"],
        rho=combo["rho"],
        top_n=combo["top_n"],
        max_weight=combo["max_weight"],
        min_history=52,
    )

    t0 = time.time()
    # 全段
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
        start_calmar.append((start, m["calmar"]))

    cals = [c for _, c in start_calmar]
    mean_c = float(np.mean(cals)) if cals else 0
    std_c = float(np.std(cals)) if cals else 0
    cv = std_c / mean_c if mean_c > 0 else 0

    return {
        "label": combo["label"],
        "note": combo["note"],
        "lambda_tv": combo["lambda_tv"],
        "lambda_l1": combo["lambda_l1"],
        "window_size": combo["window_size"],
        "rho": combo["rho"],
        "top_n": combo["top_n"],
        "max_weight": combo["max_weight"],
        "full_calmar": full["calmar"],
        "full_sharpe": full["sharpe"],
        "full_dd": full["max_dd"],
        "full_ann": full["ann_return"],
        "oos_calmar": oos["calmar"],
        "oos_sharpe": oos["sharpe"],
        "oos_dd": oos["max_dd"],
        "oos_ann": oos["ann_return"],
        "start_mean": round(mean_c, 4),
        "start_std": round(std_c, 4),
        "start_cv": round(cv, 4),
        "start_n": len(cals),
        "start_details": start_calmar,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    logging.info("=" * 60)
    logging.info("v7.6 top_n=5 起点 CV% 验证")
    logging.info("=" * 60)

    X_panel, Y, valid_codes = load_v7_6_data()
    logging.info("  X_panel: %s, Y: %s", X_panel.shape, Y.shape)

    rows = []
    summaries = []
    for combo in COMBOS:
        logging.info("=" * 60)
        logging.info("Combo: %s (%s)", combo["label"], combo["note"])
        logging.info("  λ_tv=%s, λ_l1=%s, ws=%s, rho=%s, top_n=%s, mw=%s",
                     combo["lambda_tv"], combo["lambda_l1"],
                     combo["window_size"], combo["rho"],
                     combo["top_n"], combo["max_weight"])

        try:
            r = run_combo(X_panel, Y, valid_codes, combo)
        except Exception as e:
            logging.error("  失败: %s", e)
            continue

        rows.append({k: v for k, v in r.items() if k != "start_details"})
        logging.info("  OOS Calmar=%.4f, 起点 CV%%=%.1f%%, 起点均值=%.4f, %.1fs",
                     r["oos_calmar"], r["start_cv"] * 100,
                     r["start_mean"], r["seconds"])

        for start, c in r["start_details"]:
            logging.info("    起点 %s: Calmar=%.4f", start, c)

        summaries.append(r)

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / "v7_6_topn5_validation.csv"
    df.to_csv(out_csv, index=False)
    logging.info("=" * 60)
    logging.info("CSV 已保存: %s", out_csv)

    # 输出对比
    print("\n" + "=" * 100)
    print("top_n=5 起点 CV% 验证结果")
    print("=" * 100)
    cols = ["label", "top_n", "lambda_tv", "window_size", "rho",
            "oos_calmar", "oos_sharpe", "oos_dd", "start_mean", "start_cv"]
    print(df[cols].to_string(index=False))

    # 起点对比
    print("\n" + "=" * 100)
    print("各组合的起点 Calmar:")
    print("=" * 100)
    header = "起点".ljust(12) + " | " + " | ".join([s["label"].ljust(18) for s in summaries])
    print(header)
    print("-" * len(header))
    for start in START_POINTS:
        line = start.ljust(12) + " | "
        for s in summaries:
            val = next((c for st, c in s["start_details"] if st == start), 0.0)
            line += f"{val:.4f}".ljust(20) + " | "
        print(line)
    means = "MEAN".ljust(12) + " | "
    for s in summaries:
        means += f"{s['start_mean']:.4f}".ljust(20) + " | "
    print(means)
    cvs = "CV%".ljust(12) + " | "
    for s in summaries:
        cvs += f"{s['start_cv']*100:.1f}%".ljust(20) + " | "
    print(cvs)

    # 判据
    print("\n" + "=" * 100)
    print("判据: 起点 CV% 是否从 50% 改进到 ≤25%?")
    print("=" * 100)
    base = next(s for s in summaries if s["label"] == "baseline_topn10")
    base_cv = base["start_cv"]
    print(f"基线 (top_n=10): CV%={base_cv*100:.1f}%")
    print()
    for s in summaries:
        if s["label"] == "baseline_topn10":
            continue
        diff = s["start_cv"] - base_cv
        improved = "✅ 改进" if s["start_cv"] < 0.25 else ("⬜ 未达 ≤25%")
        print(f"  {improved} {s['label']:18s}: CV%={s['start_cv']*100:5.1f}% (差 {diff*100:+.1f}pp), "
              f"OOS Calmar={s['oos_calmar']:.4f}")

    # 生成 markdown 报告
    lines = [
        "# v7.6 top_n=5 起点 CV% 验证报告",
        "",
        f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 背景",
        "",
        "- 当前 v7.6 起点 CV% = 50% (FAIL, 阈值 25%)",
        "- Phase 5 发现 top_n=5 比 top_n=10 提升 149% (OOS Calmar 1.89 → 4.71)",
        "- 验证: top_n=5 是否能同时改进起点稳定性?",
        "",
        "## 测试组合",
        "",
        "| 标签 | top_n | λ_tv | window | rho | 备注 |",
        "|------|-------|------|--------|-----|------|",
    ]
    for combo in COMBOS:
        lines.append(
            f"| {combo['label']} | {combo['top_n']} | {combo['lambda_tv']} | "
            f"{combo['window_size']} | {combo['rho']} | {combo['note']} |"
        )

    lines.extend([
        "",
        "## 各组合业绩",
        "",
        "| 标签 | OOS Calmar | OOS Sharpe | OOS DD | 起点均值 | 起点 CV% | 状态 |",
        "|------|-----------|------------|--------|----------|----------|------|",
    ])
    for s in summaries:
        status = "✅ PASS" if s["start_cv"] <= 0.25 else "❌ FAIL"
        lines.append(
            f"| {s['label']} | {s['oos_calmar']:.4f} | {s['oos_sharpe']:.2f} | "
            f"{s['oos_dd']*100:.2f}% | {s['start_mean']:.4f} | {s['start_cv']*100:.1f}% | {status} |"
        )

    lines.extend([
        "",
        "## 起点 Calmar 分布",
        "",
        "| 起点 | " + " | ".join([s["label"] for s in summaries]) + " |",
        "|------|" + "|".join(["------" for _ in summaries]) + "|",
    ])
    for start in START_POINTS:
        line = f"| {start} | "
        for s in summaries:
            val = next((c for st, c in s["start_details"] if st == start), 0.0)
            line += f"{val:.4f} | "
        lines.append(line)
    lines.append("| **MEAN** | " + " | ".join([f"**{s['start_mean']:.4f}**" for s in summaries]) + " |")
    lines.append("| **CV%** | " + " | ".join([f"**{s['start_cv']*100:.1f}%**" for s in summaries]) + " |")

    # 结论
    lines.extend([
        "",
        "## 结论",
        "",
    ])
    best_cv = min(summaries, key=lambda x: x["start_cv"])
    best_oos = max(summaries, key=lambda x: x["oos_calmar"])
    lines.append(f"- 最低 CV%: **{best_cv['label']}** = {best_cv['start_cv']*100:.1f}%")
    lines.append(f"- 最高 OOS Calmar: **{best_oos['label']}** = {best_oos['oos_calmar']:.4f}")
    lines.append(f"- 基线 CV%: {base_cv*100:.1f}% → 改进 {base_cv*100 - best_cv['start_cv']*100:.1f}pp")
    if best_cv["start_cv"] <= 0.25:
        lines.append("")
        lines.append(f"### ✅ 达到 ≤25% 阈值! 推荐锁定 **{best_cv['label']}** 配置")
    else:
        lines.append("")
        lines.append(f"### ⚠️ 仍 >25% 阈值 (最低 {best_cv['start_cv']*100:.1f}%)")
        lines.append("")
        lines.append("**进一步优化方向**:")
        lines.append("1. 加趋势过滤 (TF) - 熊市减仓")
        lines.append("2. 加硬止损 (SL) - DD > 10% 全仓债券")
        lines.append("3. K 维度从 20 维降到 5-10 维")
        lines.append("4. 改 expanding window")

    report = "\n".join(lines)
    out_md = OUTPUT_DIR / "v7_6_topn5_validation.md"
    out_md.write_text(report, encoding="utf-8")
    logging.info("报告已保存: %s", out_md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
