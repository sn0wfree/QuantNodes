# coding: utf-8
"""v7.6 验证: 全段 + OOS + 起点依赖 (日频 NAV).

用法:
  python3.11 scripts/eval_v7_6_validation.py

输出:
  reports/momentum_etf_rotation/v7_6_validation.md
  reports/momentum_etf_rotation/v7_6_nav_daily.csv
  reports/momentum_etf_rotation/v7_6_nav.csv
  reports/momentum_etf_rotation/v7_6_weights.csv
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

# λ 最优值 (from lambda_cv)
LAMBDA_TV = 0.01
LAMBDA_L1 = 0.001

# 起点
START_POINTS = [
    "2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01", "2022-01-01",
]

OUTPUT_DIR = REPO / "reports/momentum_etf_rotation"

# 频率
DAYS_PER_YEAR = 252


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
    logging.info("v7.6 验证: 全段 + OOS + 起点依赖 (日频 NAV)")
    logging.info("=" * 60)

    # 1. 加载数据
    logging.info("加载数据...")
    t0 = time.time()
    X_panel_full, Y_full, valid_codes = load_v7_6_data()
    t1 = time.time()
    logging.info("  X_panel: %s, Y: %s, 耗时: %.1fs", X_panel_full.shape, Y_full.shape, t1 - t0)

    # 配置
    cfg = V7_6Config(
        lambda_tv=LAMBDA_TV,
        lambda_l1=LAMBDA_L1,
        min_history=52,
        window_size=52,
    )

    # 初始化结果变量
    metrics_full = {"calmar": 0.0, "ann_return": 0.0, "vol": 0.0, "max_dd": 0.0, "sharpe": 0.0}
    metrics_oos = {"calmar": 0.0, "ann_return": 0.0, "vol": 0.0, "max_dd": 0.0, "sharpe": 0.0}
    mean_c, std_c, cv = 0.0, 0.0, 0.0

    results = []

    # 2. 全段回测 (2018-2026) — 返回周频 NAV + 日频 NAV
    logging.info("=" * 60)
    logging.info("全段回测 (2018-2026)")
    t0 = time.time()
    nav_weekly, nav_daily = run_v7_6_backtest(
        X_panel_full, Y_full, valid_codes, cfg, return_daily=True
    )
    t1 = time.time()
    metrics_full = compute_metrics(nav_daily)
    metrics_full["period"] = "全段 2018-2026"
    metrics_full["耗时"] = f"{t1 - t0:.1f}s"
    results.append(metrics_full)
    logging.info("  周频终值: %.2f", nav_weekly.iloc[-1])
    logging.info("  日频终值: %.2f", nav_daily.iloc[-1])
    logging.info("  日频 Calmar=%.4f, Ann=%.2f%%, Vol=%.2f%%, MaxDD=%.2f%%, Sharpe=%.2f",
                 metrics_full["calmar"], metrics_full["ann_return"] * 100,
                 metrics_full["vol"] * 100, metrics_full["max_dd"] * 100, metrics_full["sharpe"])

    # 3. OOS 回测 (2022-2026) - 从日频结果截取
    logging.info("=" * 60)
    logging.info("OOS 回测 (2022-2026) - 从日频结果截取")
    nav_daily_oos = nav_daily.loc["2022-01-01":]
    metrics_oos = compute_metrics(nav_daily_oos)
    metrics_oos["period"] = "OOS 2022-2026"
    metrics_oos["耗时"] = "-"
    results.append(metrics_oos)
    logging.info("  日频 Calmar=%.4f, Ann=%.2f%%, Vol=%.2f%%, MaxDD=%.2f%%, Sharpe=%.2f",
                 metrics_oos["calmar"], metrics_oos["ann_return"] * 100,
                 metrics_oos["vol"] * 100, metrics_oos["max_dd"] * 100, metrics_oos["sharpe"])

    # 4. 起点依赖测试
    logging.info("=" * 60)
    logging.info("起点依赖测试")
    start_calmar = []

    for start in START_POINTS:
        mask = Y_full.index >= start
        Y_start = Y_full[mask]
        X_start = X_panel_full[mask]

        if len(Y_start) < cfg.min_history + 12:
            continue

        t0 = time.time()
        _, nav_daily_start = run_v7_6_backtest(
            X_start, Y_start, valid_codes, cfg, return_daily=True
        )
        t1 = time.time()
        metrics_start = compute_metrics(nav_daily_start)
        start_calmar.append(metrics_start["calmar"])
        logging.info("  %s: Calmar=%.4f, 耗时=%.1fs", start, metrics_start["calmar"], t1 - t0)

    # 计算 CV%
    if len(start_calmar) >= 2:
        mean_c = np.mean(start_calmar)
        std_c = np.std(start_calmar)
        cv = std_c / mean_c if mean_c > 0 else 0.0
        logging.info("  Mean Calmar: %.4f", mean_c)
        logging.info("  Std Calmar: %.4f", std_c)
        logging.info("  CV%%: %.1f%% (阈值 25%%)", cv * 100)
        logging.info("  结果: %s", "PASS" if cv <= 0.25 else "FAIL")
    else:
        cv = 0.0
        logging.warning("  数据不足, 无法计算 CV%%")

    # 5. 导出 NAV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nav_weekly.to_frame("v7.6 TV-PR").to_csv(OUTPUT_DIR / "v7_6_nav.csv")
    nav_daily.to_frame("v7.6 TV-PR").to_csv(OUTPUT_DIR / "v7_6_nav_daily.csv")
    logging.info("周频 NAV 已导出: %s", OUTPUT_DIR / "v7_6_nav.csv")
    logging.info("日频 NAV 已导出: %s", OUTPUT_DIR / "v7_6_nav_daily.csv")

    # 6. 生成报告
    logging.info("=" * 60)
    logging.info("生成报告...")

    lines = [
        "# v7.6 TV-PR 验证报告",
        "",
        f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 参数",
        f"- λ_tv: {LAMBDA_TV}",
        f"- λ_l1: {LAMBDA_L1}",
        f"- 调仓频率: 周频 (W)",
        f"- 资产池: expanded (43 有效资产)",
        f"- min_history: 52 周",
        f"- window_size: 52 周",
        "",
        "## 业绩指标 (日频 NAV)",
        "",
        "| 期间 | Calmar | 年化收益 | 波动率 | 最大回撤 | 夏普 | 耗时 |",
        "|------|--------|----------|--------|----------|------|------|",
    ]

    for r in results:
        lines.append(
            f"| {r['period']} | {r['calmar']:.4f} | {r['ann_return']:.2%} | "
            f"{r['vol']:.2%} | {r['max_dd']:.2%} | {r['sharpe']:.2f} | {r['耗时']} |"
        )

    lines.extend([
        "",
        "## 周频 vs 日频 NAV 对比",
        "",
        "| 指标 | 周频 NAV | 日频 NAV |",
        "|------|----------|----------|",
        f"| 终值 | {nav_weekly.iloc[-1]:.2f} | {nav_daily.iloc[-1]:.2f} |",
        f"| 总收益 | {(nav_weekly.iloc[-1] - 1) * 100:.2f}% | {(nav_daily.iloc[-1] - 1) * 100:.2f}% |",
        f"| OOS Calmar | - | {metrics_oos['calmar']:.4f} |",
        "",
        "## 起点依赖",
        "",
        f"- Mean Calmar: {mean_c:.4f}",
        f"- Std Calmar: {std_c:.4f}",
        f"- CV%: {cv:.1%} (阈值 25%)",
        f"- 结果: {'PASS' if cv <= 0.25 else 'FAIL'}",
        "",
        "| 起点 | Calmar |",
        "|------|--------|",
    ])

    for start, c in zip(START_POINTS[:len(start_calmar)], start_calmar):
        lines.append(f"| {start} | {c:.4f} |")

    lines.extend([
        "",
        "## 与 v7.3 对比",
        "",
        "| 策略 | OOS Calmar | 起点 CV% |",
        "|------|-----------|----------|",
        f"| v7.3 baseline | 0.620 | 待测 |",
        f"| v7.6 TV-PR (日频) | {metrics_oos.get('calmar', 0):.4f} | {cv:.1%} |",
        "",
        "## 结论",
        "",
        f"v7.6 TV-PR OOS Calmar = {metrics_oos.get('calmar', 0):.4f}, "
        f"起点 CV% = {cv:.1%}. ",
    ])

    if metrics_oos.get("calmar", 0) >= 0.5 and cv <= 0.25:
        lines.append("✅ 达标 (OOS Calmar ≥ 0.5, CV% ≤ 25%)")
    else:
        lines.append("❌ 未达标 (需进一步优化)")

    report = "\n".join(lines)

    report_path = OUTPUT_DIR / "v7_6_validation.md"
    report_path.write_text(report, encoding="utf-8")
    logging.info("报告已保存: %s", report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
