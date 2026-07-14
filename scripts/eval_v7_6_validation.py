# coding: utf-8
"""v7.6 验证: 全段 + OOS + 起点依赖.

用法:
  python3.11 scripts/eval_v7_6_validation.py

输出:
  reports/momentum_etf_rotation/v7_6_validation.md
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

from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import (
    load_monthly_macro_factors,
    load_monthly_pv_factors,
    load_monthly_asset_returns,
    build_mixed_factor_panel,
)
from QuantNodes.strategy.momentum_etf_rotation.v7.tvpr_estimator import tvpr_estimator
from QuantNodes.strategy.momentum_etf_rotation.v7.macro_substrategy_v7_6 import (
    V7_6Config,
    construct_portfolio,
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


def compute_metrics(nav: pd.Series) -> dict:
    """计算业绩指标."""
    if nav.empty or len(nav) < 2:
        return {"calmar": 0.0, "ann_return": 0.0, "vol": 0.0, "max_dd": 0.0, "sharpe": 0.0}

    rets = nav.pct_change().dropna()
    if rets.empty:
        return {"calmar": 0.0, "ann_return": 0.0, "vol": 0.0, "max_dd": 0.0, "sharpe": 0.0}

    n_years = len(rets) / 12  # 月频
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = float((1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1)
    vol = float(rets.std() * np.sqrt(12))
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


def run_backtest(
    Y: pd.DataFrame,
    X_panel: np.ndarray,
    cfg: V7_6Config,
) -> pd.Series:
    """运行回测."""
    beta_path = tvpr_estimator(
        Y, X_panel,
        lambda_tv=cfg.lambda_tv,
        lambda_l1=cfg.lambda_l1,
        min_history=cfg.min_history,
        max_iter=cfg.max_iter,
        tol=cfg.tol,
    )
    return construct_portfolio(Y, beta_path, cfg)


def main() -> int:
    logging.info("=" * 60)
    logging.info("v7.6 验证: 全段 + OOS + 起点依赖")
    logging.info("=" * 60)

    # 1. 加载数据
    logging.info("加载数据...")
    t0 = time.time()

    from QuantNodes.strategy.momentum_etf_rotation.v7.data_loader_v7_6 import load_v7_6_data
    X_panel_full, Y_full, valid_codes = load_v7_6_data()

    t1 = time.time()
    logging.info("  X_panel: %s, Y: %s, 耗时: %.1fs", X_panel_full.shape, Y_full.shape, t1 - t0)

    # 配置
    cfg = V7_6Config(
        lambda_tv=LAMBDA_TV,
        lambda_l1=LAMBDA_L1,
        min_history=12,
    )

    results = []

    # 2. 全段回测
    logging.info("=" * 60)
    logging.info("全段回测 (2018-2026)")
    t0 = time.time()
    nav_full = run_backtest(Y_full, X_panel_full, cfg)
    t1 = time.time()
    metrics_full = compute_metrics(nav_full)
    metrics_full["period"] = "全段 2018-2026"
    metrics_full["耗时"] = f"{t1 - t0:.1f}s"
    results.append(metrics_full)
    logging.info("  Calmar=%.4f, Ann=%.2f%%, Vol=%.2f%%, MaxDD=%.2f%%, Sharpe=%.2f",
                 metrics_full["calmar"], metrics_full["ann_return"] * 100,
                 metrics_full["vol"] * 100, metrics_full["max_dd"] * 100, metrics_full["sharpe"])

    # 3. OOS 回测 (2022-2026)
    logging.info("=" * 60)
    logging.info("OOS 回测 (2022-2026)")
    mask_oos = Y_full.index >= "2022-01-01"
    Y_oos = Y_full[mask_oos]
    X_oos = X_panel_full[mask_oos]

    if len(Y_oos) > cfg.min_history:
        t0 = time.time()
        nav_oos = run_backtest(Y_oos, X_oos, cfg)
        t1 = time.time()
        metrics_oos = compute_metrics(nav_oos)
        metrics_oos["period"] = "OOS 2022-2026"
        metrics_oos["耗时"] = f"{t1 - t0:.1f}s"
        results.append(metrics_oos)
        logging.info("  Calmar=%.4f, Ann=%.2f%%, Vol=%.2f%%, MaxDD=%.2f%%, Sharpe=%.2f",
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
        nav_start = run_backtest(Y_start, X_start, cfg)
        t1 = time.time()
        metrics_start = compute_metrics(nav_start)
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

    # 5. 生成报告
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
        f"- 调仓频率: 月度",
        f"- 资产池: expanded (56)",
        "",
        "## 业绩指标",
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
        f"| v7.6 TV-PR | {metrics_oos.get('calmar', 0):.4f} | {cv:.1%} |",
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "v7_6_validation.md"
    report_path.write_text(report, encoding="utf-8")
    logging.info("报告已保存: %s", report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
