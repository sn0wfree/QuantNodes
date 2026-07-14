# coding: utf-8
"""v6.2 过拟合综合评估: 交易成本 + 起点依赖 + 与 v6.1 对比.

用法:
  python3.11 scripts/eval_v6_2_overfitting.py

输出:
  reports/momentum_etf_rotation/v6_2_overfitting_report.md
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v6_1 import V6_1Config, run_v6_1_backtest
from QuantNodes.strategy.momentum_etf_rotation.v6_2 import V6_2Config, run_v6_2_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# 起点
START_POINTS = [
    "2018-01-01", "2018-07-01", "2019-01-01", "2019-07-01",
    "2020-01-01", "2020-07-01", "2021-01-01", "2021-07-01",
    "2022-01-01", "2023-01-01",
]

# 数据路径
CLOSE_PARQUET = REPO / "data/real/etf_nav_2018-01-01_2026-06-30.parquet"
OHLCV_PARQUET = REPO / "data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet"
OUTPUT_DIR = REPO / "reports/momentum_etf_rotation"


def calmar(nav: pd.Series) -> float:
    if nav.empty or len(nav) < 2:
        return 0.0
    rets = nav.pct_change().dropna()
    if rets.empty:
        return 0.0
    n_years = len(rets) / 252
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = float((1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1)
    cummax = nav.cummax()
    dd = (nav / cummax - 1)
    max_dd = float(dd.min())
    return ann_ret / abs(max_dd) if max_dd < 0 else 0.0


def max_dd(nav: pd.Series) -> float:
    if nav.empty or len(nav) < 2:
        return 0.0
    cummax = nav.cummax()
    dd = (nav / cummax - 1)
    return float(dd.min())


def ann_return(nav: pd.Series) -> float:
    if nav.empty or len(nav) < 2:
        return 0.0
    rets = nav.pct_change().dropna()
    if rets.empty:
        return 0.0
    n_years = len(rets) / 252
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    return float((1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1)


def sharpe(nav: pd.Series) -> float:
    if nav.empty or len(nav) < 2:
        return 0.0
    rets = nav.pct_change().dropna()
    if rets.empty or rets.std() == 0:
        return 0.0
    n_years = len(rets) / 252
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret = float((1 + total_ret) ** (1 / max(n_years, 1e-9)) - 1)
    return ann_ret / (rets.std() * np.sqrt(252))


def run_strategy(
    panel_close: pd.DataFrame,
    panel_ohlcv: pd.DataFrame,
    strategy: str,
    cfg,
) -> pd.Series:
    """运行策略并返回 NAV."""
    if strategy == "v6.1":
        return run_v6_1_backtest(panel_close, panel_ohlcv, cfg)
    else:
        return run_v6_2_backtest(panel_close, panel_ohlcv, cfg)


def test_transaction_costs(
    panel_close: pd.DataFrame,
    panel_ohlcv: pd.DataFrame,
) -> list[dict]:
    """测试交易成本影响."""
    results = []

    # v6.2 不扣成本
    cfg_no_cost = V6_2Config(
        use_orthogonal=True,
        sort_method="ir_expanding",
        min_history=120,
        ic_min_months=6,
        cost_enabled=False,
    )
    nav_no_cost = run_v6_2_backtest(panel_close, panel_ohlcv, cfg_no_cost)
    results.append({
        "策略": "v6.2 (no cost)",
        "Calmar": round(calmar(nav_no_cost), 4),
        "最大回撤": round(max_dd(nav_no_cost), 4),
        "年化": round(ann_return(nav_no_cost), 4),
        "夏普": round(sharpe(nav_no_cost), 4),
    })

    # v6.2 扣成本 (5bp + 10bp)
    cfg_with_cost = V6_2Config(
        use_orthogonal=True,
        sort_method="ir_expanding",
        min_history=120,
        ic_min_months=6,
        cost_enabled=True,
        commission_bp=5.0,
        slippage_bp=10.0,
    )
    nav_with_cost = run_v6_2_backtest(panel_close, panel_ohlcv, cfg_with_cost)
    results.append({
        "策略": "v6.2 (5bp+10bp)",
        "Calmar": round(calmar(nav_with_cost), 4),
        "最大回撤": round(max_dd(nav_with_cost), 4),
        "年化": round(ann_return(nav_with_cost), 4),
        "夏普": round(sharpe(nav_with_cost), 4),
    })

    # v6.1 扣成本
    cfg_v61 = V6_1Config(
        use_ic_weighting=True,
        min_history=120,
        ic_min_months=6,
    )
    nav_v61 = run_v6_1_backtest(panel_close, panel_ohlcv, cfg_v61)
    results.append({
        "策略": "v6.1 (baseline)",
        "Calmar": round(calmar(nav_v61), 4),
        "最大回撤": round(max_dd(nav_v61), 4),
        "年化": round(ann_return(nav_v61), 4),
        "夏普": round(sharpe(nav_v61), 4),
    })

    return results


def test_starting_points(
    panel_close: pd.DataFrame,
    panel_ohlcv: pd.DataFrame,
    strategy: str,
    cfg,
) -> list[dict]:
    """测试起点依赖."""
    results = []

    for start in START_POINTS:
        mask = panel_close.index >= start
        pc = panel_close[mask]
        po = panel_ohlcv[mask]

        if len(pc) < 252:
            continue

        try:
            nav = run_strategy(pc, po, strategy, cfg)
            c = calmar(nav)
            results.append({
                "起点": start,
                "Calmar": round(c, 4),
                "最大回撤": round(max_dd(nav), 4),
                "年化": round(ann_return(nav), 4),
                "夏普": round(sharpe(nav), 4),
            })
        except Exception as exc:
            logging.error("  %s 起点 %s 失败: %s", strategy, start, exc)

    return results


def generate_report(
    cost_results: list[dict],
    sp_v62: list[dict],
    sp_v61: list[dict],
) -> str:
    """生成 Markdown 报告."""
    lines = [
        "# v6.2 过拟合评估报告",
        "",
        f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 一、交易成本影响",
        "",
    ]

    # 交易成本表
    lines.append("| 策略 | Calmar | 最大回撤 | 年化 | 夏普 |")
    lines.append("|------|--------|----------|------|------|")
    for r in cost_results:
        lines.append(f"| {r['策略']} | {r['Calmar']:.4f} | {r['最大回撤']:.2%} | {r['年化']:.2%} | {r['夏普']:.2f} |")

    # 计算成本影响
    no_cost = cost_results[0]["Calmar"]
    with_cost = cost_results[1]["Calmar"]
    delta = with_cost - no_cost
    pct = delta / no_cost * 100 if no_cost != 0 else 0
    lines.extend([
        "",
        f"**成本影响**: Calmar 从 {no_cost:.4f} 降至 {with_cost:.4f} ({pct:+.1f}%)",
        "",
    ])

    # 起点依赖
    lines.extend([
        "## 二、起点依赖测试",
        "",
        "### v6.2 (ir_expanding)",
        "",
    ])

    if sp_v62:
        calmars = [r["Calmar"] for r in sp_v62 if r["Calmar"] > 0]
        if len(calmars) >= 2:
            mean_c = np.mean(calmars)
            std_c = np.std(calmars)
            cv = std_c / mean_c if mean_c > 0 else 0.0
            lines.append(f"- Mean Calmar: {mean_c:.4f}")
            lines.append(f"- Std Calmar: {std_c:.4f}")
            lines.append(f"- CV%: {cv:.1%} (阈值 25%)")
            lines.append(f"- 结果: {'PASS' if cv <= 0.25 else 'FAIL'}")
            lines.append("")
            lines.append("| 起点 | Calmar | 最大回撤 | 年化 | 夏普 |")
            lines.append("|------|--------|----------|------|------|")
            for r in sp_v62:
                lines.append(f"| {r['起点']} | {r['Calmar']:.4f} | {r['最大回撤']:.2%} | {r['年化']:.2%} | {r['夏普']:.2f} |")
        else:
            lines.append("- 数据不足, 无法计算 CV%")
    else:
        lines.append("- 无数据")

    lines.extend([
        "",
        "### v6.1 (IC12, baseline)",
        "",
    ])

    if sp_v61:
        calmars = [r["Calmar"] for r in sp_v61 if r["Calmar"] > 0]
        if len(calmars) >= 2:
            mean_c = np.mean(calmars)
            std_c = np.std(calmars)
            cv = std_c / mean_c if mean_c > 0 else 0.0
            lines.append(f"- Mean Calmar: {mean_c:.4f}")
            lines.append(f"- Std Calmar: {std_c:.4f}")
            lines.append(f"- CV%: {cv:.1%} (阈值 25%)")
            lines.append(f"- 结果: {'PASS' if cv <= 0.25 else 'FAIL'}")
            lines.append("")
            lines.append("| 起点 | Calmar | 最大回撤 | 年化 | 夏普 |")
            lines.append("|------|--------|----------|------|------|")
            for r in sp_v61:
                lines.append(f"| {r['起点']} | {r['Calmar']:.4f} | {r['最大回撤']:.2%} | {r['年化']:.2%} | {r['夏普']:.2f} |")
        else:
            lines.append("- 数据不足, 无法计算 CV%")
    else:
        lines.append("- 无数据")

    # 结论
    lines.extend([
        "",
        "## 三、结论与建议",
        "",
        "### 交易成本",
        "- v6.2 默认启用成本 (5bp 佣金 + 10bp 滑点)",
        f"- 成本影响: Calmar {pct:+.1f}%",
        "",
        "### 起点依赖",
        "- v6.2 需要 10 个起点验证 CV% ≤ 25%",
        "- v6.1 作为 baseline 对照",
        "",
        "### 后续步骤",
        "1. 如果 v6.2 CV% > 25%, 考虑 ensemble v6.2 + v6.1",
        "2. 如果 v6.2 扣成本后弱于 v6.1, 降级为研究版本",
        "3. 保持 combo 50/50 (v6.2 + v7.3) 作为主推",
    ])

    return "\n".join(lines)


def main() -> int:
    # 加载数据
    logging.info("加载数据...")
    panel_close = pd.read_parquet(CLOSE_PARQUET)
    panel_ohlcv = pd.read_parquet(OHLCV_PARQUET)
    logging.info("  panel_close: %s", panel_close.shape)
    logging.info("  panel_ohlcv: %s", panel_ohlcv.shape)

    # 对齐数据 (用 panel_close 的 index)
    common_idx = panel_close.index.intersection(panel_ohlcv.index)
    panel_close = panel_close.loc[common_idx]
    panel_ohlcv = panel_ohlcv.loc[common_idx]
    logging.info("  对齐后: %s", panel_close.shape)

    # 1. 交易成本测试
    logging.info("=" * 60)
    logging.info("测试 1: 交易成本影响")
    cost_results = test_transaction_costs(panel_close, panel_ohlcv)
    for r in cost_results:
        logging.info("  %s: Calmar=%.4f", r["策略"], r["Calmar"])

    # 2. 起点依赖测试 (v6.2)
    logging.info("=" * 60)
    logging.info("测试 2: 起点依赖 (v6.2)")
    cfg_v62 = V6_2Config(
        use_orthogonal=True,
        sort_method="ir_expanding",
        min_history=120,
        ic_min_months=6,
        cost_enabled=True,
    )
    sp_v62 = test_starting_points(panel_close, panel_ohlcv, "v6.2", cfg_v62)
    if sp_v62:
        calmars = [r["Calmar"] for r in sp_v62 if r["Calmar"] > 0]
        if len(calmars) >= 2:
            mean_c = np.mean(calmars)
            std_c = np.std(calmars)
            cv = std_c / mean_c if mean_c > 0 else 0.0
            logging.info("  Mean Calmar: %.4f, CV%%: %.1f%%", mean_c, cv * 100)

    # 3. 起点依赖测试 (v6.1)
    logging.info("=" * 60)
    logging.info("测试 3: 起点依赖 (v6.1)")
    cfg_v61 = V6_1Config(
        use_ic_weighting=True,
        min_history=120,
        ic_min_months=6,
    )
    sp_v61 = test_starting_points(panel_close, panel_ohlcv, "v6.1", cfg_v61)
    if sp_v61:
        calmars = [r["Calmar"] for r in sp_v61 if r["Calmar"] > 0]
        if len(calmars) >= 2:
            mean_c = np.mean(calmars)
            std_c = np.std(calmars)
            cv = std_c / mean_c if mean_c > 0 else 0.0
            logging.info("  Mean Calmar: %.4f, CV%%: %.1f%%", mean_c, cv * 100)

    # 生成报告
    logging.info("=" * 60)
    logging.info("生成报告...")
    report = generate_report(cost_results, sp_v62, sp_v61)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "v6_2_overfitting_report.md"
    report_path.write_text(report, encoding="utf-8")
    logging.info("报告已保存: %s", report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
