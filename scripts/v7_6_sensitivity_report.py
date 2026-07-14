# coding: utf-8
"""v7.6 Phase 7: 综合敏感性报告.

目的: 汇总所有 Phase CSV, 生成 markdown 报告.

用法:
   python3.11 scripts/v7_6_sensitivity_report.py

输出:
   reports/momentum_etf_rotation/v7_6_sensitivity_report.md
"""
from __future__ import annotations

import logging
import sys
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

OUTPUT_DIR = REPO / "reports/momentum_etf_rotation"

PHASE_FILES = {
    "0+8_paper_default": OUTPUT_DIR / "v7_6_sensitivity_paper_default.csv",
    "1_single": OUTPUT_DIR / "v7_6_sensitivity_single.csv",
    "2_holdout": OUTPUT_DIR / "v7_6_holdout_test.csv",
    "3_bootstrap": OUTPUT_DIR / "v7_6_sensitivity_bootstrap.csv",
    "4_missing": OUTPUT_DIR / "v7_6_sensitivity_missing.csv",
    "5_construction": OUTPUT_DIR / "v7_6_sensitivity_construction.csv",
    "6_beta": OUTPUT_DIR / "v7_6_beta_stability.csv",
}


def load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def grade_section(score: float, red: float, yellow: float) -> str:
    """根据越界程度评级."""
    if abs(score) >= red:
        return "🔴"
    elif abs(score) >= yellow:
        return "🟡"
    else:
        return "🟢"


def main() -> int:
    logging.info("=" * 60)
    logging.info("Phase 7: 综合敏感性报告")
    logging.info("=" * 60)

    rows = []
    grades = {}

    # === Phase 0 + 8: 论文默认对比 ===
    df = load_csv(PHASE_FILES["0+8_paper_default"])
    if df is not None:
        base = df[df["label"] == "baseline_current"].iloc[0]
        paper = df[df["label"] == "paper_default"].iloc[0]
        degradation = (base["oos_calmar"] - paper["oos_calmar"]) / base["oos_calmar"] if base["oos_calmar"] > 0 else 0
        verdict = "🟢 低过拟合" if abs(degradation) <= 0.3 else ("🟡 中度" if abs(degradation) <= 0.5 else "🔴 严重")
        grades["0+8"] = verdict
        rows.append(("Phase 0+8: 论文默认 λ 对比", f"OOS Calmar {base['oos_calmar']:.4f} vs {paper['oos_calmar']:.4f}, 退化 {degradation*100:+.1f}%", verdict))

    # === Phase 1: 单参数 ===
    df = load_csv(PHASE_FILES["1_single"])
    if df is not None:
        base = df[df["param"] == "default"].iloc[0]
        base_oos = base["oos_calmar"]
        others = df[df["param"] != "default"]
        max_uptake = (others["oos_calmar"].max() - base_oos) / base_oos if base_oos > 0 else 0
        worst_degrade = (base_oos - others["oos_calmar"].min()) / base_oos if base_oos > 0 else 0
        # 评级: 退化 vs 提升 → 主要看退化
        if abs(worst_degrade) > 0.5:
            verdict = "🔴 严重参数敏感"
        elif abs(worst_degrade) > 0.3:
            verdict = "🟡 中度参数敏感"
        else:
            verdict = "🟢 低参数敏感"
        grades["1"] = verdict
        rows.append(("Phase 1: 单参数敏感性", f"OOS Calmar 退化 {worst_degrade*100:+.1f}%, 提升 {max_uptake*100:+.1f}%", verdict))

        # 按参数分组
        for param in ["lambda_tv", "lambda_l1", "window_size", "rho"]:
            sub = others[others["param"] == param]
            if len(sub) > 0:
                logging.info("  %s: OOS Calmar range %.4f ~ %.4f",
                             param, sub["oos_calmar"].min(), sub["oos_calmar"].max())

    # === Phase 2: Hold-out 段 ===
    df = load_csv(PHASE_FILES["2_holdout"])
    if df is not None:
        segments = df[df["segment"] != "FULL"]
        cals = segments["calmar"]
        if len(cals) >= 2 and cals.min() > 0:
            ratio = cals.max() / cals.min()
            recent_seg = segments[segments["segment"] == "B"]
            full_cal = df[df["segment"] == "FULL"].iloc[0]["calmar"]
            if len(recent_seg) > 0 and full_cal > 0:
                recent_degrade = (full_cal - recent_seg.iloc[0]["calmar"]) / full_cal
            else:
                recent_degrade = 0
            if ratio > 3:
                verdict = "🔴 段间不一致"
            elif ratio > 2:
                verdict = "🟡 段间差异"
            else:
                verdict = "🟢 段间一致"
            if abs(recent_degrade) > 0.5:
                verdict = "🔴 近期段严重退化"
            grades["2"] = verdict
            rows.append(("Phase 2: Hold-out 多段", f"段 max/min={ratio:.2f}x, 最近段退化 {recent_degrade*100:+.1f}%", verdict))

    # === Phase 3: Bootstrap ===
    df = load_csv(PHASE_FILES["3_bootstrap"])
    if df is not None:
        ok = df[df["status"] == "OK"]
        if len(ok) >= 5:
            cals = ok["calmar"]
            cv = cals.std() / cals.mean() if cals.mean() > 0 else 0
            if cv > 0.5:
                verdict = "🔴 Bootstrap 不稳定"
            elif cv > 0.3:
                verdict = "🟡 Bootstrap 中度"
            else:
                verdict = "🟢 Bootstrap 稳定"
            grades["3"] = verdict
            rows.append(("Phase 3: Bootstrap 稳定性", f"CV={cv:.2%}, mean Calmar={cals.mean():.4f}", verdict))

    # === Phase 4: 缺失数据 ===
    df = load_csv(PHASE_FILES["4_missing"])
    if df is not None:
        ok = df[df["status"] == "OK"]
        base = ok[ok["rate"] == 0.0].iloc[0] if len(ok[ok["rate"] == 0.0]) > 0 else None
        if base is not None and len(ok) > 1:
            base_calmar = base["calmar"]
            for rate in [0.05, 0.10, 0.20]:
                sub = ok[ok["rate"] == rate]
                if len(sub) > 0:
                    avg = sub["calmar"].mean()
                    degradation = (base_calmar - avg) / base_calmar if base_calmar > 0 else 0
                    if rate == 0.20:
                        if abs(degradation) > 0.5:
                            verdict = "🔴 20% 缺失退化严重"
                        elif abs(degradation) > 0.3:
                            verdict = "🟡 20% 缺失退化中度"
                        else:
                            verdict = "🟢 20% 缺失鲁棒"
                        grades["4"] = verdict
                        rows.append(("Phase 4: 缺失数据扰动", f"20% 缺失退化 {degradation*100:+.1f}%", verdict))

    # === Phase 5: 构造层 ===
    df = load_csv(PHASE_FILES["5_construction"])
    if df is not None:
        base = df[df["param"] == "default"].iloc[0]
        base_oos = base["oos_calmar"]
        others = df[df["param"] != "default"]
        worst = (base_oos - others["oos_calmar"].min()) / base_oos if base_oos > 0 else 0
        if abs(worst) > 0.5:
            verdict = "🔴 构造层严重敏感"
        elif abs(worst) > 0.3:
            verdict = "🟡 构造层中度敏感"
        else:
            verdict = "🟢 构造层低敏感"
        grades["5"] = verdict
        rows.append(("Phase 5: 构造层扰动", f"最大退化 {worst*100:+.1f}%", verdict))

    # === Phase 6: β 稳定性 ===
    df = load_csv(PHASE_FILES["6_beta"])
    if df is not None:
        bp_freq = float(df[df["metric"] == "bp_freq"].iloc[0]["value"]) if len(df[df["metric"] == "bp_freq"]) > 0 else 0
        beta_cv = float(df[df["metric"] == "beta_cv_per_dim_mean"].iloc[0]["value"]) if len(df[df["metric"] == "beta_cv_per_dim_mean"]) > 0 else 0
        acf = float(df[df["metric"] == "beta_acf_lag1_mean"].iloc[0]["value"]) if len(df[df["metric"] == "beta_acf_lag1_mean"]) > 0 else 0
        if bp_freq > 0.5 or beta_cv > 1.0 or acf < 0.5:
            verdict = "🔴 β 不稳定"
        elif bp_freq > 0.3 or beta_cv > 0.5 or acf < 0.7:
            verdict = "🟡 β 中度"
        else:
            verdict = "🟢 β 稳定"
        grades["6"] = verdict
        rows.append(("Phase 6: β_path 稳定性", f"断点频率 {bp_freq:.2%}, CV {beta_cv:.2f}, ACF {acf:.2f}", verdict))

    # === 总体评级 ===
    grade_count = sum(1 for g in grades.values() if "🟢" in g)
    yellow_count = sum(1 for g in grades.values() if "🟡" in g)
    red_count = sum(1 for g in grades.values() if "🔴" in g)

    if red_count >= 2:
        overall = "🔴 红色 (严重过拟合嫌疑)"
    elif red_count >= 1 or yellow_count >= 3:
        overall = "🟡 黄色 (中度过拟合嫌疑, 需优化)"
    else:
        overall = "🟢 绿色 (低过拟合嫌疑, 可锁定)"

    # 生成报告
    lines = [
        "# v7.6 TV-PR 参数敏感性综合报告",
        "",
        f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 总体评级",
        "",
        f"**{overall}**",
        "",
        f"- 🟢 绿色: {grade_count}",
        f"- 🟡 黄色: {yellow_count}",
        f"- 🔴 红色: {red_count}",
        "",
        "## 各 Phase 评级",
        "",
        "| Phase | 内容 | 关键指标 | 评级 |",
        "|-------|------|----------|------|",
    ]

    for name, detail, verdict in rows:
        lines.append(f"| {name.split(':')[0]} | {name.split(':')[1].strip()} | {detail} | {verdict} |")

    lines.extend([
        "",
        "## 关键发现",
        "",
    ])

    # 建议
    if "🔴" in overall:
        lines.extend([
            "### 🔴 严重过拟合嫌疑",
            "",
            "- 关键参数敏感度过高",
            "- Hold-out 段间不一致",
            "- 数据扰动下不稳定",
            "",
            "**建议**:",
            "1. 重新设计 v7.6, 用更保守的参数范围",
            "2. K 维度从 20 维降到 5-10 维",
            "3. λ_tv 限定在 [0.05, 0.10] (论文推荐范围)",
            "4. 改为 expanding window 而非 rolling",
        ])
    elif "🟡" in overall:
        lines.extend([
            "### 🟡 中度过拟合嫌疑",
            "",
            "- 部分参数敏感, 部分段不一致",
            "- bootstrap 中等不稳定",
            "",
            "**建议**:",
            "1. 缩小 λ 范围至 [0.05, 0.10]",
            "2. 在 hold-out 段间加额外约束",
            "3. 观察是否有结构性问题",
            "4. 与 v1.0 ensemble 分散风险",
        ])
    else:
        lines.extend([
            "### 🟢 低过拟合嫌疑",
            "",
            "- 多维度参数敏感度低",
            "- 段间一致性良好",
            "- Bootstrap 稳定",
            "- β_path 时变结构合理",
            "",
            "**建议**:",
            "1. ✅ 锁定 v7.6 baseline",
            "2. 进入 ensemble (v1.0 + v7.6) 阶段",
            "3. 实盘前实盘 sim 3-6 个月",
            "4. 监控 OOS Calmar 是否 ≥ 1.5",
        ])

    report = "\n".join(lines)
    out_path = OUTPUT_DIR / "v7_6_sensitivity_report.md"
    out_path.write_text(report, encoding="utf-8")

    # 输出汇总
    print("\n" + "=" * 80)
    print("Phase 7 综合评级")
    print("=" * 80)
    for name, detail, verdict in rows:
        print(f"  {verdict} {name}: {detail}")
    print(f"\n**总体: {overall}**")

    return 0


if __name__ == "__main__":
    sys.exit(main())
