# coding=utf-8
"""scripts/v4/v4_full_sensitivity.py — 完整敏感性测试.

3 个维度的网格搜索:
  1. 4 因子权重 (3 个典型配置)
  2. top_k (3, 4, 5, 6, 8, 10)
  3. 相关约束 (False, 0.7, 0.8)

共 3 × 6 × 3 = 54 个组合, 在完整 v4 回测引擎中跑.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from QuantNodes.strategy.momentum_etf_rotation.v4 import (
    V4Config,
    SmartBetaConfig,
    run_v4_backtest,
)
from QuantNodes.strategy.momentum_etf_rotation.v4.regime_detector_v4 import (
    detect_regime_simple,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.backtest import (
    compute_metrics,
)


def load_data():
    """加载 43 ETF 数据 (转换为价格序列)."""
    data_dir = REPO / "data" / "high_freq_macro"
    etf = pd.read_parquet(data_dir / "v7_10_Y_weekly.parquet")
    etf_clean = etf.fillna(0).replace([np.inf, -np.inf], 0)
    etf_count = (etf_clean != 0).sum()
    etf_clean = etf_clean.loc[:, etf_count > 100]
    etf_price = (1 + etf_clean).cumprod() * 100
    return etf_price


# 3 套典型 4 因子权重
WEIGHT_GRID = {
    "Stage28 (加动量)": {"value": 0.20, "quality": 0.30, "low_vol": 0.20, "momentum": 0.30},
    "偏动量": {"value": 0.10, "quality": 0.20, "low_vol": 0.20, "momentum": 0.50},
    "偏质量": {"value": 0.25, "quality": 0.50, "low_vol": 0.25, "momentum": 0.0},
}

# top_k 维度
TOP_K_GRID = [3, 4, 5, 6, 8, 10]

# 相关约束维度
CORR_GRID = [
    (False, 0.0, "无约束"),
    (True, 0.7, "corr=0.7"),
    (True, 0.8, "corr=0.8"),
]


def run_one(panel, regime_series, weights, top_k, use_corr, corr_thr):
    """跑一个组合."""
    sb = SmartBetaConfig(
        top_n=top_k,
        proxy_value_weight=weights["value"],
        proxy_quality_weight=weights["quality"],
        proxy_low_vol_weight=weights["low_vol"],
        proxy_momentum_weight=weights["momentum"],
        proxy_corr_constraint=use_corr,
        proxy_corr_threshold=corr_thr,
    )

    v4_config = V4Config(
        mode="v4B_smartbeta",
        style_enabled=False,
        factor_timing_enabled=False,
        industry_rotation_enabled=False,
        smart_beta=sb,
    )

    result = run_v4_backtest(panel, v4_config, hmm_regime_series=regime_series)
    metrics = compute_metrics(result.nav.pct_change().fillna(0), freq="W")
    return metrics


def main():
    print("=" * 80)
    print("完整敏感性测试 (权重 × top_k × 相关约束)")
    print("=" * 80)

    panel = load_data()
    print(f"\n数据: {panel.shape[0]} 周, {panel.shape[1]} ETF")
    print(f"时间: {panel.index.min()} ~ {panel.index.max()}")

    # 计算 regime
    panel_for_regime = panel.replace(0, np.nan).ffill().fillna(0)
    print(f"\n计算 regime...")
    regime_series = detect_regime_simple(panel_for_regime, list(panel.columns))

    # 总组合数
    total = len(WEIGHT_GRID) * len(TOP_K_GRID) * len(CORR_GRID)
    print(f"\n跑 {total} 个组合 (3 权重 × {len(TOP_K_GRID)} top_k × {len(CORR_GRID)} corr)...")

    results = []
    idx = 0
    for wname, weights in WEIGHT_GRID.items():
        for top_k in TOP_K_GRID:
            for use_corr, corr_thr, corr_name in CORR_GRID:
                idx += 1
                name = f"{wname}, top={top_k}, {corr_name}"
                try:
                    m = run_one(panel, regime_series, weights, top_k, use_corr, corr_thr)
                    results.append({
                        "权重": wname,
                        "top_k": top_k,
                        "corr_name": corr_name,
                        "corr_thr": corr_thr,
                        **m,
                    })
                    if idx % 10 == 0 or idx == total:
                        print(f"  [{idx}/{total}] 完成")
                except Exception as e:
                    print(f"  [{idx}/{total}] {name} FAILED: {e}")

    df = pd.DataFrame(results)
    df = df.sort_values("Sharpe", ascending=False).reset_index(drop=True)

    print(f"\n{'=' * 80}")
    print("完整 v4 回测结果 (按 Sharpe 排序, 前 20)")
    print("=" * 80)

    display_cols = [
        "权重", "top_k", "corr_name",
        "Sharpe", "Calmar", "AnnRet", "MaxDD", "WinRate", "TotalReturn",
    ]
    available = [c for c in display_cols if c in df.columns]
    print(df.head(20)[available].to_string(index=False))

    # 保存
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v4"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "v4_full_sensitivity.csv", index=False)
    print(f"\n保存: {output_dir / 'v4_full_sensitivity.csv'}")

    # 分析
    best = df.iloc[0]
    print(f"\n{'=' * 80}")
    print(f"最优配置")
    print("=" * 80)
    print(f"权重: {best['权重']}")
    print(f"top_k: {best['top_k']}")
    print(f"相关约束: {best['corr_name']}")
    print(f"Sharpe: {best['Sharpe']:.3f}")
    print(f"Calmar: {best['Calmar']:.3f}")
    print(f"年化: {best['AnnRet']:.2%}")
    print(f"MaxDD: {best['MaxDD']:.2%}")
    print(f"总收益: {best['TotalReturn']:.2%}")

    # 按 top_k 分组
    print(f"\n{'=' * 80}")
    print("按 top_k 分组 (按 Sharpe 平均排序)")
    print("=" * 80)
    tk_summary = df.groupby("top_k").agg(
        avg_sharpe=("Sharpe", "mean"),
        max_sharpe=("Sharpe", "max"),
        avg_calmar=("Calmar", "mean"),
        avg_maxdd=("MaxDD", "mean"),
        avg_total=("TotalReturn", "mean"),
    ).sort_values("avg_sharpe", ascending=False)
    print(tk_summary.to_string())

    # 按 corr 分组
    print(f"\n{'=' * 80}")
    print("按相关约束分组")
    print("=" * 80)
    corr_summary = df.groupby("corr_name").agg(
        avg_sharpe=("Sharpe", "mean"),
        max_sharpe=("Sharpe", "max"),
        avg_calmar=("Calmar", "mean"),
    ).sort_values("avg_sharpe", ascending=False)
    print(corr_summary.to_string())

    # 按 权重 分组
    print(f"\n{'=' * 80}")
    print("按 4 因子权重分组")
    print("=" * 80)
    w_summary = df.groupby("权重").agg(
        avg_sharpe=("Sharpe", "mean"),
        max_sharpe=("Sharpe", "max"),
        avg_calmar=("Calmar", "mean"),
    ).sort_values("avg_sharpe", ascending=False)
    print(w_summary.to_string())

    # 生成报告
    report_lines = [
        "# v4 完整敏感性测试 (Stage 30)",
        "",
        f"> 数据: {panel.index.min()} ~ {panel.index.max()} ({panel.shape[0]} 周)",
        f"> ETF: {panel.shape[1]} 个",
        f"> 总组合: {total} (3 权重 × {len(TOP_K_GRID)} top_k × {len(CORR_GRID)} corr)",
        "",
        "## 测试维度",
        "",
        f"### 权重 ({len(WEIGHT_GRID)} 个)",
    ]
    for wname, w in WEIGHT_GRID.items():
        report_lines.append(f"- **{wname}**: value={w['value']}, quality={w['quality']}, "
                            f"low_vol={w['low_vol']}, momentum={w['momentum']}")

    report_lines.extend([
        "",
        f"### top_k ({len(TOP_K_GRID)} 个)",
        f"- {TOP_K_GRID}",
        "",
        f"### 相关约束 ({len(CORR_GRID)} 个)",
    ])
    for use_corr, corr_thr, name in CORR_GRID:
        report_lines.append(f"- **{name}**: use_corr={use_corr}, threshold={corr_thr}")

    report_lines.extend([
        "",
        "## 最优配置 (Top 20)",
        "",
        "| 排序 | 权重 | top_k | 相关约束 | Sharpe | Calmar | 年化 | MaxDD | 胜率 | 总收益 |",
        "|------|------|-------|----------|--------|--------|------|-------|------|--------|",
    ])

    for i, (_, row) in enumerate(df.head(20).iterrows(), 1):
        report_lines.append(
            f"| {i} | {row['权重']} | {int(row['top_k'])} | {row['corr_name']} | "
            f"{row['Sharpe']:.3f} | {row['Calmar']:.3f} | {row['AnnRet']:.2%} | "
            f"{row['MaxDD']:.2%} | {row['WinRate']:.2%} | {row['TotalReturn']:.2%} |"
        )

    report_lines.extend([
        "",
        "## 按 top_k 分组",
        "",
        "| top_k | 平均 Sharpe | 最高 Sharpe | 平均 Calmar | 平均 MaxDD | 平均总收益 |",
        "|-------|------------|------------|------------|------------|------------|",
    ])

    for tk, row in tk_summary.iterrows():
        report_lines.append(
            f"| {tk} | {row['avg_sharpe']:.3f} | {row['max_sharpe']:.3f} | "
            f"{row['avg_calmar']:.3f} | {row['avg_maxdd']:.2%} | {row['avg_total']:.2%} |"
        )

    report_lines.extend([
        "",
        "## 按相关约束分组",
        "",
        "| 相关约束 | 平均 Sharpe | 最高 Sharpe | 平均 Calmar |",
        "|----------|------------|------------|------------|",
    ])

    for cn, row in corr_summary.iterrows():
        report_lines.append(
            f"| {cn} | {row['avg_sharpe']:.3f} | {row['max_sharpe']:.3f} | "
            f"{row['avg_calmar']:.3f} |"
        )

    report_lines.extend([
        "",
        "## 按 4 因子权重分组",
        "",
        "| 权重 | 平均 Sharpe | 最高 Sharpe | 平均 Calmar |",
        "|------|------------|------------|------------|",
    ])

    for wn, row in w_summary.iterrows():
        report_lines.append(
            f"| {wn} | {row['avg_sharpe']:.3f} | {row['max_sharpe']:.3f} | "
            f"{row['avg_calmar']:.3f} |"
        )

    # Top 10
    report_lines.extend([
        "",
        "## Top 10 详细",
        "",
    ])

    for i, (_, row) in enumerate(df.head(10).iterrows(), 1):
        report_lines.extend([
            f"### {i}. {row['权重']}, top={int(row['top_k'])}, {row['corr_name']}",
            f"- Sharpe: {row['Sharpe']:.3f}, Calmar: {row['Calmar']:.3f}",
            f"- 年化: {row['AnnRet']:.2%}, MaxDD: {row['MaxDD']:.2%}",
            f"- 胜率: {row['WinRate']:.2%}, 总收益: {row['TotalReturn']:.2%}",
            "",
        ])

    report_lines.extend([
        "## 关键发现",
        "",
    ])

    # 关键发现
    best_tk = tk_summary["avg_sharpe"].idxmax()
    best_corr = corr_summary["avg_sharpe"].idxmax()
    best_w = w_summary["avg_sharpe"].idxmax()

    report_lines.extend([
        f"1. **最优 top_k (平均)**: {best_tk} (avg Sharpe {tk_summary.loc[best_tk, 'avg_sharpe']:.3f})",
        f"2. **最优相关约束 (平均)**: {best_corr} (avg Sharpe {corr_summary.loc[best_corr, 'avg_sharpe']:.3f})",
        f"3. **最优权重 (平均)**: {best_w} (avg Sharpe {w_summary.loc[best_w, 'avg_sharpe']:.3f})",
        f"4. **全局最优**: {best['权重']}, top={int(best['top_k'])}, {best['corr_name']}",
        f"   - Sharpe: {best['Sharpe']:.3f}, Calmar: {best['Calmar']:.3f}",
        f"   - 年化: {best['AnnRet']:.2%}, MaxDD: {best['MaxDD']:.2%}",
        "",
        "## 文件清单",
        "",
        "- `v4_full_sensitivity.csv`: 完整 54 个组合数据",
        "- `v4_full_sensitivity_report.md`: 本报告",
    ])

    report_path = output_dir / "v4_full_sensitivity_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n报告: {report_path}")

    print(f"\n{'='*80}")
    print("完成!")


if __name__ == "__main__":
    main()