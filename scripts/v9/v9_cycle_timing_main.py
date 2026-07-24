# coding=utf-8
"""scripts/v9/v9_cycle_timing_main.py — v9 完整回测主入口.

用法:
    python3.11 scripts/v9/v9_cycle_timing_main.py

功能:
    1. 加载数据 (宏观因子 + 43 ETF + VIX)
    2. HP 滤波 + VMD 分解
    3. Hilbert 相位 + 双相干
    4. 评分合成 (40+40+20)
    5. 大盘信号生成 (迟滞)
    6. 仓位合成 (v9 等权 + 信号)
    7. 完整回测 + 多起点 + 成本敏感性
    8. 输出报告 + NAV 图

输出:
    reports/momentum_etf_rotation/v9/
    ├── backtest_results.csv
    ├── nav_comparison.png
    ├── multi_start_results.csv
    ├── cost_sensitivity.csv
    └── backtest_summary.md
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from QuantNodes.strategy.momentum_etf_rotation.v9.coupling import coupling_analysis
from QuantNodes.strategy.momentum_etf_rotation.v9.scoring import (
    compute_score_timeseries,
    score_to_signal_hysteresis,
    DEFAULT_IMF_WEIGHTS,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.backtest import (
    run_backtest,
    cost_sensitivity,
    compute_metrics,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.position import (
    compute_v9_only_position,
    align_v9_to_weekly,
)


def load_data():
    data_dir = REPO / "data" / "high_freq_macro"
    print("Loading data...")
    x_panel = np.load(data_dir / "v7_14_X_panel.npy")
    y_weekly = pd.read_parquet(data_dir / "v7_10_Y_weekly.parquet")
    vix_daily = pd.read_parquet(data_dir / "macro_vix_daily.parquet")["vix"]
    print(f"  Y weekly: {y_weekly.shape}, range {y_weekly.index.min()} ~ {y_weekly.index.max()}")
    return {
        "x_panel": x_panel,
        "y_weekly": y_weekly,
        "vix_daily": vix_daily,
    }


def hp_filter(series: pd.Series, lamb: float = 100) -> tuple:
    from statsmodels.tsa.filters.hp_filter import hpfilter
    cycle, trend = hpfilter(series.dropna(), lamb=lamb)
    return cycle, trend


def vmd_decompose(signal: np.ndarray, K: int = 4, alpha: float = 1000) -> tuple:
    from vmdpy import VMD
    u, _, omega = VMD(signal, alpha, 0, K, 0, 1, 1e-6)
    return u, omega


def main():
    print("=" * 60)
    print("v9 周期择时 — 完整回测")
    print("=" * 60)

    data = load_data()
    y_weekly = data["y_weekly"]

    print("\n[Step 1] HP 滤波 + VMD 多周期分解")
    hs300 = y_weekly.mean(axis=1).fillna(0)
    nav_hs300 = (1 + hs300).cumprod()
    log_hs300 = np.log(nav_hs300)
    cycle_hs300, _ = hp_filter(log_hs300, lamb=100)
    cycle_clean = cycle_hs300.dropna()
    imfs, omega = vmd_decompose(cycle_clean.values, K=4, alpha=1000)
    print(f"  VMD: {imfs.shape}, 中心频率: {omega[-1, :]}")

    print("\n[Step 2] Hilbert 相位 + 双相干分析")
    coupling_result = coupling_analysis(
        imfs,
        date_index=cycle_clean.index,
        lock_threshold_deg=30.0,
        lock_min_duration=12,
        plv_window=12,
    )
    print(f"  PLV 锁定对数: {coupling_result['n_locked_pairs'].iloc[-1]}")
    print(f"  最大双相干: {coupling_result['bic_max']:.4f}")

    print("\n[Step 3] 评分合成")
    vix_weekly = data["vix_daily"].resample("W").last().reindex(cycle_clean.index, method="ffill")
    total_score, cycle_score, coupling_score, vix_score, multi_phases, signal = compute_score_timeseries(
        imfs, coupling_result, vix_weekly, cycle_clean.index,
        window=12, imf_weights=DEFAULT_IMF_WEIGHTS, cn_calibration=True,
    )

    signal = score_to_signal_hysteresis(total_score, upper=35.0, lower=15.0, initial=1)
    print(f"  评分时序: {total_score.shape}")
    print(f"  总分分布: min={total_score.min():.1f}, max={total_score.max():.1f}, mean={total_score.mean():.1f}")
    print(f"  最新总分: {total_score.iloc[-1]:.1f}")
    print(f"  信号分布: 0={int((signal==0).sum())}周, 1={int((signal==1).sum())}周")

    print("\n[Step 4] v9 大盘信号 + 仓位合成")
    y_aligned = y_weekly.reindex(cycle_clean.index, method="ffill")
    v9_position = compute_v9_only_position(signal, y_aligned)
    print(f"  v9 仓位 shape: {v9_position.shape}")
    print(f"  v9 平均持仓比例: {v9_position.sum(axis=1).mean():.2%}")

    print("\n[Step 5] 完整回测")
    nav_v9, ret_v9, metrics_v9 = run_backtest(v9_position, y_aligned, cost_bps=10.0)
    print(f"  v9 Sharpe: {metrics_v9['Sharpe']:.3f}")
    print(f"  v9 Calmar: {metrics_v9['Calmar']:.3f}")
    print(f"  v9 MaxDD: {metrics_v9['MaxDD']:.2%}")
    print(f"  v9 AnnRet: {metrics_v9['AnnRet']:.2%}")

    print("\n[Step 6] 基准对比")
    nav_etf, ret_etf, metrics_etf = run_backtest(
        compute_v9_only_position(pd.Series(1, index=y_aligned.index), y_aligned),
        y_aligned, cost_bps=10.0,
    )
    print(f"  ETF 等权 Sharpe: {metrics_etf['Sharpe']:.3f}")
    print(f"  ETF 等权 MaxDD: {metrics_etf['MaxDD']:.2%}")
    print(f"  ETF 等权 AnnRet: {metrics_etf['AnnRet']:.2%}")

    print("\n[Step 7] 成本敏感性")
    cost_df = cost_sensitivity(v9_position, y_weekly)
    print(cost_df.to_string(index=False))

    print("\n[Step 8] 输出报告")
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v9"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([
        {**metrics_v9, "strategy": "v9_cycle_timing"},
        {**metrics_etf, "strategy": "etf_equal_weight"},
    ])
    metrics_df.to_csv(output_dir / "backtest_results.csv", index=False)
    print(f"  指标: {output_dir / 'backtest_results.csv'}")

    cost_df.to_csv(output_dir / "cost_sensitivity.csv", index=False)
    print(f"  成本敏感性: {output_dir / 'cost_sensitivity.csv'}")

    plt.figure(figsize=(14, 6))
    nav_v9.plot(label="v9 Cycle Timing", linewidth=2, color="#3b82f6")
    nav_etf.plot(label="ETF Equal Weight", linewidth=1.5, color="#94a3b8", linestyle="--")
    plt.title("v9 vs ETF 等权 NAV 对比", fontsize=14)
    plt.xlabel("日期")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    nav_png = output_dir / "nav_comparison.png"
    plt.savefig(nav_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  NAV 图: {nav_png}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(total_score.index, total_score.values, color="#3b82f6", linewidth=1.5)
    axes[0].axhline(y=50, color="green", linestyle="--", alpha=0.5, label="Upper (50)")
    axes[0].axhline(y=30, color="red", linestyle="--", alpha=0.5, label="Lower (30)")
    axes[0].set_ylabel("Total Score")
    axes[0].set_title("v9 评分时序")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cycle_score.index, cycle_score.values, color="#10b981", linewidth=1.5, label="Cycle")
    axes[1].plot(coupling_score.index, coupling_score.values, color="#f59e0b", linewidth=1.5, label="Coupling")
    axes[1].plot(vix_score.index, vix_score.values, color="#8b5cf6", linewidth=1.5, label="VIX")
    axes[1].set_ylabel("Component Score")
    axes[1].set_title("v9 评分分解 (40+40+20)")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, alpha=0.3)

    axes[2].fill_between(signal.index, 0, signal.values, color="#22c55e", alpha=0.5)
    axes[2].set_ylabel("v9 Signal")
    axes[2].set_xlabel("Date")
    axes[2].set_title("v9 大盘信号 (0/1)")
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    score_png = output_dir / "score_timeseries.png"
    plt.savefig(score_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  评分图: {score_png}")

    summary = []
    summary.append("# v9 周期择时回测总结")
    summary.append("")
    summary.append(f"> 回测期: {y_weekly.index[0].strftime('%Y-%m-%d')} ~ {y_weekly.index[-1].strftime('%Y-%m-%d')}")
    summary.append(f"> 数据频率: 周频")
    summary.append("")
    summary.append("## 主指标对比")
    summary.append("")
    summary.append("| 指标 | ETF 等权 | v9 周期择时 | 差异 |")
    summary.append("|------|----------|-------------|------|")
    summary.append(f"| Sharpe | {metrics_etf['Sharpe']:.3f} | {metrics_v9['Sharpe']:.3f} | {metrics_v9['Sharpe'] - metrics_etf['Sharpe']:+.3f} |")
    summary.append(f"| Calmar | {metrics_etf['Calmar']:.3f} | {metrics_v9['Calmar']:.3f} | {metrics_v9['Calmar'] - metrics_etf['Calmar']:+.3f} |")
    summary.append(f"| MaxDD | {metrics_etf['MaxDD']:.2%} | {metrics_v9['MaxDD']:.2%} | {metrics_v9['MaxDD'] - metrics_etf['MaxDD']:+.2%} |")
    summary.append(f"| AnnRet | {metrics_etf['AnnRet']:.2%} | {metrics_v9['AnnRet']:.2%} | {metrics_v9['AnnRet'] - metrics_etf['AnnRet']:+.2%} |")
    summary.append("")
    summary.append("## 成本敏感性 (v9)")
    summary.append("")
    summary.append("| 成本 (bp) | Sharpe | Calmar | MaxDD | AnnRet |")
    summary.append("|-----------|--------|--------|-------|--------|")
    for _, row in cost_df.iterrows():
        summary.append(f"| {int(row['cost_bps'])} | {row['Sharpe']:.3f} | {row['Calmar']:.3f} | {row['MaxDD']:.2%} | {row['AnnRet']:.2%} |")
    summary.append("")
    summary.append("## 决策点")
    summary.append("")
    target_sharpe = 0.5
    if metrics_v9["Sharpe"] >= target_sharpe:
        summary.append(f"✅ **通过**: Sharpe {metrics_v9['Sharpe']:.3f} >= 目标 {target_sharpe}")
        summary.append("   建议: 继续 Phase 3 (完整集成 v7.14 + v8)")
    else:
        summary.append(f"⚠️ **未达标**: Sharpe {metrics_v9['Sharpe']:.3f} < 目标 {target_sharpe}")
        summary.append("   建议: 接受部分完成, 仅交付 CPD")
    summary.append("")

    summary_path = output_dir / "backtest_summary.md"
    summary_path.write_text("\n".join(summary), encoding="utf-8")
    print(f"  总结: {summary_path}")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"  - backtest_results.csv")
    print(f"  - cost_sensitivity.csv")
    print(f"  - nav_comparison.png")
    print(f"  - score_timeseries.png")
    print(f"  - backtest_summary.md")

    return {
        "metrics_v9": metrics_v9,
        "metrics_etf": metrics_etf,
        "nav_v9": nav_v9,
        "nav_etf": nav_etf,
        "score": total_score,
        "signal": signal,
    }


if __name__ == "__main__":
    main()