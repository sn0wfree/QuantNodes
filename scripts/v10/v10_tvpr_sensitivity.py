# coding=utf-8
"""scripts/v10/v10_tvpr_sensitivity.py — TV-PR 权重敏感性测试.

测试不同 TV-PR 权重 (0%, 25%, 50%, 75%, 100%) 对 v10 表现的影响.

参数:
    tvpr_weight: TV-PR 信号在 Layer 1 中的占比
        0.0   = 仅熵权法
        0.25  = TV-PR 25% + 熵权 75%
        0.5   = TV-PR 50% + 熵权 50% (当前默认)
        0.75  = TV-PR 75% + 熵权 25%
        1.0   = 仅 TV-PR

输出:
    reports/momentum_etf_rotation/v10/v10_tvpr_sensitivity.csv
    reports/momentum_etf_rotation/v10/v10_tvpr_sensitivity_report.md
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from QuantNodes.strategy.momentum_etf_rotation.v10 import (
    V10Config,
    MacroLayerConfig,
    run_v10_backtest,
)


def load_data():
    """加载数据."""
    data_dir = REPO / "data" / "high_freq_macro"
    etf = pd.read_parquet(data_dir / "v7_10_Y_weekly.parquet")
    etf_clean = etf.fillna(0).replace([np.inf, -np.inf], 0)
    etf_count = (etf_clean != 0).sum()
    etf_clean = etf_clean.loc[:, etf_count > 100]
    macro = pd.read_parquet(data_dir / "v7_6_X_macro_weekly.parquet")

    # v9 窗口 (与 v9 银河方案-动态仓位对比)
    etf_v9 = etf_clean.loc['2021-08-01':]
    macro_v9 = macro.loc['2021-08-01':]

    # 完整窗口 (评估稳健性)
    return etf_v9, macro_v9, etf_clean, macro


def run_with_tvpr_weight(returns_df, macro_df, tvpr_weight, freq='W'):
    """跑指定 TV-PR 权重的 v10."""
    cfg = V10Config()
    cfg.rebal_freq = freq
    cfg.macro.tvpr_weight = tvpr_weight

    try:
        result = run_v10_backtest(returns_df, macro_df, cfg)
        return result
    except Exception as e:
        print(f"  [FAIL] tvpr_weight={tvpr_weight}: {e}")
        return None


def main():
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v10"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v10 TV-PR 权重敏感性测试")
    print("=" * 70)

    etf_v9, macro_v9, etf_full, macro_full = load_data()
    print(f"\nv9 窗口: {etf_v9.shape[0]} 周, {etf_v9.shape[1]} ETF")
    print(f"完整窗口: {etf_full.shape[0]} 周, {etf_full.shape[1]} ETF")

    # 测试的 TV-PR 权重
    tvpr_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # === Test 1: v9 窗口 (周频) ===
    print(f"\n{'=' * 70}")
    print("Test 1: v9 窗口 (周频)")
    print("=" * 70)

    results_v9 = []
    for w in tvpr_weights:
        print(f"  tvpr_weight = {w:.1f}...", end='', flush=True)
        result = run_with_tvpr_weight(etf_v9, macro_v9, w, 'W')
        if result is not None:
            m = result.metrics
            results_v9.append({
                'tvpr_weight': w,
                '窗口': 'v9 (W)',
                **m,
            })
            print(f" Sharpe={m['sharpe']:.3f}, 年化={m['ann_return']:.2%}, MaxDD={m['max_drawdown']:.2%}")
        else:
            print(" FAILED")

    # === Test 2: v9 窗口 (月频) ===
    print(f"\n{'=' * 70}")
    print("Test 2: v9 窗口 (月频)")
    print("=" * 70)

    results_v9_m = []
    for w in tvpr_weights:
        print(f"  tvpr_weight = {w:.1f}...", end='', flush=True)
        result = run_with_tvpr_weight(etf_v9, macro_v9, w, 'M')
        if result is not None:
            m = result.metrics
            results_v9_m.append({
                'tvpr_weight': w,
                '窗口': 'v9 (M)',
                **m,
            })
            print(f" Sharpe={m['sharpe']:.3f}, 年化={m['ann_return']:.2%}, MaxDD={m['max_drawdown']:.2%}")
        else:
            print(" FAILED")

    # === Test 3: 完整窗口 (评估稳健性) ===
    print(f"\n{'=' * 70}")
    print("Test 3: 完整窗口 8.4 年 (周频, 评估稳健性)")
    print("=" * 70)

    results_full = []
    for w in tvpr_weights:
        print(f"  tvpr_weight = {w:.1f}...", end='', flush=True)
        result = run_with_tvpr_weight(etf_full, macro_full, w, 'W')
        if result is not None:
            m = result.metrics
            results_full.append({
                'tvpr_weight': w,
                '窗口': '完整 (W)',
                **m,
            })
            print(f" Sharpe={m['sharpe']:.3f}, 年化={m['ann_return']:.2%}, MaxDD={m['max_drawdown']:.2%}")
        else:
            print(" FAILED")

    # === 汇总结果 ===
    all_results = results_v9 + results_v9_m + results_full
    if not all_results:
        print("\n所有测试都失败, 无法生成报告")
        return

    df = pd.DataFrame(all_results)

    # 标准化列名
    df = df.rename(columns={
        'sharpe': 'Sharpe',
        'calmar': 'Calmar',
        'ann_return': '年化',
        'max_drawdown': 'MaxDD',
        'total_return': '总收益',
        'win_rate': '胜率',
    })

    print(f"\n{'=' * 70}")
    print("完整结果 (按 tvpr_weight 排序)")
    print("=" * 70)

    display_cols = ['窗口', 'tvpr_weight', 'Sharpe', 'Calmar', '年化', 'MaxDD', '总收益', '胜率']
    available = [c for c in display_cols if c in df.columns]
    print(df[available].to_string(index=False))

    # 保存 CSV
    df.to_csv(output_dir / "v10_tvpr_sensitivity.csv", index=False)
    print(f"\n保存: {output_dir / 'v10_tvpr_sensitivity.csv'}")

    # === 分析最优权重 ===
    print(f"\n{'=' * 70}")
    print("最优权重分析")
    print("=" * 70)

    # 按窗口分别找最优
    best_by_window = {}
    for window in df['窗口'].unique():
        sub = df[df['窗口'] == window]
        best_idx = sub['Sharpe'].idxmax()
        best = sub.loc[best_idx]
        best_by_window[window] = best
        print(f"  {window}: 最优 tvpr_weight = {best['tvpr_weight']:.1f}, "
              f"Sharpe = {best['Sharpe']:.3f}, 年化 = {best['年化']:.2%}")

    # 全局最优 (按平均 Sharpe)
    avg_sharpe = df.groupby('tvpr_weight')['Sharpe'].mean().sort_values(ascending=False)
    print(f"\n  全局最优 (按平均 Sharpe):")
    for tvpr_w, avg_sh in avg_sharpe.head(3).items():
        print(f"    tvpr_weight={tvpr_w:.1f}: 平均 Sharpe={avg_sh:.3f}")

    global_best_w = avg_sharpe.index[0]
    print(f"\n  ★ 推荐 tvpr_weight = {global_best_w:.1f} (平均 Sharpe 最高)")

    # === 画图 ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Sharpe vs tvpr_weight
    ax = axes[0, 0]
    for window in df['窗口'].unique():
        sub = df[df['窗口'] == window]
        ax.plot(sub['tvpr_weight'], sub['Sharpe'], marker='o', label=window, linewidth=2)
    ax.set_xlabel('TV-PR Weight')
    ax.set_ylabel('Sharpe')
    ax.set_title('Sharpe vs TV-PR Weight')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axvline(x=global_best_w, color='red', linestyle='--', alpha=0.5, label='推荐')

    # 2. Calmar vs tvpr_weight
    ax = axes[0, 1]
    for window in df['窗口'].unique():
        sub = df[df['窗口'] == window]
        ax.plot(sub['tvpr_weight'], sub['Calmar'], marker='o', label=window, linewidth=2)
    ax.set_xlabel('TV-PR Weight')
    ax.set_ylabel('Calmar')
    ax.set_title('Calmar vs TV-PR Weight')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 年化 vs tvpr_weight
    ax = axes[1, 0]
    for window in df['窗口'].unique():
        sub = df[df['窗口'] == window]
        ax.plot(sub['tvpr_weight'], sub['年化'], marker='o', label=window, linewidth=2)
    ax.set_xlabel('TV-PR Weight')
    ax.set_ylabel('年化收益')
    ax.set_title('Annual Return vs TV-PR Weight')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. MaxDD vs tvpr_weight
    ax = axes[1, 1]
    for window in df['窗口'].unique():
        sub = df[df['窗口'] == window]
        ax.plot(sub['tvpr_weight'], sub['MaxDD'], marker='o', label=window, linewidth=2)
    ax.set_xlabel('TV-PR Weight')
    ax.set_ylabel('Max Drawdown')
    ax.set_title('Max Drawdown vs TV-PR Weight')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "v10_tvpr_sensitivity.png", dpi=120, bbox_inches='tight')
    print(f"\n图片: {output_dir / 'v10_tvpr_sensitivity.png'}")
    plt.close()

    # === 生成报告 ===
    report_lines = [
        "# v10 TV-PR 权重敏感性测试",
        "",
        f"> 测试日期: 2026-07-23",
        f"> 数据窗口:",
        f"  - v9 窗口: 2021-08-01 ~ 2026-05-31 ({etf_v9.shape[0]} 周)",
        f"  - 完整窗口: 2018-01 ~ 2026-05 ({etf_full.shape[0]} 周)",
        f"> 资产: {etf_v9.shape[1]} ETF",
        "",
        "## 测试说明",
        "",
        "TV-PR 权重 tvpr_weight ∈ [0, 1]:",
        "- 0.0 = Layer 1 仅用熵权法 (5 宏观因子 z-score)",
        "- 1.0 = Layer 1 仅用 TV-PR 时变β",
        "- 0.5 = 50/50 混合 (默认)",
        "",
        "其他配置保持 v10.0 默认值:",
        "- rebal_freq: W/M",
        "- Top-K=10, candidate_pool=50%",
        "- Jump Model: 启用, bear_prob × 0.5 调整仓位",
        "- 动态仓位: pos = (0.7 - 0.5z).clip(0.2, 1.0)",
        "",
        "## 完整结果",
        "",
        "| 窗口 | tvpr_weight | Sharpe | Calmar | 年化 | MaxDD | 总收益 | 胜率 |",
        "|------|-------------|--------|--------|------|-------|--------|------|",
    ]

    for _, row in df.iterrows():
        report_lines.append(
            f"| {row['窗口']} | {row['tvpr_weight']:.1f} | "
            f"{row['Sharpe']:.3f} | {row['Calmar']:.3f} | "
            f"{row['年化']:.2%} | {row['MaxDD']:.2%} | "
            f"{row['总收益']:.2%} | {row['胜率']:.2%} |"
        )

    report_lines.extend([
        "",
        "## 各窗口最优权重",
        "",
    ])

    for window, best in best_by_window.items():
        report_lines.append(
            f"- **{window}**: tvpr_weight = **{best['tvpr_weight']:.1f}**, "
            f"Sharpe = **{best['Sharpe']:.3f}**, 年化 = {best['年化']:.2%}, MaxDD = {best['MaxDD']:.2%}"
        )

    report_lines.extend([
        "",
        "## 全局最优 (跨窗口平均)",
        "",
        "| tvpr_weight | 平均 Sharpe | 平均 Calmar | 平均年化 |",
        "|-------------|-------------|-------------|----------|",
    ])

    for tvpr_w in sorted(avg_sharpe.index):
        sub = df[df['tvpr_weight'] == tvpr_w]
        avg_c = sub['Calmar'].mean()
        avg_a = sub['年化'].mean()
        marker = " ⭐" if tvpr_w == global_best_w else ""
        report_lines.append(
            f"| {tvpr_w:.1f}{marker} | {avg_sharpe[tvpr_w]:.3f} | {avg_c:.3f} | {avg_a:.2%} |"
        )

    report_lines.extend([
        "",
        f"## ★ 推荐配置",
        "",
        f"**`tvpr_weight = {global_best_w:.1f}`** (跨窗口平均 Sharpe 最高)",
        "",
        "理由:",
        f"1. 在 v9 (W) 窗口: Sharpe {best_by_window['v9 (W)']['Sharpe']:.3f}",
        f"2. 在 v9 (M) 窗口: Sharpe {best_by_window['v9 (M)']['Sharpe']:.3f}",
        f"3. 在完整窗口: Sharpe {best_by_window['完整 (W)']['Sharpe']:.3f}",
        f"4. 三窗口平均 Sharpe {avg_sharpe[global_best_w]:.3f}",
        "",
        "## 调优结论",
        "",
    ])

    # 分析趋势
    avg_sharpe_sorted = avg_sharpe.sort_values(ascending=False)
    top3 = avg_sharpe_sorted.head(3)
    top3_range = (top3.index.max() - top3.index.min()) * 100

    report_lines.append(
        f"1. **最优区间**: TV-PR 权重在 {top3.index.min():.1f} - {top3.index.max():.1f} 之间, "
        f"差异 {top3_range:.0f}%, 说明权重敏感度{'较低' if top3_range < 30 else '较高'}"
    )
    report_lines.append(
        f"2. **极端值对比**:",
    )

    if 0.0 in avg_sharpe.index and 1.0 in avg_sharpe.index:
        report_lines.append(
            f"   - 纯熵权 (0.0): 平均 Sharpe {avg_sharpe[0.0]:.3f}"
        )
        report_lines.append(
            f"   - 纯 TV-PR (1.0): 平均 Sharpe {avg_sharpe[1.0]:.3f}"
        )
        report_lines.append(
            f"   - 混合 ({global_best_w:.1f}): 平均 Sharpe {avg_sharpe[global_best_w]:.3f}, "
            f"比纯{'熵权' if avg_sharpe[global_best_w] > avg_sharpe[0.0] else 'TV-PR'}高 "
            f"{abs(avg_sharpe[global_best_w] - max(avg_sharpe[0.0], avg_sharpe[1.0])):.3f}"
        )

    report_lines.extend([
        "",
        "## 应用建议",
        "",
        f"将 v10 默认 `MacroLayerConfig.tvpr_weight` 从 0.5 改为 **{global_best_w:.1f}**.",
        "",
        "代码改动:",
        "```python",
        f"# QuantNodes/strategy/momentum_etf_rotation/v10/config_v10.py",
        f"tvpr_weight: float = {global_best_w:.1f}    # 调优后 (原 0.5)",
        "```",
        "",
        "## 文件清单",
        "",
        "- `v10_tvpr_sensitivity.csv`: 33 个组合完整数据",
        "- `v10_tvpr_sensitivity_report.md`: 本报告",
        "- `v10_tvpr_sensitivity.png`: 4 子图可视化",
    ])

    report_path = output_dir / "v10_tvpr_sensitivity_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n报告: {report_path}")

    print(f"\n{'=' * 70}")
    print("测试完成!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()