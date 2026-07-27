# coding=utf-8
"""scripts/v10/v10_backtest_vol_parity.py — v10 4 策略 Vol-parity 组合回测.

5 个动态权重方案对比:
  A: 市场状态切换 (Regime-Based)
  B: 波动率目标 (Vol-Targeting)
  C: 回撤控制 (Drawdown Control)
  D: 信号强度加权 (Signal-Weighted)
  E: 混合方案 (Hybrid)

用法:
    python3 scripts/v10/v10_backtest_vol_parity.py
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

from QuantNodes.strategy.momentum_etf_rotation.v10.dynamic_weight_schemes import (
    load_navs,
    compute_nav,
    scheme_a_regime,
    scheme_b_vol_target,
    scheme_c_drawdown,
    scheme_d_signal_weighted,
    scheme_e_hybrid,
    BASE_WEIGHTS,
)


def performance_metrics(nav: pd.Series, freq: str = 'D') -> dict:
    """计算业绩指标."""
    n = len(nav)
    if n < 2:
        return {}

    freq_map = {"D": 252, "W": 52, "M": 12}
    periods = freq_map.get(freq, 252)

    ret = nav.pct_change().dropna()
    if len(ret) < 2:
        return {"ann_return": 0.0}

    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    n_years = n / periods
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    ann_vol = ret.std() * np.sqrt(periods)
    sharpe = (ret.mean() * periods) / ann_vol if ann_vol > 0 else 0

    running_max = nav.cummax()
    drawdown = nav / running_max - 1
    max_dd = float(drawdown.min())

    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0

    win_rate = (ret > 0).sum() / len(ret) if len(ret) > 0 else 0

    return {
        "ann_return": float(ann_ret),
        "total_return": float(total_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "win_rate": float(win_rate),
        "final_nav": float(nav.iloc[-1]),
    }


def main():
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v10"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v10 4 策略 Vol-parity 组合回测")
    print("=" * 70)

    # 加载 4 策略 NAV
    print("\n加载 4 策略 NAV...")
    prices = load_navs()
    print(f"时间范围: {prices.index.min()} ~ {prices.index.max()}")
    print(f"策略数: {len(prices.columns)}")
    print(f"策略: {list(prices.columns)}")

    # === 运行 6 个方案 ===
    schemes = {
        "静态 Vol-parity": pd.DataFrame(BASE_WEIGHTS, index=prices.index),
        "方案 A: 市场状态": scheme_a_regime(prices),
        "方案 B: 波动率目标": scheme_b_vol_target(prices),
        "方案 C: 回撤控制": scheme_c_drawdown(prices),
        "方案 D: 信号加权": scheme_d_signal_weighted(prices),
        "方案 E: 混合": scheme_e_hybrid(prices),
    }

    results = []
    navs = {}

    for name, weights in schemes.items():
        print(f"\n运行 {name}...")
        nav = compute_nav(prices, weights, cost_bp=10)
        navs[name] = nav
        m = performance_metrics(nav, freq='D')
        results.append({"方案": name, **m})
        print(f"  Sharpe: {m['sharpe']:.3f}, Calmar: {m['calmar']:.3f}, "
              f"年化: {m['ann_return']:.2%}, MaxDD: {m['max_drawdown']:.2%}")

    # === 输出结果 ===
    df_results = pd.DataFrame(results)
    print(f"\n{'=' * 70}")
    print("回测结果对比")
    print("=" * 70)
    print(df_results.to_string(index=False))

    # 保存
    df_results.to_csv(output_dir / "v10_vol_parity_results.csv", index=False)
    print(f"\n保存: {output_dir / 'v10_vol_parity_results.csv'}")

    # === 画图 ===
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # NAV 对比
    ax = axes[0]
    for name, nav in navs.items():
        ax.plot(nav.index, nav.values, label=name, linewidth=1.5, alpha=0.8)
    ax.set_title('v10 4-Strategy Vol-parity: NAV Comparison', fontsize=14)
    ax.set_ylabel('NAV')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # 回撤对比
    ax = axes[1]
    for name, nav in navs.items():
        dd = nav / nav.cummax() - 1
        ax.plot(dd.index, dd.values, label=name, linewidth=1.2, alpha=0.7)
    ax.set_title('Drawdown Comparison', fontsize=14)
    ax.set_ylabel('Drawdown')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "v10_vol_parity_comparison.png", dpi=120, bbox_inches='tight')
    print(f"图片: {output_dir / 'v10_vol_parity_comparison.png'}")
    plt.close()

    # === 输出 Markdown 报告 ===
    report_lines = [
        "# v10 4 策略 Vol-parity 组合回测报告",
        "",
        f"> 数据: {prices.index.min()} ~ {prices.index.max()}",
        f"> 策略: {list(prices.columns)}",
        f"> 成本: 10bp",
        "",
        "## 回测结果",
        "",
        "| 方案 | Sharpe | Calmar | 年化 | MaxDD | 胜率 | 总收益 |",
        "|------|--------|--------|------|-------|------|--------|",
    ]

    for _, row in df_results.iterrows():
        report_lines.append(
            f"| {row['方案']} | {row['sharpe']:.3f} | {row['calmar']:.3f} | "
            f"{row['ann_return']:.2%} | {row['max_drawdown']:.2%} | "
            f"{row['win_rate']:.2%} | {row['total_return']:.2%} |"
        )

    report_lines.extend([
        "",
        "## 基础权重",
        "",
        f"- v1.0: {BASE_WEIGHTS['v1.0']:.0%}",
        f"- v9macro: {BASE_WEIGHTS['v9macro']:.0%}",
        f"- v7.10: {BASE_WEIGHTS['v7.10']:.0%}",
        f"- DualMom: {BASE_WEIGHTS['DualMom']:.0%}",
    ])

    report_path = output_dir / "v10_vol_parity_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"报告: {report_path}")
    print(f"\n完成!")


if __name__ == "__main__":
    main()
