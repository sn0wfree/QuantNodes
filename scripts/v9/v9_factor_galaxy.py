# coding=utf-8
"""v9 银河因子配置回测脚本.

银河证券因子配置 (5 类宏观指标 → 因子风险预算权重):
    Step 1: 5 类宏观指标映射 (17 因子 → 5 类)
    Step 2: 熵权法合成综合得分 (104 周滚动)
    Step 3: 滚动 β 回归 (52 周)
    Step 4: 因子风险预算权重反推

对比策略:
    1. 等权基准
    2. 60/40 股债
    3. 基础风险平价
    4. 银河因子配置 (本次实施)

用法:
    python3.11 scripts/v9/v9_factor_galaxy.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from QuantNodes.strategy.momentum_etf_rotation.v9.factor_allocator import (
    run_factor_allocator, map_to_categories,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.factor_galaxy import (
    compute_factor_metrics,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.dynamic_risk_parity import (
    compute_risk_parity_base,
)


def load_data():
    data_dir = REPO / "data" / "high_freq_macro"
    macro = pd.read_parquet(data_dir / "v7_6_X_macro_weekly.parquet")
    indices = pd.read_parquet(data_dir / "v9_indices_daily.parquet")
    return macro, indices


def run_backtest_daily(weights_df, returns_df, cost_bps=5.0):
    """日频回测."""
    daily_ret = returns_df.copy()
    nav = pd.Series(1.0, index=daily_ret.index)
    prev_w = pd.Series(0.0, index=daily_ret.columns)

    for i in range(len(daily_ret)):
        date = daily_ret.index[i]
        nearest = weights_df.index[weights_df.index <= date]
        w = weights_df.loc[nearest[-1]] if len(nearest) > 0 else weights_df.iloc[0]
        r = daily_ret.iloc[i]
        port_ret = (w * r).sum()
        turnover = (w - prev_w).abs().sum()
        cost = turnover * cost_bps / 10000.0
        nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret - cost) if i > 0 else 1.0
        prev_w = w.copy()
    return nav


def main():
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v9"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v9 银河因子配置回测")
    print("=" * 70)

    macro, indices = load_data()
    print(f"\n数据:")
    print(f"  宏观: {macro.shape}, {macro.index.min().strftime('%Y-%m')} ~ {macro.index.max().strftime('%Y-%m')}")
    print(f"  指数: {indices.shape}, {indices.index.min().strftime('%Y-%m')} ~ {indices.index.max().strftime('%Y-%m')}")

    print(f"\n[Step 1] 17 因子准备")
    print(f"  因子: {macro.shape}")

    print(f"\n[Step 2] 周频收益准备")
    returns_weekly = indices.resample('W').last().pct_change().dropna().fillna(0)
    print(f"  周频收益: {returns_weekly.shape}")

    print(f"\n[Step 3] 银河因子配置")
    weights, factor_score, betas, used_macro = run_factor_allocator(
        returns_df=returns_weekly,
        macro_df=macro,
        lookback_score=104,
        lookback_beta=52,
        floor=0.02,
        cap=0.20,
    )
    print(f"  权重时序: {weights.shape}")
    print(f"  最终权重 Top-5:")
    avg_w = weights.mean().sort_values(ascending=False).head(5)
    for asset, w in avg_w.items():
        print(f"    {asset}: {w:.2%}")

    print(f"\n[Step 4] 回测对比")
    returns_daily = indices.copy()

    bond_cols = [c for c in indices.columns if '债' in c]
    stock_cols = [c for c in indices.columns if any(x in c for x in ['沪深', '中证', '恒生'])]
    comm_cols = [c for c in indices.columns if any(x in c for x in ['南华', '原油', '沪金'])]

    strategies = {}

    eq_w = pd.DataFrame(1.0/len(indices.columns), index=returns_daily.index, columns=indices.columns)
    strategies['等权基准'] = eq_w

    w_6040 = pd.DataFrame(0.0, index=returns_daily.index, columns=indices.columns)
    if stock_cols:
        w_6040[stock_cols] = 0.6 / len(stock_cols)
    if bond_cols:
        w_6040[bond_cols] = 0.4 / len(bond_cols)
    strategies['60/40股债'] = w_6040

    rp_w = compute_risk_parity_base(returns_daily, lookback=260)
    rp_w = rp_w.reindex(returns_daily.index, method='ffill').fillna(0)
    strategies['基础风险平价'] = rp_w

    galaxy_daily = weights.reindex(returns_daily.index, method='ffill').fillna(0)
    galaxy_daily = galaxy_daily.div(galaxy_daily.sum(axis=1), axis=0).fillna(0)
    strategies['银河因子配置'] = galaxy_daily

    results = []
    navs = {}
    for name, w in strategies.items():
        nav = run_backtest_daily(w, returns_daily)
        navs[name] = nav
        metrics = compute_factor_metrics(w, returns_daily)
        metrics['strategy'] = name
        results.append(metrics)

    df_results = pd.DataFrame(results)

    print(f"\n  指标对比:")
    print(df_results[['strategy', 'Sharpe', 'Calmar', 'MaxDD', 'AnnRet', 'Vol', 'WinRate']].to_string(index=False))

    print(f"\n[Step 5] 输出报告")

    df_results.to_csv(output_dir / "factor_galaxy_results.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [2, 1]})

    ax0 = axes[0]
    for name, nav in navs.items():
        if name == '银河因子配置':
            color = '#3b82f6'
            lw = 2.5
        elif name == '基础风险平价':
            color = '#f59e0b'
            lw = 1.5
        elif name == '60/40股债':
            color = '#10b981'
            lw = 1.5
        else:
            color = '#94a3b8'
            lw = 1.0
        ls = '-' if name == '银河因子配置' else '--'
        ax0.plot(nav.index, nav.values, label=name, color=color, linewidth=lw, linestyle=ls, alpha=0.8)

    ax0.set_title('v9 银河因子配置 vs 基准 (2008-2026)', fontsize=14, fontweight='bold')
    ax0.set_ylabel('NAV')
    ax0.legend(loc='upper left', fontsize=10)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    w_rolling = weights.mean(axis=1)
    w_rolling.plot(ax=ax1, color='#3b82f6', linewidth=1.5, label='因子配置平均仓位')
    ax1.set_ylabel('仓位')
    ax1.set_title('银河因子配置仓位时序', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "factor_galaxy_backtest.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  {output_dir / 'factor_galaxy_backtest.png'}")
    print(f"  {output_dir / 'factor_galaxy_results.csv'}")

    report_lines = [
        "# v9 银河因子配置回测报告",
        "",
        f"> 数据: {returns_daily.index[0].strftime('%Y-%m')} ~ {returns_daily.index[-1].strftime('%Y-%m')}",
        f"> 方法: 银河证券因子配置 (5 类宏观指标 + 风险预算调整)",
        f"> 资产: {len(indices.columns)} 个指数",
        "",
        "## 5 类宏观指标映射",
        "",
        "- **消费/内需**: 宏观增长因子, 宏观通胀因子_生活端",
        "- **出口/外部**: 宏观汇率因子, dxy_logret, cn_us_spread",
        "- **工业/生产**: 宏观通胀因子_生产端",
        "- **信贷/金融**: 信用利差因子, 无风险收益率, real_rate, 期限利差因子_债, 期限利差因子_股",
        "- **风险/情绪**: vix, vix_rank20, tf_dummy, gold_oil_corr",
        "",
        "## 银河因子配置方法",
        "",
        "1. 5 类宏观指标映射 (类内等权平均)",
        "2. 熵权法合成综合得分 (104 周滚动窗口)",
        "3. 滚动 β 回归 (52 周窗口)",
        "4. 风险预算权重反推 (w ∝ |β| × target_risk × risk_scalar / σ²)",
        "5. 风险预算动态调整: factor_score↑ → 降仓防御; factor_score↓ → 加仓进攻",
        "6. 周频调仓, 单资产 2%-20%",
        "",
        "## 回测结果对比",
        "",
        "| 策略 | Sharpe | Calmar | MaxDD | AnnRet | Vol | WinRate |",
        "|------|--------|--------|-------|--------|-----|---------|",
    ]

    for _, row in df_results.iterrows():
        report_lines.append(
            f"| {row['strategy']} | {row['Sharpe']:.3f} | {row['Calmar']:.3f} | "
            f"{row['MaxDD']:.2%} | {row['AnnRet']:.2%} | {row['Vol']:.4f} | {row['WinRate']:.2%} |"
        )

    report_lines.extend([
        "",
        "## 银河方案 Top-5 平均权重",
        "",
        "| 资产 | 权重 |",
        "|------|------|",
    ])

    for _, row in df_results.iterrows():
        report_lines.append(
            f"| {row['strategy']} | {row['Sharpe']:.3f} | {row['Calmar']:.3f} | "
            f"{row['MaxDD']:.2%} | {row['AnnRet']:.2%} | {row['Vol']:.4f} | {row['WinRate']:.2%} |"
        )

    report_lines.extend([
        "",
        "## 银河方案 Top-5 平均权重",
        "",
        "| 资产 | 权重 |",
        "|------|------|",
    ])
    for asset, w in avg_w.items():
        report_lines.append(f"| {asset} | {w:.2%} |")

    report_path = output_dir / "factor_galaxy_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  {report_path}")

    print(f"\n{'='*70}")
    print("完成!")


if __name__ == "__main__":
    main()