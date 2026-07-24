# coding=utf-8
"""scripts/v9/v9_factor_galaxy_etf.py — 银河因子配置 (扩展到 43 个 ETF).

数据:
  - 43 个 ETF 周频 (2018-2026)
  - 17 个宏观因子 (周频)

对比策略:
  1. 等权基准 (43 ETF)
  2. 60/40 股债 ETF
  3. 基础风险平价
  4. 银河因子配置 (本次实施)

用法:
    python3.11 scripts/v9/v9_factor_galaxy_etf.py
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
    run_factor_allocator,
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
    etf = pd.read_parquet(data_dir / "v7_10_Y_weekly.parquet")
    return macro, etf


def run_backtest_daily(weights_df, returns_df, cost_bps=5.0, freq='W'):
    """回测 (统一用 freq 参数计算年化, 避免时间起点不一致)."""
    from QuantNodes.strategy.momentum_etf_rotation.v9.backtest import run_backtest

    common_cols = [c for c in returns_df.columns if c in weights_df.columns]
    w = weights_df[common_cols].copy()
    r = returns_df[common_cols].copy()

    common_idx = w.index.intersection(r.index)
    w = w.loc[common_idx].fillna(0)
    r = r.loc[common_idx].fillna(0)

    nav, ret, metrics = run_backtest(w, r, cost_bps=cost_bps)
    return nav, ret, metrics


def main():
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v9"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v9 银河因子配置 (43 ETF 扩展)")
    print("=" * 70)

    macro, etf = load_data()
    print(f"\n数据:")
    print(f"  宏观: {macro.shape}, {macro.index.min().strftime('%Y-%m')} ~ {macro.index.max().strftime('%Y-%m')}")
    print(f"  ETF:  {etf.shape}, {etf.index.min().strftime('%Y-%m')} ~ {etf.index.max().strftime('%Y-%m')}")

    print(f"\n[Step 1] 数据预处理")
    etf_clean = etf.fillna(0)
    print(f"  ETF 收益: {etf_clean.shape}")
    print(f"  NaN 填充: {etf_clean.isna().sum().sum()}")
    print(f"  Inf 填充: {np.isinf(etf_clean.values).sum()}")
    etf_clean = etf_clean.replace([np.inf, -np.inf], 0)

    etf_count = (etf_clean != 0).sum()
    print(f"  非零 ETF 数: {(etf_count > 100).sum()}/43")
    print(f"  全部非零 ETF: {(etf_count > 0).sum()}/43")

    etf_clean = etf_clean.loc[:, etf_count > 100]
    print(f"  过滤后 ETF: {etf_clean.shape}")

    print(f"\n[Step 2] 银河因子配置 (两种变体)")
    weights, factor_score, betas, used_macro = run_factor_allocator(
        returns_df=etf_clean,
        macro_df=macro,
        lookback_score=104,
        lookback_beta=52,
        floor=0.01,
        cap=0.15,
    )
    print(f"  权重时序: {weights.shape}")
    print(f"  最终 Top-10 平均权重 (变体A: 基础风险预算):")
    avg_w = weights.mean().sort_values(ascending=False).head(10)
    for asset, w in avg_w.items():
        print(f"    {asset}: {w:.2%}")

    print(f"\n[Step 3] 回测对比")

    bond_etfs = [c for c in etf_clean.columns if c in ['511260', '511010', '511090', '159937', '159816']]
    if not bond_etfs:
        bond_etfs = [c for c in etf_clean.columns if '511' in c or '国债' in c][:3]
    stock_etfs = [c for c in etf_clean.columns if c not in bond_etfs]

    strategies = {}

    eq_w = pd.DataFrame(1.0/len(etf_clean.columns), index=etf_clean.index, columns=etf_clean.columns)
    strategies['等权基准'] = eq_w

    w_6040 = pd.DataFrame(0.0, index=etf_clean.index, columns=etf_clean.columns)
    if stock_etfs:
        w_6040[stock_etfs] = 0.6 / len(stock_etfs)
    if bond_etfs:
        w_6040[bond_etfs] = 0.4 / len(bond_etfs)
    else:
        w_6040[stock_etfs[:5]] = w_6040[stock_etfs[:5]] + 0.4 / 5
    strategies['60/40股债'] = w_6040

    ret_clean = etf_clean.replace([np.inf, -np.inf], 0).fillna(0)
    rp_w = compute_risk_parity_base(ret_clean, lookback=52)
    rp_w = rp_w.reindex(etf_clean.index, method='ffill').fillna(0)
    strategies['基础风险平价'] = rp_w

    galaxy_w = weights.reindex(etf_clean.index, method='ffill').fillna(0)
    galaxy_w = galaxy_w.div(galaxy_w.sum(axis=1).replace(0, 1), axis=0).fillna(0)
    common_cols = [c for c in etf_clean.columns if c in galaxy_w.columns]
    galaxy_w = galaxy_w[common_cols]
    strategies['银河因子配置'] = galaxy_w

    active_w = galaxy_w.copy()
    active_w[active_w.columns] = 0
    score_z = (factor_score - factor_score.rolling(52).mean()) / (factor_score.rolling(52).std() + 1e-10)
    active_level = (0.7 - 0.5 * score_z).clip(0.2, 1.0)
    for date in active_w.index:
        if date in active_level.index:
            level = active_level.loc[date]
            if not np.isnan(level):
                active_w.loc[date] = galaxy_w.loc[date] * level
    strategies['银河方案-动态仓位'] = active_w

    active_level_for_plot = active_level.reindex(etf_clean.index, method='ffill').fillna(0)
    active_level_for_plot = active_level_for_plot.where(active_level_for_plot > 0, 0)
    active_w_total = active_level_for_plot

    print(f"\n[统一时间窗口] 找所有策略共同的回测起点")
    first_valid_dates = {}
    for name, w in strategies.items():
        active = w[w.sum(axis=1) > 0.05]
        if len(active) > 0:
            first_valid_dates[name] = active.index.min()
        else:
            first_valid_dates[name] = w.index.max()

    common_start = max(first_valid_dates.values())
    print(f"  共同起点: {common_start}")

    aligned_weights = {}
    for name, w in strategies.items():
        if common_start in w.index:
            aligned_weights[name] = w.loc[common_start:].copy()
        else:
            cutoff = w.index[w.index >= common_start]
            if len(cutoff) > 0:
                aligned_weights[name] = w.loc[cutoff[0]:].copy()
            else:
                aligned_weights[name] = w.copy()

    results = []
    navs = {}
    for name, w in strategies.items():
        try:
            w_aligned = aligned_weights[name]
            nav, ret, _ = run_backtest_daily(w_aligned, etf_clean, freq='W')
            navs[name] = nav
            metrics = compute_factor_metrics(w_aligned, etf_clean, freq='W')
            metrics['strategy'] = name
            metrics['start_date'] = str(w_aligned.index.min().date())
            metrics['end_date'] = str(w_aligned.index.max().date())
            results.append(metrics)
        except Exception as e:
            print(f"  {name} 失败: {e}")

    df_results = pd.DataFrame(results)

    print(f"\n  指标对比 (统一时间窗口, 起点={common_start.date()}):")
    display_cols = ['strategy', 'start_date', 'end_date', 'Sharpe', 'Calmar', 'MaxDD', 'AnnRet', 'Vol', 'WinRate']
    print(df_results[display_cols].to_string(index=False))

    print(f"\n[Step 4] 输出报告")

    df_results.to_csv(output_dir / "factor_galaxy_etf_results.csv", index=False)

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

    ax0.set_title('v9 银河因子配置 vs 基准 (43 ETF, 2018-2026)', fontsize=14, fontweight='bold')
    ax0.set_ylabel('NAV')
    ax0.legend(loc='upper left', fontsize=10)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    ax1.plot(active_w_total.index, active_w_total.values, color='#3b82f6', linewidth=1.5, label='银河方案-动态仓位')
    ax1.set_ylabel('仓位')
    ax1.set_title('银河方案-动态仓位时序', fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "factor_galaxy_etf_backtest.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  {output_dir / 'factor_galaxy_etf_backtest.png'}")
    print(f"  {output_dir / 'factor_galaxy_etf_results.csv'}")

    report_lines = [
        "# v9 银河因子配置回测报告 (43 ETF 扩展)",
        "",
        f"> 数据: {etf_clean.index[0].strftime('%Y-%m')} ~ {etf_clean.index[-1].strftime('%Y-%m')}",
        f"> 方法: 银河证券因子配置 (5 类宏观指标 + 43 ETF)",
        f"> 资产: {len(etf_clean.columns)} 个 ETF (过滤后)",
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
        "5. 风险预算动态调整: factor_score↑ → 降仓; ↓ → 加仓",
        "6. 周频调仓, 单资产 0.5%-10%",
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
        "## 银河方案 Top-10 平均权重",
        "",
        "| 资产 | 权重 |",
        "|------|------|",
    ])
    for asset, w in avg_w.items():
        report_lines.append(f"| {asset} | {w:.2%} |")

    report_path = output_dir / "factor_galaxy_etf_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  {report_path}")

    print(f"\n{'='*70}")
    print("完成!")


if __name__ == "__main__":
    main()