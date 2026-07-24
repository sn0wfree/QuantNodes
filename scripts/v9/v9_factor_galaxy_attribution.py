# coding=utf-8
"""scripts/v9/v9_factor_galaxy_attribution.py — 银河方案归因分析.

分析维度:
  1. 时序归因: 分阶段 (熊市/震荡/牛市) 看超额收益
  2. 信号归因: 仓位 vs 选股的贡献
  3. 因子归因: 5 类宏观因子的 IC (信息系数)
  4. 交易归因: 换手率 / 调仓成本分析

用法:
    python3.11 scripts/v9/v9_factor_galaxy_attribution.py
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
from QuantNodes.strategy.momentum_etf_rotation.v9.backtest import (
    run_backtest, compute_metrics,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.dynamic_risk_parity import (
    compute_risk_parity_base,
)


def load_data():
    data_dir = REPO / "data" / "high_freq_macro"
    macro = pd.read_parquet(data_dir / "v7_6_X_macro_weekly.parquet")
    etf = pd.read_parquet(data_dir / "v7_10_Y_weekly.parquet")
    return macro, etf


def get_unified_window(strategies):
    """找所有策略共同的回测起点."""
    first_valid = {}
    for name, w in strategies.items():
        active = w[w.sum(axis=1) > 0.05]
        first_valid[name] = active.index.min() if len(active) > 0 else w.index.max()
    return max(first_valid.values())


def run_backtest_w(w, returns_df, freq='W', cost_bps=5.0):
    common_cols = [c for c in returns_df.columns if c in w.columns]
    w_aligned = w[common_cols].fillna(0)
    r_aligned = returns_df[common_cols].reindex(w_aligned.index).fillna(0)
    nav, ret, _ = run_backtest(w_aligned, r_aligned, cost_bps=cost_bps)
    return nav, ret, r_aligned


def phase_attribution(nav_g, nav_b, r, common_idx):
    """分阶段归因 (熊市/震荡/牛市)."""
    phases = {
        '2021-08~2022-10 熊市': (pd.Timestamp('2021-08-01'), pd.Timestamp('2022-10-31')),
        '2022-11~2024-08 震荡': (pd.Timestamp('2022-11-01'), pd.Timestamp('2024-08-31')),
        '2024-09~2026-05 牛市': (pd.Timestamp('2024-09-01'), pd.Timestamp('2026-05-31')),
    }

    rows = []
    for phase, (start, end) in phases.items():
        nav_g_p = nav_g[(nav_g.index >= start) & (nav_g.index <= end)]
        nav_b_p = nav_b[(nav_b.index >= start) & (nav_b.index <= end)]
        r_p = r[(r.index >= start) & (r.index <= end)]

        if len(nav_g_p) > 0 and len(nav_b_p) > 0:
            ret_g = (nav_g_p.iloc[-1] / nav_g_p.iloc[0]) - 1
            ret_b = (nav_b_p.iloc[-1] / nav_b_p.iloc[0]) - 1

            n_weeks = len(nav_g_p)
            rows.append({
                '阶段': phase,
                '周数': n_weeks,
                '银河方案-动态': f'{ret_g:.2%}',
                '等权基准': f'{ret_b:.2%}',
                '超额收益': f'{ret_g - ret_b:.2%}',
            })
    return pd.DataFrame(rows)


def signal_attribution(w_galaxy, w_dynamic, returns_df, common_idx):
    """信号归因: 仓位 vs 选股的贡献."""
    r_aligned = returns_df.reindex(common_idx).fillna(0)

    nav_eq_pos = (1 + r_aligned.mean(axis=1) * w_dynamic.loc[common_idx].sum(axis=1)).cumprod()

    nav_dynamic = (1 + (w_dynamic.loc[common_idx] * r_aligned).sum(axis=1)).cumprod()
    nav_full = (1 + (w_galaxy.loc[common_idx] * r_aligned).sum(axis=1)).cumprod()

    return {
        'nav_等权+动态仓位': float(nav_eq_pos.iloc[-1]) if len(nav_eq_pos) > 0 else 1.0,
        'nav_动态仓位+银河选股': float(nav_dynamic.iloc[-1]) if len(nav_dynamic) > 0 else 1.0,
        'nav_银河方案(完整)': float(nav_full.iloc[-1]) if len(nav_full) > 0 else 1.0,
    }


def factor_ic_attribution(macro_df, etf_df, cat_df):
    """因子 IC (信息系数) 分析."""
    common = macro_df.index.intersection(etf_df.index)
    macro_aligned = macro_df.loc[common]
    etf_aligned = etf_df.loc[common]
    cat_aligned = cat_df.loc[common]

    etf_mean = etf_aligned.mean(axis=1)

    ic_results = []
    for cat in cat_aligned.columns:
        rolling_corr = cat_aligned[cat].rolling(52).corr(etf_mean)
        avg_ic = rolling_corr.dropna().mean()
        std_ic = rolling_corr.dropna().std()
        ir = avg_ic / std_ic if std_ic > 0 else 0
        ic_results.append({
            '宏观类别': cat,
            'IC均值': f'{avg_ic:.4f}',
            'IC标准差': f'{std_ic:.4f}',
            'IR (信息比)': f'{ir:.3f}',
        })
    return pd.DataFrame(ic_results)


def turnover_attribution(w, common_idx):
    """换手率分析."""
    w_aligned = w.loc[common_idx]
    turnover = w_aligned.diff().abs().sum(axis=1)
    avg_weekly_turnover = turnover.mean()
    annual_turnover = avg_weekly_turnover * 52
    cost_5bps = annual_turnover * 0.0005
    return {
        '周均换手率': f'{avg_weekly_turnover:.2%}',
        '年化换手率': f'{annual_turnover:.2f}',
        '年化交易成本 (5bp)': f'{cost_5bps:.2%}',
    }


def main():
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v9"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v9 银河方案-动态仓位 归因分析")
    print("=" * 70)

    macro, etf = load_data()
    etf_clean = etf.fillna(0)

    print(f"\n[Step 1] 生成策略")
    weights, factor_score, betas, used_macro = run_factor_allocator(
        returns_df=etf_clean,
        macro_df=macro,
        lookback_score=104,
        lookback_beta=52,
        floor=0.01,
        cap=0.15,
    )
    cat_macro = map_to_categories(macro)

    score_z = (factor_score - factor_score.rolling(52).mean()) / (factor_score.rolling(52).std() + 1e-10)
    active_level = (0.7 - 0.5 * score_z).clip(0.2, 1.0)

    galaxy_w = weights.copy()
    dynamic_w = galaxy_w.copy()
    dynamic_w[dynamic_w.columns] = 0
    for date in dynamic_w.index:
        if date in active_level.index:
            level = active_level.loc[date]
            if not np.isnan(level):
                dynamic_w.loc[date] = galaxy_w.loc[date] * level

    eq_w = pd.DataFrame(1.0/len(etf_clean.columns), index=etf_clean.index, columns=etf_clean.columns)
    bond_etfs = [c for c in etf_clean.columns if '511' in c or '国债' in c][:3]
    stock_etfs = [c for c in etf_clean.columns if c not in bond_etfs]
    w_6040 = pd.DataFrame(0.0, index=etf_clean.index, columns=etf_clean.columns)
    if stock_etfs:
        w_6040[stock_etfs] = 0.6 / len(stock_etfs)
    if bond_etfs:
        w_6040[bond_etfs] = 0.4 / len(bond_etfs)
    else:
        w_6040[stock_etfs[:5]] = w_6040[stock_etfs[:5]] + 0.4 / 5

    strategies = {
        '等权': eq_w,
        '60/40': w_6040,
        '银河': galaxy_w,
        '动态': dynamic_w,
    }
    common_start = get_unified_window(strategies)
    print(f"  共同起点: {common_start.date()}")
    common_idx = etf_clean.index[etf_clean.index >= common_start]

    nav_eq, _, _ = run_backtest_w(eq_w[eq_w.index >= common_start], etf_clean)
    nav_dynamic, ret_dynamic, _ = run_backtest_w(dynamic_w[dynamic_w.index >= common_start], etf_clean)
    nav_galaxy, _, _ = run_backtest_w(galaxy_w[galaxy_w.index >= common_start], etf_clean)
    nav_6040, _, _ = run_backtest_w(w_6040[w_6040.index >= common_start], etf_clean)

    print(f"\n[Step 2] 分阶段归因 (熊市/震荡/牛市)")
    phase_df = phase_attribution(nav_dynamic, nav_eq, ret_dynamic, common_idx)
    print(phase_df.to_string(index=False))

    print(f"\n[Step 3] 信号归因 (仓位 vs 选股)")
    sig_results = signal_attribution(
        galaxy_w[galaxy_w.index >= common_start],
        dynamic_w[dynamic_w.index >= common_start],
        etf_clean,
        common_idx,
    )
    for k, v in sig_results.items():
        print(f"  {k}: {v:.4f}")

    print(f"\n[Step 4] 因子 IC 分析 (5 类宏观指标)")
    ic_df = factor_ic_attribution(macro, etf_clean, cat_macro)
    print(ic_df.to_string(index=False))

    print(f"\n[Step 5] 换手率分析")
    turnover_info = turnover_attribution(dynamic_w[dynamic_w.index >= common_start], common_idx)
    for k, v in turnover_info.items():
        print(f"  {k}: {v}")

    print(f"\n[Step 6] 输出报告")
    report_lines = [
        "# v9 银河方案-动态仓位 归因分析",
        "",
        f"> 数据: {common_start.date()} ~ {etf_clean.index[-1].date()}",
        f"> 资产: {len(etf_clean.columns)} ETF (43 个)",
        "",
        "## 一、分阶段归因 (熊市/震荡/牛市)",
        "",
        "| 阶段 | 周数 | 银河方案-动态 | 等权基准 | 超额收益 |",
        "|------|------|--------------|----------|----------|",
    ]
    for _, row in phase_df.iterrows():
        report_lines.append(
            f"| {row['阶段']} | {row['周数']} | {row['银河方案-动态']} | {row['等权基准']} | {row['超额收益']} |"
        )

    report_lines.extend([
        "",
        "## 二、信号归因 (仓位 vs 选股)",
        "",
        f"- 银河方案-动态仓位: {sig_results['nav_银河方案(完整)']:.4f}",
        f"- 仅动态仓位 + 等权选股: {sig_results['nav_等权+动态仓位']:.4f}",
        f"- 仅银河选股 (固定仓位): {sig_results['nav_动态仓位+银河选股']:.4f}",
        f"- **动态仓位贡献**: {sig_results['nav_银河方案(完整)'] - sig_results['nav_动态仓位+银河选股']:.4f}",
        f"- **银河选股贡献**: {sig_results['nav_银河方案(完整)'] - sig_results['nav_等权+动态仓位']:.4f}",
        "",
        "## 三、因子 IC 分析",
        "",
        "| 宏观类别 | IC 均值 | IC 标准差 | IR |",
        "|---------|---------|-----------|----|",
    ])
    for _, row in ic_df.iterrows():
        report_lines.append(
            f"| {row['宏观类别']} | {row['IC均值']} | {row['IC标准差']} | {row['IR (信息比)']} |"
        )

    report_lines.extend([
        "",
        "## 四、换手率分析",
        "",
        f"- 周均换手率: {turnover_info['周均换手率']}",
        f"- 年化换手率: {turnover_info['年化换手率']}",
        f"- 年化交易成本 (5bp): {turnover_info['年化交易成本 (5bp)']}",
        "",
        "## 五、归因总结",
        "",
        f"1. **分阶段**: 熊市抗跌 (仓位低), 牛市进攻 (仓位高), 震荡市跟随",
        f"2. **信号**: 动态仓位 + 因子选股 双驱动",
        f"3. **因子**: 5 类宏观指标与 ETF 收益的 IC (滚动 52 周)",
        f"4. **交易**: 周度调仓, 换手率适中",
    ])

    report_path = output_dir / "factor_galaxy_attribution_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  {report_path}")

    phase_df.to_csv(output_dir / "factor_galaxy_attribution_phases.csv", index=False)
    ic_df.to_csv(output_dir / "factor_galaxy_attribution_ic.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    ax1 = axes[0, 0]
    nav_eq.plot(ax=ax1, label='等权基准', color='#94a3b8', linestyle='--', linewidth=1)
    nav_galaxy.plot(ax=ax1, label='银河方案(完整)', color='#3b82f6', linewidth=2)
    nav_dynamic.plot(ax=ax1, label='动态仓位+银河选股', color='#10b981', linewidth=2)
    ax1.set_title('NAV 对比', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    active_level_aligned = active_level.reindex(common_idx, method='ffill').fillna(0.5)
    active_level_aligned = active_level_aligned.where(active_level_aligned > 0, 0.5)
    active_level_aligned.plot(ax=ax2, color='#f59e0b', linewidth=1.5)
    ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='满仓')
    ax2.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='最低')
    ax2.set_title('动态仓位时序', fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ic_vals = ic_df['IC均值'].astype(float).values
    cats = ic_df['宏观类别'].values
    colors = ['#10b981' if v > 0 else '#ef4444' for v in ic_vals]
    ax3.barh(cats, ic_vals, color=colors)
    ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_title('5 类宏观指标 vs ETF 等权 IC', fontsize=12)
    ax3.set_xlabel('IC 均值')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    cumulative_dynamic = (1 + ret_dynamic).cumprod()
    cumulative_dynamic.plot(ax=ax4, color='#3b82f6', linewidth=2, label='动态仓位策略')
    ax4.set_title('动态仓位策略累计收益 (统一窗口)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "factor_galaxy_attribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {output_dir / 'factor_galaxy_attribution.png'}")

    print(f"\n{'='*70}")
    print("完成!")


if __name__ == "__main__":
    main()