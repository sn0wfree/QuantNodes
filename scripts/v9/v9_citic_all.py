# coding=utf-8
"""scripts/v9/v9_citic_all.py — 中信 4 策略 vs 5 原策略 + 银河方案 综合对比.

对比策略:
  原版 5 策略 (来自 v9_factor_galaxy_etf.py):
    - 等权基准
    - 60/40股债
    - 基础风险平价
    - 银河因子配置
    - 银河方案-动态仓位

  中信 4 策略 (本次):
    - 中信里昂全天候 (All-Weather)
    - 中信大类资产配置 (Macro 5 因子)
    - 中信多因子选股
    - 中信行业轮动 (动量+质量)

输出:
  - reports/momentum_etf_rotation/v9/citic_all_results.csv
  - reports/momentum_etf_rotation/v9/citic_all_backtest.png
  - reports/momentum_etf_rotation/v9/citic_all_report.md
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
from QuantNodes.strategy.momentum_etf_rotation.v9.dynamic_risk_parity import (
    compute_risk_parity_base,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.factor_galaxy import (
    compute_factor_metrics,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.backtest import run_backtest
from QuantNodes.strategy.momentum_etf_rotation.v9.citic_all_weather import (
    run_all_weather,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.citic_macro import (
    run_macro_allocation,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.citic_multifactor import (
    build_multifactor_weights,
)
from QuantNodes.strategy.momentum_etf_rotation.v9.citic_rotation import (
    build_rotation_weights,
)


def load_data():
    data_dir = REPO / "data" / "high_freq_macro"
    macro = pd.read_parquet(data_dir / "v7_6_X_macro_weekly.parquet")
    etf = pd.read_parquet(data_dir / "v7_10_Y_weekly.parquet")
    return macro, etf


def get_unified_window(strategies):
    first_valid = {}
    for name, w in strategies.items():
        active = w[w.sum(axis=1) > 0.05]
        first_valid[name] = active.index.min() if len(active) > 0 else w.index.max()
    return max(first_valid.values())


def run_backtest_w(w, returns_df, cost_bps=5.0):
    common_cols = [c for c in returns_df.columns if c in w.columns]
    w_aligned = w[common_cols].fillna(0)
    r_aligned = returns_df[common_cols].reindex(w_aligned.index).fillna(0)
    nav, ret, _ = run_backtest(w_aligned, r_aligned, cost_bps=cost_bps)
    return nav, ret


def main():
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v9"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("v9 中信 4 策略 + 原版 5 策略 综合对比")
    print("=" * 70)

    macro, etf = load_data()
    etf_clean = etf.fillna(0).replace([np.inf, -np.inf], 0)
    etf_count = (etf_clean != 0).sum()
    etf_clean = etf_clean.loc[:, etf_count > 100]
    print(f"\n数据: {etf_clean.shape[0]} 周, {etf_clean.shape[1]} ETF")

    print(f"\n[Step 1] 构造原版 5 策略 (与 v9_factor_galaxy_etf.py 一致)")
    weights_g, factor_score, betas, used_macro = run_factor_allocator(
        returns_df=etf_clean,
        macro_df=macro,
        lookback_score=104,
        lookback_beta=52,
        floor=0.01,
        cap=0.15,
    )

    score_z = (factor_score - factor_score.rolling(52).mean()) / (factor_score.rolling(52).std() + 1e-10)
    active_level = (0.7 - 0.5 * score_z).clip(0.2, 1.0)

    galaxy_w = weights_g.copy()
    dynamic_w = galaxy_w.copy()
    dynamic_w[dynamic_w.columns] = 0
    for date in dynamic_w.index:
        if date in active_level.index:
            level = active_level.loc[date]
            if not np.isnan(level):
                dynamic_w.loc[date] = galaxy_w.loc[date] * level

    eq_w = pd.DataFrame(1.0 / len(etf_clean.columns),
                        index=etf_clean.index, columns=etf_clean.columns)

    bond_etfs = [c for c in etf_clean.columns if c in ['511260', '511010', '511090', '159937', '159816']]
    if not bond_etfs:
        bond_etfs = [c for c in etf_clean.columns if '511' in c or '国债' in c][:3]
    stock_etfs = [c for c in etf_clean.columns if c not in bond_etfs]
    w_6040 = pd.DataFrame(0.0, index=etf_clean.index, columns=etf_clean.columns)
    if stock_etfs:
        w_6040[stock_etfs] = 0.6 / len(stock_etfs)
    if bond_etfs:
        w_6040[bond_etfs] = 0.4 / len(bond_etfs)
    elif stock_etfs:
        w_6040[stock_etfs[:5]] = w_6040[stock_etfs[:5]] + 0.4 / 5

    ret_clean = etf_clean.replace([np.inf, -np.inf], 0).fillna(0)
    rp_w = compute_risk_parity_base(ret_clean, lookback=52)
    rp_w = rp_w.reindex(etf_clean.index, method='ffill').fillna(0)

    print(f"\n[Step 2] 构造中信 4 策略")
    aw_w, _ = run_all_weather(etf_clean, macro)
    ma_w, _ = run_macro_allocation(etf_clean, macro)
    br_w, _ = build_multifactor_weights(etf_clean, top_k=10)
    rt_w, _ = build_rotation_weights(etf_clean, top_k=5)

    print(f"\n[Step 3] 找共同起点")
    strategies_all = {
        '等权基准': eq_w,
        '60/40股债': w_6040,
        '基础风险平价': rp_w,
        '银河因子配置': galaxy_w,
        '银河方案-动态仓位': dynamic_w,
        '中信里昂全天候': aw_w,
        '中信大类资产配置': ma_w,
        '中信多因子选股': br_w,
        '中信行业轮动': rt_w,
    }
    common_start = get_unified_window(strategies_all)
    common_idx = etf_clean.index[etf_clean.index >= common_start]
    years = len(common_idx) / 52
    print(f"  共同起点: {common_start.date()}")
    print(f"  统一窗口: {common_idx[0].date()} ~ {common_idx[-1].date()}, {len(common_idx)} 周 ({years:.2f} 年)")

    print(f"\n[Step 4] 跑 9 个回测 (含 5bp 成本)")
    results = []
    navs = {}
    for name, w in strategies_all.items():
        try:
            w_aligned = w[w.index >= common_start]
            nav, ret = run_backtest_w(w_aligned, etf_clean)
            navs[name] = nav
            metrics = compute_factor_metrics(w_aligned, etf_clean, freq='W')
            metrics['strategy'] = name
            metrics['group'] = '原版' if name in ['等权基准', '60/40股债', '基础风险平价', '银河因子配置', '银河方案-动态仓位'] else '中信'
            metrics['start_date'] = str(w_aligned.index.min().date())
            metrics['end_date'] = str(w_aligned.index.max().date())
            results.append(metrics)
        except Exception as e:
            print(f"  {name} 失败: {e}")

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Sharpe', ascending=False).reset_index(drop=True)

    print(f"\n[统一时间窗口] {len(common_idx)} 周 ({years:.2f} 年) 指标对比 (按 Sharpe 排序):")
    display_cols = ['strategy', 'group', 'Sharpe', 'Calmar', 'MaxDD', 'AnnRet', 'Vol', 'WinRate']
    print(df_results[display_cols].to_string(index=False))

    print(f"\n[Step 5] 输出报告")
    df_results.to_csv(output_dir / "citic_all_results.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={'height_ratios': [2, 1]})

    colors = {
        '等权基准': '#94a3b8',
        '60/40股债': '#a3a3a3',
        '基础风险平价': '#fbbf24',
        '银河因子配置': '#3b82f6',
        '银河方案-动态仓位': '#1e40af',
        '中信里昂全天候': '#ef4444',
        '中信大类资产配置': '#f97316',
        '中信多因子选股': '#10b981',
        '中信行业轮动': '#a855f7',
    }
    lws = {
        '银河方案-动态仓位': 2.5,
        '中信里昂全天候': 2.0,
        '中信大类资产配置': 2.0,
        '中信多因子选股': 2.0,
        '中信行业轮动': 2.0,
    }
    lss = {
        '等权基准': '--',
        '60/40股债': '--',
        '基础风险平价': '--',
        '银河因子配置': '--',
    }

    ax0 = axes[0]
    for name, nav in navs.items():
        ax0.plot(nav.index, nav.values,
                 label=name,
                 color=colors.get(name, '#666'),
                 linewidth=lws.get(name, 1.0),
                 linestyle=lss.get(name, '-'),
                 alpha=0.85)

    ax0.set_title(f'v9 9 策略 NAV 对比 ({years:.1f}年, 43 ETF)', fontsize=14, fontweight='bold')
    ax0.set_ylabel('NAV')
    ax0.legend(loc='upper left', fontsize=9, ncol=2)
    ax0.grid(True, alpha=0.3)

    ax1 = axes[1]
    sr = df_results.set_index('strategy')['Sharpe']
    colors_bar = ['#1e40af' if '银河' in s else ('#ef4444' if '中信' in s else '#94a3b8') for s in sr.index]
    ax1.barh(sr.index, sr.values, color=colors_bar)
    ax1.set_xlabel('Sharpe')
    ax1.set_title('Sharpe 排序', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    fig.savefig(output_dir / "citic_all_backtest.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {output_dir / 'citic_all_backtest.png'}")

    report_lines = [
        "# v9 中信 4 策略 vs 原版 5 策略 综合对比",
        "",
        f"> 数据窗口: {common_idx[0].date()} ~ {common_idx[-1].date()} ({len(common_idx)} 周, {years:.2f} 年)",
        f"> 资产: {len(etf_clean.columns)} ETF",
        "",
        "## 一、9 策略 NAV 对比 (含 5bp 成本)",
        "",
        "| 排序 | 策略 | 组别 | Sharpe | Calmar | MaxDD | AnnRet | Vol | WinRate |",
        "|------|------|------|--------|--------|-------|--------|-----|---------|",
    ]
    for i, row in df_results.iterrows():
        report_lines.append(
            f"| {i+1} | {row['strategy']} | {row['group']} | {row['Sharpe']:.3f} | {row['Calmar']:.3f} | "
            f"{row['MaxDD']:.2%} | {row['AnnRet']:.2%} | {row['Vol']:.4f} | {row['WinRate']:.2%} |"
        )

    report_lines.extend([
        "",
        "## 二、策略分组小结",
        "",
    ])

    for grp in ['原版', '中信']:
        sub = df_results[df_results['group'] == grp]
        if len(sub) == 0:
            continue
        report_lines.append(f"### {grp}组 (共 {len(sub)} 个策略)")
        report_lines.append("")
        report_lines.append("| 策略 | Sharpe | AnnRet | MaxDD |")
        report_lines.append("|------|--------|--------|-------|")
        for _, row in sub.iterrows():
            report_lines.append(
                f"| {row['strategy']} | {row['Sharpe']:.3f} | {row['AnnRet']:.2%} | {row['MaxDD']:.2%} |"
            )
        report_lines.append("")

    citic_best = df_results[df_results['group'] == '中信'].iloc[0] if len(df_results[df_results['group'] == '中信']) > 0 else None
    original_best = df_results[df_results['group'] == '原版'].iloc[0] if len(df_results[df_results['group'] == '原版']) > 0 else None

    report_lines.extend([
        "## 三、对比结论",
        "",
    ])
    if citic_best is not None and original_best is not None:
        report_lines.append(f"- **中信最优**: {citic_best['strategy']} (Sharpe {citic_best['Sharpe']:.3f}, 年化 {citic_best['AnnRet']:.2%})")
        report_lines.append(f"- **原版最优**: {original_best['strategy']} (Sharpe {original_best['Sharpe']:.3f}, 年化 {original_best['AnnRet']:.2%})")
        if citic_best['Sharpe'] > original_best['Sharpe']:
            report_lines.append(f"- **中信胜出**: Sharpe 高 {(citic_best['Sharpe'] - original_best['Sharpe']):.3f}")
        else:
            report_lines.append(f"- **原版胜出**: Sharpe 高 {(original_best['Sharpe'] - citic_best['Sharpe']):.3f}")

    report_lines.extend([
        "",
        "## 四、中信 4 策略实现要点",
        "",
        "### 4.1 中信里昂全天候 (All-Weather)",
        "- 风险平价基础 × 增长/通胀象限系数",
        "- 4 类资产: broad (宽基) / sector (行业) / overseas (海外) / gold (黄金)",
        "- 4 象限: ↑G↓I (加股) / ↑G↑I (平衡) / ↓G↑I (防御) / ↓G↓I (防御+长债)",
        "",
        "### 4.2 中信大类资产配置 (Macro 5 因子)",
        "- 5 宏观因子: 增长 (正) / 通胀 (中性, 排除) / 信贷 (反) / 汇率 (反) / 利率 (正)",
        "- 滚动 z-score → 综合得分 → 风险资产占比 30%-80%",
        "- 黄金作为防御资产 (反相关)",
        "",
        "### 4.3 中信多因子选股",
        "- 5 风格因子: Momentum / Volatility (反) / Quality / Size (反) / Value (反转)",
        "- 横截面 z-score 等权合成",
        "- 每周选 Top-10 候选 (softmax 加权) + 33 个等权底仓",
        "",
        "### 4.4 中信行业轮动",
        "- 在 23 个行业 ETF 内做轮动",
        "- 综合得分 = z(动量) - z(波动率)",
        "- Top-5 行业 5x 权重, 其他 0.5x, 非行业等权",
        "",
        "## 五、文件清单",
        "",
        "- `QuantNodes/strategy/momentum_etf_rotation/v9/citic_all_weather.py`",
        "- `QuantNodes/strategy/momentum_etf_rotation/v9/citic_macro.py`",
        "- `QuantNodes/strategy/momentum_etf_rotation/v9/citic_multifactor.py`",
        "- `QuantNodes/strategy/momentum_etf_rotation/v9/citic_rotation.py`",
        "- `scripts/v9/v9_citic_all.py` (本脚本)",
        "",
        "## 六、产出",
        "",
        "- `reports/momentum_etf_rotation/v9/citic_all_results.csv`",
        "- `reports/momentum_etf_rotation/v9/citic_all_backtest.png`",
        "- `reports/momentum_etf_rotation/v9/citic_all_report.md`",
    ])

    report_path = output_dir / "citic_all_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  {report_path}")

    print(f"\n{'='*70}")
    print("完成!")


if __name__ == "__main__":
    main()
