# coding=utf-8
"""v9 宏观周期回测 — 用 GDP/CPI 趋势做择时.

逻辑:
    GDP 趋势 ↑ AND CPI 趋势 ↓ → Recovery → 满仓
    GDP 趋势 ↓ OR CPI 趋势 ↑ → 其他阶段 → 减仓/空仓

用法:
    python3.11 scripts/v9/v9_macro_backtest.py
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

from statsmodels.tsa.filters.hp_filter import hpfilter

from QuantNodes.strategy.momentum_etf_rotation.v9.backtest import (
    run_backtest,
    compute_metrics,
    cost_sensitivity,
)


def load_data():
    data_dir = REPO / "data" / "high_freq_macro"
    macro = pd.read_parquet(data_dir / "v7_6_X_macro_weekly.parquet")
    y_weekly = pd.read_parquet(data_dir / "v7_10_Y_weekly.parquet")
    return macro, y_weekly


def get_trend_direction(series, window=52):
    """判断趋势方向: 上行/下行/平."""
    if len(series.dropna()) < window:
        return 'flat'
    recent = series.dropna().iloc[-window:]
    slope = np.polyfit(np.arange(len(recent)), recent.values, 1)[0]
    std = recent.std()
    threshold = std / len(recent) * 0.02 if std > 0 else 1e-10
    if slope > threshold:
        return 'up'
    elif slope < -threshold:
        return 'down'
    return 'flat'


def generate_macro_signals(macro, y_weekly):
    """基于宏观周期生成择时信号."""
    print("=" * 70)
    print("v9 宏观周期回测")
    print("=" * 70)

    growth = macro['宏观增长因子'].dropna()
    cpi = macro['宏观通胀因子_生活端'].dropna()
    ppi = macro['宏观通胀因子_生产端'].dropna()

    common_idx = growth.index.intersection(cpi.index).intersection(ppi.index)
    growth = growth.loc[common_idx]
    cpi = cpi.loc[common_idx]
    ppi = ppi.loc[common_idx]

    # Z-score 标准化
    growth_z = (growth - growth.mean()) / (growth.std() + 1e-10)
    cpi_z = (cpi - cpi.mean()) / (cpi.std() + 1e-10)
    ppi_z = (ppi - ppi.mean()) / (ppi.std() + 1e-10)

    # HP 滤波提取趋势
    g_cycle, g_trend = hpfilter(growth_z, lamb=100)
    c_cycle, c_trend = hpfilter(cpi_z, lamb=100)
    p_cycle, p_trend = hpfilter(ppi_z, lamb=100)

    print(f"\n[Step 1] 宏观趋势分析")
    print(f"  经济增长趋势: {get_trend_direction(g_trend)}")
    print(f"  CPI通胀趋势:  {get_trend_direction(c_trend)}")
    print(f"  PPI通胀趋势:  {get_trend_direction(p_trend)}")

    # 生成信号 (基于滚动窗口)
    signals = pd.Series(0, index=common_idx, dtype=float)
    phases = pd.Series('Recession', index=common_idx, dtype=object)

    window = 52  # 1年窗口判断趋势
    for t in range(window, len(common_idx)):
        g_dir = get_trend_direction(g_trend.iloc[:t+1], window)
        c_dir = get_trend_direction(c_trend.iloc[:t+1], window)

        if g_dir == 'up' and c_dir == 'down':
            phases.iloc[t] = 'Recovery'
            signals.iloc[t] = 1.0
        elif g_dir == 'up' and c_dir == 'up':
            phases.iloc[t] = 'Overheat'
            signals.iloc[t] = 0.5
        elif g_dir == 'down' and c_dir == 'up':
            phases.iloc[t] = 'Stagflation'
            signals.iloc[t] = 0.0
        else:
            phases.iloc[t] = 'Recession'
            signals.iloc[t] = 0.2

    print(f"\n[Step 2] 信号分布")
    print(f"  Recovery (满仓): {(signals == 1.0).sum()} 周 ({(signals == 1.0).mean():.1%})")
    print(f"  Overheat (半仓): {(signals == 0.5).sum()} 周 ({(signals == 0.5).mean():.1%})")
    print(f"  Stagflation (空仓): {(signals == 0.0).sum()} 周 ({(signals == 0.0).mean():.1%})")
    print(f"  Recession (20%仓): {(signals == 0.2).sum()} 周 ({(signals == 0.2).mean():.1%})")

    print(f"\n[Step 3] 最近状态")
    for col_name in ['宏观增长因子', '宏观通胀因子_生活端', '宏观通胀因子_生产端']:
        if col_name in macro.columns:
            s = macro[col_name].dropna()
            z = (s - s.mean()) / (s.std() + 1e-10)
            _, trend = hpfilter(z, lamb=100)
            print(f"  {col_name}: Z={z.iloc[-1]:+.2f}, 趋势={get_trend_direction(trend)}")

    print(f"  最新阶段: {phases.iloc[-1]}")
    print(f"  最新信号: {signals.iloc[-1]}")

    return signals, phases, g_trend, c_trend, p_trend, common_idx


def run_backtests(signals, y_weekly, common_idx):
    """运行回测对比."""
    print(f"\n[Step 4] 回测")

    # 对齐到 ETF 数据
    y_aligned = y_weekly.reindex(common_idx, method='ffill').fillna(0)
    n_etfs = y_aligned.shape[1]

    # 策略 1: ETF 等权 (基准)
    equal_w = pd.DataFrame(1.0/n_etfs, index=common_idx, columns=y_aligned.columns)
    nav_eq, _, met_eq = run_backtest(equal_w, y_aligned, cost_bps=10.0)
    met_eq['strategy'] = 'ETF等权(基准)'

    # 策略 2: 宏观周期择时 (v9)
    macro_w = equal_w.multiply(signals, axis=0)
    nav_macro, _, met_macro = run_backtest(macro_w, y_aligned, cost_bps=10.0)
    met_macro['strategy'] = 'v9宏观周期'

    # 策略 3: 宏观周期择时 (严格版: 只有 Recovery 满仓, 其他空仓)
    strict_signal = (signals == 1.0).astype(float)
    strict_w = equal_w.multiply(strict_signal, axis=0)
    nav_strict, _, met_strict = run_backtest(strict_w, y_aligned, cost_bps=10.0)
    met_strict['strategy'] = 'v9严格版'

    # 策略 4: 宏观周期 + 成本敏感性
    print(f"\n  宏观周期择时 (10bp):")
    for k, v in met_macro.items():
        print(f"    {k}: {v}")

    print(f"\n  严格版 (10bp):")
    for k, v in met_strict.items():
        print(f"    {k}: {v}")

    # 成本敏感性
    print(f"\n[Step 5] 成本敏感性 (v9宏观周期)")
    cost_df = cost_sensitivity(macro_w, y_aligned)
    print(cost_df[['cost_bps', 'Sharpe', 'Calmar', 'MaxDD', 'AnnRet']].to_string(index=False))

    return pd.DataFrame([met_eq, met_macro, met_strict]), cost_df, nav_eq, nav_macro, nav_strict


def plot_results(signals, phases, g_trend, c_trend, nav_eq, nav_macro, nav_strict, output_dir):
    """绘制回测结果."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [1.5, 1, 1]})

    # 上: NAV 对比
    ax0 = axes[0]
    nav_eq.plot(ax=ax0, label='ETF等权(基准)', color='#94a3b8', linewidth=1.5, linestyle='--')
    nav_macro.plot(ax=ax0, label='v9宏观周期', color='#3b82f6', linewidth=2)
    nav_strict.plot(ax=ax0, label='v9严格版', color='#10b981', linewidth=1.5)
    ax0.set_title('v9 宏观周期择时 vs 基准', fontsize=14, fontweight='bold')
    ax0.set_ylabel('NAV')
    ax0.legend(loc='upper left')
    ax0.grid(True, alpha=0.3)

    # 中: 信号时序
    ax1 = axes[1]
    ax1.fill_between(signals.index, 0, signals.values, color='#3b82f6', alpha=0.3, label='仓位比例')
    ax1.plot(signals.index, signals.values, color='#3b82f6', linewidth=1)
    ax1.set_ylabel('仓位比例')
    ax1.set_title('v9 仓位信号', fontsize=12)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 下: 宏观趋势
    ax2 = axes[2]
    ax2.plot(g_trend.index, g_trend.values, label='经济增长趋势', color='#22c55e', linewidth=1.5)
    ax2.plot(c_trend.index, c_trend.values, label='CPI通胀趋势', color='#ef4444', linewidth=1.5)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('HP趋势')
    ax2.set_title('宏观经济趋势', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "macro_backtest.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  {output_dir / 'macro_backtest.png'}")


def main():
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v9"
    output_dir.mkdir(parents=True, exist_ok=True)

    macro, y_weekly = load_data()
    signals, phases, g_trend, c_trend, p_trend, common_idx = generate_macro_signals(macro, y_weekly)
    comparison, cost_df, nav_eq, nav_macro, nav_strict = run_backtests(signals, y_weekly, common_idx)

    comparison.to_csv(output_dir / "macro_backtest_results.csv", index=False)
    cost_df.to_csv(output_dir / "macro_cost_sensitivity.csv", index=False)

    plot_results(signals, phases, g_trend, c_trend, nav_eq, nav_macro, nav_strict, output_dir)

    print(f"\n[Step 6] 指标对比")
    print(comparison[['strategy', 'Sharpe', 'Calmar', 'MaxDD', 'AnnRet', 'Vol', 'WinRate']].to_string(index=False))

    print(f"\n  报告: {output_dir / 'macro_backtest_results.csv'}")
    print(f"  成本敏感性: {output_dir / 'macro_cost_sensitivity.csv'}")

    print("\n" + "=" * 70)
    print("完成!")


if __name__ == "__main__":
    main()