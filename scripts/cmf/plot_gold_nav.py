"""scripts/cmf/plot_gold_nav.py — 黄金策略净值曲线对比 (真实shift2 + Open-to-Open).

数据流: 信号收盘计算(因子用T-2数据) → 次日开盘价成交
收益: pos[t] * (open[t+1]-open[t])/open[t]

输出:
- reports/gold/gold_nav_curve.png
- reports/gold/gold_nav_curve.csv
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, '/home/ll/Public/QuantNodes')
sys.path.insert(0, '/home/ll/Public/QuantNodes/scripts/cmf')

from QuantNodes.strategy.commodity_fundamental.gold import (
    compute_layered_ensemble_signal, compute_risk_overlay,
)


def compute_nav_o2o(pos, open_price, start="2010-01-01", end="2026-08-27"):
    """Open-to-Open净值: 信号收盘算, 次日开盘成交."""
    sub = pos[(pos.index >= start) & (pos.index <= end)]
    o = open_price.reindex(sub.index)
    ret_o2o = o.pct_change().shift(-1)
    sr = (sub * ret_o2o).dropna()
    cost = sub.diff().abs() * 5 / 10000
    sr = (sr - cost).dropna()
    return (1 + sr).cumprod()


def get_metrics(nav):
    sr = nav.pct_change().dropna()
    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (252 / len(nav)) - 1
    vol = sr.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    sharpe = ann_ret / vol if vol > 0 else 0
    calmar = ann_ret / abs(dd) if dd != 0 else 0
    return ann_ret, sharpe, dd, calmar


def main():
    # 生产函数 (shift2统一入口)
    sig = compute_layered_ensemble_signal()
    close = sig["close"]
    open_price = sig["open"]
    pos_base = sig["position"]

    # CPPI L=1.5 (shift2因子已在信号内, 此处只做杠杆+死带)
    pos_15 = (pos_base * 1.5).clip(0, 1.8)
    for i in range(1, len(pos_15)):
        if abs(pos_15.iloc[i] - pos_15.iloc[i-1]) < 0.05:
            pos_15.iloc[i] = pos_15.iloc[i-1]

    pos_20 = (pos_base * 2.0).clip(0, 2.4)
    for i in range(1, len(pos_20)):
        if abs(pos_20.iloc[i] - pos_20.iloc[i-1]) < 0.05:
            pos_20.iloc[i] = pos_20.iloc[i-1]

    bh_pos = pd.Series(1.0, index=close.index)

    # 净值
    nav_base = compute_nav_o2o(pos_base, open_price)
    nav_15 = compute_nav_o2o(pos_15, open_price)
    nav_20 = compute_nav_o2o(pos_20, open_price)
    nav_bh = compute_nav_o2o(bh_pos, open_price)

    # 指标
    m_base = get_metrics(nav_base)
    m_15 = get_metrics(nav_15)
    m_20 = get_metrics(nav_20)
    m_bh = get_metrics(nav_bh)

    print("=" * 100)
    print("黄金策略净值对比 (真实shift2 + Open-to-Open, 5bps)")
    print("=" * 100)
    print(f'{"指标":<12} {"L=1.5 CPPI":>14} {"L=2.0 CPPI":>14} {"基线(L=1)":>14} {"B&H":>14}')
    print("-" * 75)
    print(f'{"年化收益":<12} {m_15[0]*100:>13.2f}% {m_20[0]*100:>13.2f}% {m_base[0]*100:>13.2f}% {m_bh[0]*100:>13.2f}%')
    print(f'{"Sharpe":<12} {m_15[1]:>14.2f} {m_20[1]:>14.2f} {m_base[1]:>14.2f} {m_bh[1]:>14.2f}')
    print(f'{"MaxDD":<12} {m_15[2]*100:>13.2f}% {m_20[2]*100:>13.2f}% {m_base[2]*100:>13.2f}% {m_bh[2]*100:>13.2f}%')
    print(f'{"Calmar":<12} {m_15[3]:>14.2f} {m_20[3]:>14.2f} {m_base[3]:>14.2f} {m_bh[3]:>14.2f}')
    total = [(nav_15.iloc[-1]-1)*100, (nav_20.iloc[-1]-1)*100, (nav_base.iloc[-1]-1)*100, (nav_bh.iloc[-1]-1)*100]
    print(f'{"总收益":<12} {total[0]:>13.2f}% {total[1]:>13.2f}% {total[2]:>13.2f}% {total[3]:>13.2f}%')

    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

    ax1 = axes[0]
    ax1.plot(nav_15.index, nav_15.values, label='L=1.5 CPPI (shift2)', linewidth=2.5, color='#2196F3')
    ax1.plot(nav_20.index, nav_20.values, label='L=2.0 CPPI (shift2)', linewidth=2, color='#9C27B0', linestyle='--')
    ax1.plot(nav_base.index, nav_base.values, label='基线 L=1 (shift2)', linewidth=2, color='#4CAF50', linestyle='-.')
    ax1.plot(nav_bh.index, nav_bh.values, label='Buy&Hold', linewidth=2, color='#FF9800', alpha=0.7)
    ax1.set_ylabel('累计净值')
    ax1.set_title('黄金策略净值曲线 (真实shift2 + Open-to-Open, 5bps成本)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    ax2 = axes[1]
    dd_15 = nav_15 / nav_15.cummax() - 1
    dd_20 = nav_20 / nav_20.cummax() - 1
    dd_bh = nav_bh / nav_bh.cummax() - 1
    ax2.fill_between(dd_15.index, dd_15.values*100, 0, alpha=0.4, color='#2196F3', label='L=1.5 CPPI')
    ax2.fill_between(dd_20.index, dd_20.values*100, 0, alpha=0.3, color='#9C27B0', label='L=2.0 CPPI')
    ax2.fill_between(dd_bh.index, dd_bh.values*100, 0, alpha=0.2, color='#FF9800', label='B&H')
    ax2.set_ylabel('回撤 (%)')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.axhline(y=-15, color='red', linestyle='--', alpha=0.5)

    ax3 = axes[2]
    ax3.step(pos_15.index, pos_15.values, where='post', linewidth=1.5, color='#2196F3', label='L=1.5 CPPI')
    ax3.step(pos_base.index, pos_base.values, where='post', linewidth=1, color='#4CAF50', label='基线 L=1', alpha=0.7)
    ax3.set_ylabel('仓位')
    ax3.set_xlabel('日期')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=1.5, color='red', linestyle=':', alpha=0.5)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    output_dir = Path("reports/gold")
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "gold_nav_curve.png"
    csv_path = output_dir / "gold_nav_curve.csv"

    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"\n净值曲线图已保存: {png_path}")

    nav_df = pd.DataFrame({
        'nav_L1.5_CPPI_shift2': nav_15,
        'nav_L2.0_CPPI_shift2': nav_20,
        'nav_baseline_L1_shift2': nav_base,
        'nav_BuyHold': nav_bh,
        'position_L1.5': pos_15.reindex(nav_15.index).fillna(method='ffill').fillna(0),
        'position_L2.0': pos_20.reindex(nav_20.index).fillna(method='ffill').fillna(0),
        'position_baseline': pos_base.reindex(nav_base.index).fillna(method='ffill').fillna(0),
    })
    nav_df.to_csv(csv_path)
    print(f"净值数据已保存: {csv_path}")


if __name__ == "__main__":
    main()
