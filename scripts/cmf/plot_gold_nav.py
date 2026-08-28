"""scripts/cmf/plot_gold_nav.py — 黄金策略净值曲线对比 (Open-to-Open回测方式).

信号在收盘计算, 次日(或夜盘)开盘价成交.
收益计算: pos[t] * (open[t+1]-open[t])/open[t]

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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, '/home/ll/Public/QuantNodes')
sys.path.insert(0, '/home/ll/Public/QuantNodes/scripts/cmf')

from option_utils import calc_metrics
from QuantNodes.strategy.commodity_fundamental.gold import compute_layered_ensemble_signal


def compute_nav_open(pos, open_price, close, start="2010-01-01", end="2026-08-25"):
    """Open-to-Open 净值计算: 信号在收盘计算, 次日开盘价成交."""
    sub_pos = pos[(pos.index >= start) & (pos.index <= end)]
    o = open_price.reindex(sub_pos.index)
    c = close.reindex(sub_pos.index)
    # open-to-open收益: pos[t] * (open[t+1]-open[t])/open[t]
    ret_o2o = o.pct_change().shift(-1)  # 次日open收益
    sr = (sub_pos * ret_o2o).dropna()
    cost = sub_pos.diff().abs() * 5 / 10000
    sr = (sr - cost).dropna()
    nav = (1 + sr).cumprod()
    return nav


def get_metrics(nav):
    sr = nav.pct_change().dropna()
    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (252 / len(nav)) - 1
    vol = sr.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    sharpe = ann_ret / vol if vol > 0 else 0
    calmar = ann_ret / abs(dd) if dd != 0 else 0
    return ann_ret, sharpe, dd, calmar


def main():
    sig = compute_layered_ensemble_signal()
    close = sig["close"]
    open_price = sig["open"]
    pos = sig["position"]

    # CPPI版本 (L=1.5)
    from QuantNodes.strategy.commodity_fundamental.gold import compute_risk_overlay
    from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import _load_indicator_d
    vix = _load_indicator_d("vix", "gold_fred").reindex(close.index).ffill()
    gvz = _load_indicator_d("gvz", "gold_fred").reindex(close.index).ffill()

    pos_15_raw = compute_risk_overlay(
        close=close, position=(pos * 1.5).clip(0, 1.8), vix=vix, gvz=gvz,
        hard_cap=-0.15, soft_cap=-0.12, daily_cap=-0.03, vol_cap=0.20, cooldown_days=5)
    pos_15 = pos_15_raw.copy()
    for i in range(1, len(pos_15)):
        if abs(pos_15.iloc[i] - pos_15.iloc[i-1]) < 0.05:
            pos_15.iloc[i] = pos_15.iloc[i-1]

    pos_20_raw = compute_risk_overlay(
        close=close, position=(pos * 2.0).clip(0, 2.4), vix=vix, gvz=gvz,
        hard_cap=-0.15, soft_cap=-0.12, daily_cap=-0.03, vol_cap=0.20, cooldown_days=5)
    pos_20 = pos_20_raw.copy()
    for i in range(1, len(pos_20)):
        if abs(pos_20.iloc[i] - pos_20.iloc[i-1]) < 0.05:
            pos_20.iloc[i] = pos_20.iloc[i-1]

    bh_pos = pd.Series(1.0, index=close.index)

    # 净值
    nav_base = compute_nav_open(pos, open_price, close)
    nav_15 = compute_nav_open(pos_15, open_price, close)
    nav_20 = compute_nav_open(pos_20, open_price, close)
    nav_bh = compute_nav_open(bh_pos, open_price, close)

    # 指标
    m_base = get_metrics(nav_base)
    m_15 = get_metrics(nav_15)
    m_20 = get_metrics(nav_20)
    m_bh = get_metrics(nav_bh)

    print("=" * 100)
    print("黄金策略净值曲线 (Open-to-Open回测)")
    print("=" * 100)
    print(f'{"指标":<15} {"L=1.5 CPPI":>15} {"L=2.0 CPPI":>15} {"基线(L=1)":>15} {"Buy&Hold":>15}')
    print("-" * 80)
    print(f'{"年化收益":<15} {m_15[0]*100:>14.2f}% {m_20[0]*100:>14.2f}% {m_base[0]*100:>14.2f}% {m_bh[0]*100:>14.2f}%')
    print(f'{"Sharpe":<15} {m_15[1]:>15.2f} {m_20[1]:>15.2f} {m_base[1]:>15.2f} {m_bh[1]:>15.2f}')
    print(f'{"MaxDD":<15} {m_15[2]*100:>14.2f}% {m_20[2]*100:>14.2f}% {m_base[2]*100:>14.2f}% {m_bh[2]*100:>14.2f}%')
    print(f'{"Calmar":<15} {m_15[3]:>15.2f} {m_20[3]:>15.2f} {m_base[3]:>15.2f} {m_bh[3]:>15.2f}')
    total = [(nav_15.iloc[-1]-1)*100, (nav_20.iloc[-1]-1)*100, (nav_base.iloc[-1]-1)*100, (nav_bh.iloc[-1]-1)*100]
    print(f'{"总收益":<15} {total[0]:>14.2f}% {total[1]:>14.2f}% {total[2]:>14.2f}% {total[3]:>14.2f}%')

    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

    ax1 = axes[0]
    ax1.plot(nav_15.index, nav_15.values, label='L=1.5 CPPI', linewidth=2.5, color='#2196F3')
    ax1.plot(nav_20.index, nav_20.values, label='L=2.0 CPPI', linewidth=2, color='#9C27B0', linestyle='--')
    ax1.plot(nav_base.index, nav_base.values, label='基线(L=1)', linewidth=2, color='#4CAF50', linestyle='-.')
    ax1.plot(nav_bh.index, nav_bh.values, label='Buy&Hold', linewidth=2, color='#FF9800', alpha=0.7)
    ax1.set_ylabel('累计净值')
    ax1.set_title('黄金策略净值曲线 (Open-to-Open回测)', fontsize=14, fontweight='bold')
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
    ax3.step(pos_20.index, pos_20.values, where='post', linewidth=1, color='#9C27B0', label='L=2.0 CPPI', alpha=0.7)
    ax3.set_ylabel('仓位')
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
        'nav_L1.5_CPPI': nav_15,
        'nav_L2.0_CPPI': nav_20,
        'nav_baseline_L1': nav_base,
        'nav_BuyHold': nav_bh,
        'position_L1.5': pos_15.reindex(nav_15.index).fillna(method='ffill').fillna(0),
        'position_L2.0': pos_20.reindex(nav_20.index).fillna(method='ffill').fillna(0),
        'position_baseline': pos.reindex(nav_base.index).fillna(method='ffill').fillna(0),
    })
    nav_df.to_csv(csv_path)
    print(f"净值数据已保存: {csv_path}")

    plt.show()


if __name__ == "__main__":
    main()
