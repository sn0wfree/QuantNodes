"""scripts/cmf/plot_gold_nav.py — 生成黄金策略净值曲线 (bw=0.2 vs 基线).

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


def load_close():
    import duckdb
    con = duckdb.connect("data/cmf/fundamental.duckdb", read_only=True)
    df = con.execute(
        "SELECT trade_date, close FROM futures_main_close_au ORDER BY trade_date"
    ).df()
    con.close()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return (df.drop_duplicates("trade_date", keep="last")
              .set_index("trade_date")["close"])


def compute_nav(pos, close, start="2010-01-01", end="2026-08-25"):
    sub = pos[(pos.index >= start) & (pos.index <= end)]
    c = close.reindex(sub.index)
    ret = c.pct_change()
    sr = (sub.fillna(0).shift(1) * ret).dropna()
    cost = sub.fillna(0).diff().abs() * 5 / 10000
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
    close = load_close()

    # 策略: bw=0.2 (新默认)
    sig_new = compute_layered_ensemble_signal(breakout_weight=0.2)
    pos_new = sig_new["position"]

    # 基线: bw=0 (原三路等权)
    sig_base = compute_layered_ensemble_signal(breakout_weight=0.0)
    pos_base = sig_base["position"]

    # Buy&Hold
    bh_pos = pd.Series(1.0, index=close.index)

    # 计算净值
    nav_new = compute_nav(pos_new, close)
    nav_base = compute_nav(pos_base, close)
    nav_bh = compute_nav(bh_pos, close)

    # 计算指标
    m_new = get_metrics(nav_new)
    m_base = get_metrics(nav_base)
    m_bh = get_metrics(nav_bh)

    # 输出指标
    print("=" * 90)
    print("黄金策略净值曲线对比 (bw=0.2 vs 基线 vs Buy&Hold)")
    print("=" * 90)
    print(f'{"指标":<20} {"新策略(bw=0.2)":>18} {"基线(bw=0)":>18} {"Buy&Hold":>18}')
    print("-" * 80)
    print(f'{"年化收益":<20} {m_new[0]*100:>17.2f}% {m_base[0]*100:>17.2f}% {m_bh[0]*100:>17.2f}%')
    print(f'{"Sharpe":<20} {m_new[1]:>18.2f} {m_base[1]:>18.2f} {m_bh[1]:>18.2f}')
    print(f'{"MaxDD":<20} {m_new[2]*100:>17.2f}% {m_base[2]*100:>17.2f}% {m_bh[2]*100:>17.2f}%')
    print(f'{"Calmar":<20} {m_new[3]:>18.2f} {m_base[3]:>18.2f} {m_bh[3]:>18.2f}')
    print(f'{"总收益":<20} {(nav_new.iloc[-1]-1)*100:>17.2f}% {(nav_base.iloc[-1]-1)*100:>17.2f}% {(nav_bh.iloc[-1]-1)*100:>17.2f}%')

    # 绘图
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1, 1]})

    # 图1: 净值曲线
    ax1 = axes[0]
    ax1.plot(nav_new.index, nav_new.values, label='新策略 (bw=0.2)', linewidth=2, color='#2196F3')
    ax1.plot(nav_base.index, nav_base.values, label='基线 (bw=0)', linewidth=2, color='#4CAF50', linestyle='--')
    ax1.plot(nav_bh.index, nav_bh.values, label='Buy & Hold', linewidth=2, color='#FF9800', alpha=0.7)
    ax1.set_ylabel('累计净值')
    ax1.set_title('黄金策略净值曲线对比', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # 标注关键点
    ax1.axvline(x=pd.Timestamp('2013-04-15'), color='red', linestyle=':', alpha=0.5, label='2013 taper')
    ax1.axvline(x=pd.Timestamp('2020-03-16'), color='red', linestyle=':', alpha=0.5, label='COVID')
    ax1.axvline(x=pd.Timestamp('2026-01-01'), color='purple', linestyle=':', alpha=0.5, label='2026')

    # 图2: 回撤曲线
    ax2 = axes[1]
    dd_new = nav_new / nav_new.cummax() - 1
    dd_base = nav_base / nav_base.cummax() - 1
    dd_bh = nav_bh / nav_bh.cummax() - 1
    ax2.fill_between(dd_new.index, dd_new.values * 100, 0, alpha=0.3, color='#2196F3', label='新策略')
    ax2.fill_between(dd_base.index, dd_base.values * 100, 0, alpha=0.3, color='#4CAF50', label='基线')
    ax2.fill_between(dd_bh.index, dd_bh.values * 100, 0, alpha=0.2, color='#FF9800', label='B&H')
    ax2.set_ylabel('回撤 (%)')
    ax2.set_title('回撤曲线', fontsize=12)
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.axhline(y=-15, color='red', linestyle='--', alpha=0.5, label='DD -15%')

    # 图3: 仓位
    ax3 = axes[2]
    pos_sub = pos_new[(pos_new.index >= nav_new.index[0]) & (pos_new.index <= nav_new.index[-1])]
    ax3.step(pos_sub.index, pos_sub.values, where='post', linewidth=1.5, color='#9C27B0', label='新策略')
    ax3.set_ylabel('仓位')
    ax3.set_title('策略仓位', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # 格式化x轴
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    # 保存
    output_dir = Path("reports/gold")
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "gold_nav_curve.png"
    csv_path = output_dir / "gold_nav_curve.csv"

    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"\n净值曲线图已保存: {png_path}")

    # 保存CSV
    nav_df = pd.DataFrame({
        'strategy_nav_bw20': nav_new,
        'strategy_nav_bw0': nav_base,
        'bh_nav': nav_bh,
        'position_bw20': pos_new.reindex(nav_new.index).fillna(method='ffill').fillna(0),
        'position_bw0': pos_base.reindex(nav_new.index).fillna(method='ffill').fillna(0),
        'drawdown_bw20': dd_new,
        'drawdown_bw0': dd_base,
    })
    nav_df.to_csv(csv_path)
    print(f"净值数据已保存: {csv_path}")

    plt.show()


if __name__ == "__main__":
    main()
