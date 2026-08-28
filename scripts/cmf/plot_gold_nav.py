"""scripts/cmf/plot_gold_nav.py — 黄金策略净值曲线对比.

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
from QuantNodes.strategy.commodity_fundamental.gold import compute_risk_overlay
from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    _load_indicator_d, compute_macro_regime_signal,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_real_rate import compute_real_rate_signal
from QuantNodes.strategy.commodity_fundamental.gold.gold_momentum import compute_momentum_signal
from QuantNodes.strategy.commodity_fundamental.gold.gold_breakout import compute_breakout_signal


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
    close = load_close()
    tips = _load_indicator_d("tips10y", "fred_dfii10").reindex(close.index).ffill()
    vix = _load_indicator_d("vix", "gold_fred").reindex(close.index).ffill()
    gvz = _load_indicator_d("gvz", "gold_fred").reindex(close.index).ffill()

    # ===== Δtips信号 =====
    d = tips.diff(10)
    rank = d.rolling(504, min_periods=126).rank(pct=True)
    delta_bin = (rank <= 0.5).astype(float).fillna(0)

    # ===== 四路信号+风控 =====
    macro_raw = compute_macro_regime_signal()["position"].reindex(close.index).fillna(method="ffill").fillna(0)
    rr_raw = compute_real_rate_signal(variant="tips_ma")["position"].reindex(close.index).fillna(method="ffill").fillna(0)
    mom_raw = compute_momentum_signal(variant="long_only")["position"].reindex(close.index).fillna(method="ffill").fillna(0)
    bo_raw = compute_breakout_signal()["position"].reindex(close.index).fillna(0)

    def ov(p):
        return compute_risk_overlay(close=close, position=p, vix=vix, gvz=gvz,
                                    hard_cap=-0.10, soft_cap=-0.07, daily_cap=-0.02, vol_cap=0.15, cooldown_days=5)

    macro_ov = ov(macro_raw)
    rr_ov = ov(rr_raw)
    mom_ov = ov(mom_raw)
    bo_ov = ov(bo_raw)

    # ===== 策略构建 =====
    # 1. 基线Δtips (L=1.0, 死带0.05)
    w4 = 0.85 / 4
    pos_base = (w4*macro_ov + w4*rr_ov + w4*mom_ov + w4*bo_ov + 0.15*delta_bin).clip(0, 1.2)
    # 死带平滑
    for i in range(1, len(pos_base)):
        if abs(pos_base.iloc[i] - pos_base.iloc[i-1]) < 0.05:
            pos_base.iloc[i] = pos_base.iloc[i-1]

    # 2. L=1.5 CPPI + 死带
    pos_15_raw = compute_risk_overlay(
        close=close, position=(pos_base * 1.5).clip(0, 1.8), vix=vix, gvz=gvz,
        hard_cap=-0.15, soft_cap=-0.12, daily_cap=-0.03, vol_cap=0.20, cooldown_days=5)
    # 死带平滑CPPI仓位
    pos_15 = pos_15_raw.copy()
    for i in range(1, len(pos_15)):
        if abs(pos_15.iloc[i] - pos_15.iloc[i-1]) < 0.05:
            pos_15.iloc[i] = pos_15.iloc[i-1]

    # 3. L=2.0 CPPI + 死带
    pos_20_raw = compute_risk_overlay(
        close=close, position=(pos_base * 2.0).clip(0, 2.4), vix=vix, gvz=gvz,
        hard_cap=-0.15, soft_cap=-0.12, daily_cap=-0.03, vol_cap=0.20, cooldown_days=5)
    pos_20 = pos_20_raw.copy()
    for i in range(1, len(pos_20)):
        if abs(pos_20.iloc[i] - pos_20.iloc[i-1]) < 0.05:
            pos_20.iloc[i] = pos_20.iloc[i-1]

    # 4. L=1.5 CPPI 无死带 (对照)
    pos_15_nodead = compute_risk_overlay(
        close=close, position=(pos_base * 1.5).clip(0, 1.8), vix=vix, gvz=gvz,
        hard_cap=-0.15, soft_cap=-0.12, daily_cap=-0.03, vol_cap=0.20, cooldown_days=5)

    # 5. Buy&Hold
    bh_pos = pd.Series(1.0, index=close.index)

    # ===== 计算净值 =====
    nav_base = compute_nav(pos_base, close)
    nav_15 = compute_nav(pos_15, close)
    nav_20 = compute_nav(pos_20, close)
    nav_15_nodead = compute_nav(pos_15_nodead, close)
    nav_bh = compute_nav(bh_pos, close)

    # ===== 指标输出 =====
    m_base = get_metrics(nav_base)
    m_15 = get_metrics(nav_15)
    m_20 = get_metrics(nav_20)
    m_15_nodead = get_metrics(nav_15_nodead)
    m_bh = get_metrics(nav_bh)

    print("=" * 100)
    print("黄金策略净值曲线对比 (死带平滑 vs 无死带)")
    print("=" * 100)
    print(f'{"指标":<15} {"L=1.5死带":>15} {"L=1.5无死带":>15} {"L=2.0死带":>15} {"基线":>15} {"B&H":>15}')
    print("-" * 95)
    for name, idx, fmt in [("年化收益", 0, True), ("Sharpe", 1, False), ("MaxDD", 2, True), ("Calmar", 3, False)]:
        vals = [m_15[idx], m_15_nodead[idx], m_20[idx], m_base[idx], m_bh[idx]]
        if fmt:
            print(f'{name:<15} {vals[0]*100:>14.2f}% {vals[1]*100:>14.2f}% {vals[2]*100:>14.2f}% {vals[3]*100:>14.2f}% {vals[4]*100:>14.2f}%')
        else:
            print(f'{name:<15} {vals[0]:>15.2f} {vals[1]:>15.2f} {vals[2]:>15.2f} {vals[3]:>15.2f} {vals[4]:>15.2f}')
    total = [(nav_15.iloc[-1]-1)*100, (nav_15_nodead.iloc[-1]-1)*100, (nav_20.iloc[-1]-1)*100, (nav_base.iloc[-1]-1)*100, (nav_bh.iloc[-1]-1)*100]
    print(f'{"总收益":<15} {total[0]:>14.2f}% {total[1]:>14.2f}% {total[2]:>14.2f}% {total[3]:>14.2f}% {total[4]:>14.2f}%')

    # ===== 绘图 =====
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), gridspec_kw={'height_ratios': [3, 1, 1, 1]})

    # 图1: 净值曲线
    ax1 = axes[0]
    ax1.plot(nav_15.index, nav_15.values, label='L=1.5 CPPI (死带0.05)', linewidth=2.5, color='#2196F3')
    ax1.plot(nav_15_nodead.index, nav_15_nodead.values, label='L=1.5 CPPI (无死带)', linewidth=1.5, color='#2196F3', linestyle='--', alpha=0.7)
    ax1.plot(nav_20.index, nav_20.values, label='L=2.0 CPPI (死带0.05)', linewidth=2, color='#9C27B0', linestyle='-.')
    ax1.plot(nav_base.index, nav_base.values, label='基线(L=1, 死带0.05)', linewidth=2, color='#4CAF50', linestyle=':')
    ax1.plot(nav_bh.index, nav_bh.values, label='Buy&Hold', linewidth=2, color='#FF9800', alpha=0.7)
    ax1.set_ylabel('累计净值')
    ax1.set_title('黄金策略净值曲线对比 (死带平滑 vs 无死带)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # 图2: 回撤
    ax2 = axes[1]
    dd_15 = nav_15 / nav_15.cummax() - 1
    dd_15_nd = nav_15_nodead / nav_15_nodead.cummax() - 1
    dd_bh = nav_bh / nav_bh.cummax() - 1
    ax2.fill_between(dd_15.index, dd_15.values*100, 0, alpha=0.4, color='#2196F3', label='L=1.5死带')
    ax2.fill_between(dd_15_nd.index, dd_15_nd.values*100, 0, alpha=0.2, color='#2196F3', label='L=1.5无死带')
    ax2.fill_between(dd_bh.index, dd_bh.values*100, 0, alpha=0.2, color='#FF9800', label='B&H')
    ax2.set_ylabel('回撤 (%)')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax2.axhline(y=-15, color='red', linestyle='--', alpha=0.5)

    # 图3: 仓位对比
    ax3 = axes[2]
    ax3.step(pos_15.index, pos_15.values, where='post', linewidth=1.5, color='#2196F3', label='L=1.5死带')
    ax3.step(pos_15_nodead.index, pos_15_nodead.values, where='post', linewidth=0.8, color='#2196F3', label='L=1.5无死带', alpha=0.5)
    ax3.set_ylabel('仓位')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=1.5, color='red', linestyle=':', alpha=0.5)

    # 图4: 相对净值 (vs B&H)
    ax4 = axes[3]
    ax4.plot(nav_15.index, (nav_15/nav_bh).values, label='L=1.5死带 / B&H', linewidth=1.5, color='#2196F3')
    ax4.plot(nav_15_nodead.index, (nav_15_nodead/nav_bh).values, label='L=1.5无死带 / B&H', linewidth=1, color='#2196F3', linestyle='--', alpha=0.7)
    ax4.set_ylabel('相对净值 (vs B&H)')
    ax4.set_xlabel('日期')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

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
        'nav_L1.5_CPPI_deadband': nav_15,
        'nav_L1.5_CPPI_nodead': nav_15_nodead,
        'nav_L2.0_CPPI_deadband': nav_20,
        'nav_baseline_L1': nav_base,
        'nav_BuyHold': nav_bh,
        'position_L1.5_deadband': pos_15.reindex(nav_15.index).fillna(method='ffill').fillna(0),
        'position_L1.5_nodead': pos_15_nodead.reindex(nav_15.index).fillna(method='ffill').fillna(0),
        'position_L2.0_deadband': pos_20.reindex(nav_20.index).fillna(method='ffill').fillna(0),
        'position_baseline': pos_base.reindex(nav_base.index).fillna(method='ffill').fillna(0),
        'drawdown_L1.5_deadband': dd_15,
        'drawdown_L1.5_nodead': dd_15_nd,
    })
    nav_df.to_csv(csv_path)
    print(f"净值数据已保存: {csv_path}")

    plt.show()


if __name__ == "__main__":
    main()
