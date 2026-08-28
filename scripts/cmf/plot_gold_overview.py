"""scripts/cmf/plot_gold_overview.py — 黄金策略综合介绍图.

布局 (从缓存加载, 秒级生成):
  A. 全周期净值 + 业绩指标标注框
  B1. Walk-Forward滚动验证 (15折拼接)
  B2. 2026纯OOS (净值+金价+仓位)
  C. 逐年指标对比表 (L=1.5 vs B&H)
  D. 大跌场景日频仓位 (2×2)

输出: reports/gold/gold_strategy_overview.png
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

sys.path.insert(0, '/home/ll/Public/QuantNodes')

CACHE = "reports/gold/overview_data.pkl"
OUT = "reports/gold/gold_strategy_overview.png"


def setup_cjk(font_name):
    import matplotlib.font_manager as fm
    # 直接注册字体文件 (matplotlib字体缓存可能不含系统CJK字体)
    for fpath in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
        try:
            fm.fontManager.addfont(fpath)
        except Exception:
            pass
    matplotlib.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False


def pct(v, digits=1):
    return f"{v*100:.{digits}f}%"


def main():
    with open(CACHE, "rb") as f:
        D = pickle.load(f)

    setup_cjk("Noto Sans CJK JP")
    nav_15 = D["nav_15"]
    nav_base = D["nav_base"]
    nav_bh = D["nav_bh"]
    m15 = D["metrics_15"]
    mbh = D["metrics_bh"]

    fig = plt.figure(figsize=(20, 26))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[2.6, 1.4, 1.6, 1.6],
                  hspace=0.32, wspace=0.18)

    # ================= A. 全周期净值 + 指标框 =================
    axA = fig.add_subplot(gs[0, :])
    axA.plot(nav_15.index, nav_15.values, label='L=1.5 CPPI (生产策略)',
             linewidth=2.2, color='#1565C0')
    axA.plot(nav_base.index, nav_base.values, label='基线 L=1',
             linewidth=1.5, color='#2E7D32', linestyle='-.')
    ax1b = axA.twinx()
    dd15 = nav_15 / nav_15.cummax() - 1
    ax1b.fill_between(dd15.index, dd15.values * 100, 0, alpha=0.18, color='#EF5350')
    ax1b.set_ylabel('回撤 (%)', color='#EF5350', fontsize=11)
    ax1b.set_ylim(dd15.min() * 100 * 2.2, 5)
    ax1b.tick_params(axis='y', colors='#EF5350')

    axA.plot(nav_bh.index, nav_bh.values, label='Buy&Hold',
             linewidth=1.8, color='#F57C00', alpha=0.85)
    axA.set_ylabel('累计净值', fontsize=12)
    axA.set_title('A. 全周期业绩表现 (无前视 Open-to-Open + shift2 + 5bps成本)',
                  fontsize=15, fontweight='bold', loc='left')
    axA.legend(loc='upper left', fontsize=11)
    axA.grid(True, alpha=0.3)
    axA.set_ylim(0.8, max(nav_15.max(), nav_bh.max()) * 1.15)

    # 指标标注框
    box = ("\n".join([
        "业绩指标对比",
        f"{'指标':<10}{'L=1.5 CPPI':>12}{'B&H':>10}",
        "─" * 34,
        f"{'年化收益':<10}{pct(m15['cagr'], 2):>12}{pct(mbh['cagr'], 2):>10}",
        f"{'Sharpe':<10}{m15['sharpe']:>12.2f}{mbh['sharpe']:>10.2f}",
        f"{'MaxDD':<10}{pct(m15['mdd'], 1):>12}{pct(mbh['mdd'], 1):>10}",
        f"{'Calmar':<10}{m15['calmar']:>12.2f}{mbh['calmar']:>10.2f}",
        f"{'总收益':<10}{pct(m15['total'], 0):>12}{pct(mbh['total'], 0):>10}",
    ]))
    axA.text(0.985, 0.97, box, transform=axA.transAxes, fontsize=10.5,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                       edgecolor='#1565C0', alpha=0.92, linewidth=1.5),
             family=D["font_cjk"])

    # ================= B1. Walk-Forward =================
    axB1 = fig.add_subplot(gs[1, 0])
    wf = D["wf_nav"]
    axB1.plot(wf.index, wf.values, linewidth=2, color='#6A1B9A')
    axB1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    # 标注每年分隔
    for year in range(2013, 2027):
        try:
            boundary = pd.Timestamp(f"{year}-01-01")
            if wf.index[0] <= boundary <= wf.index[-1]:
                axB1.axvline(x=boundary, color='gray', linestyle=':', alpha=0.35, linewidth=0.8)
        except Exception:
            pass
    wf_ret = wf.iloc[-1] / wf.iloc[0] - 1
    n_y = len(wf) / 252
    wf_cagr = (wf.iloc[-1]) ** (1 / n_y) - 1
    axB1.set_title(f'B1. Walk-Forward滚动验证 (15折测试段拼接, CAGR={pct(wf_cagr, 2)}, 总收益={pct(wf_ret, 0)})',
                   fontsize=12, fontweight='bold', loc='left')
    axB1.set_ylabel('累计净值')
    axB1.grid(True, alpha=0.3)
    axB1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axB1.xaxis.set_major_locator(mdates.YearLocator(2))
    for lbl in axB1.get_xticklabels():
        lbl.set_rotation(45)

    # ================= B2. 2026纯OOS =================
    axB2 = fig.add_subplot(gs[1, 1])
    nav26 = D["nav26"]
    c26 = D["close26"]
    p26 = D["pos26"]
    axB2.plot(nav26.index, nav26.values, color='#1565C0', linewidth=2.2, label='策略净值')
    ax2b = axB2.twinx()
    ax2b.plot(c26.index, c26.values, color='#F57C00', linewidth=1.4, alpha=0.75, label='金价(右轴)')
    ax2b.set_ylabel('金价 (元/克)', color='#F57C00', fontsize=10)
    ax2b.tick_params(axis='y', colors='#F57C00')
    # 标注3月崩盘
    crash_day = pd.Timestamp("2026-03-23")
    if crash_day in c26.index:
        axB2.axvline(x=crash_day, color='red', linestyle='--', alpha=0.6, linewidth=1.2)
        axB2.annotate('3/23金价单日-9.5%', xy=(crash_day, nav26.min()),
                      xytext=(crash_day - pd.Timedelta(days=35), nav26.min() * 0.995),
                      fontsize=9, color='red',
                      arrowprops=dict(arrowstyle='->', color='red', lw=1))
    axB2.set_title('B2. 2026纯OOS验证 (完全样本外, YTD收益见净值)',
                   fontsize=12, fontweight='bold', loc='left')
    axB2.set_ylabel('策略累计净值', color='#1565C0', fontsize=10)
    axB2.legend(loc='upper left', fontsize=9)
    axB2.grid(True, alpha=0.3)
    axB2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for lbl in axB2.get_xticklabels():
        lbl.set_rotation(45)

    # ================= C. 逐年指标表 =================
    axC = fig.add_subplot(gs[2, 0])
    axC.axis('off')
    ydf = D["yearly"]

    cols = ['年份', '年化收益', '波动率', 'MaxDD', '夏普', 'Calmar', '|', '年化收益 ', 'MaxDD ']
    cell_rows = []
    row_colors = []
    for year, r in ydf.iterrows():
        l15_ret = r['L15_ret'] if pd.notna(r['L15_ret']) else np.nan
        bh_ret = r['BH_ret'] if pd.notna(r['BH_ret']) else np.nan
        cell_rows.append([
            str(year),
            pct(l15_ret, 1) if pd.notna(l15_ret) else '-',
            pct(r['L15_vol'], 1) if pd.notna(r['L15_vol']) else '-',
            pct(r['L15_mdd'], 1) if pd.notna(r['L15_mdd']) else '-',
            f"{r['L15_sharpe']:.2f}" if pd.notna(r['L15_sharpe']) else '-',
            f"{r['L15_calmar']:.2f}" if pd.notna(r['L15_calmar']) else '-',
            '|',
            pct(bh_ret, 1) if pd.notna(bh_ret) else '-',
            pct(r['BH_mdd'], 1) if pd.notna(r['BH_mdd']) else '-',
        ])
        # 颜色: 策略跑赢绿色, 跑输红色
        if pd.notna(l15_ret) and pd.notna(bh_ret):
            row_colors.append('#E8F5E9' if l15_ret >= bh_ret else '#FFEBEE')
        else:
            row_colors.append('white')

    tbl = axC.table(cellText=cell_rows, colLabels=cols,
                    cellLoc='center', loc='center',
                    colWidths=[0.09, 0.13, 0.11, 0.12, 0.10, 0.11, 0.03, 0.13, 0.12])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.05)
    # 表头样式
    for j in range(len(cols)):
        cell = tbl[0, j]
        cell.set_facecolor('#1565C0')
        cell.set_text_props(color='white', fontweight='bold', fontsize=8.5)
    # 分组表头着色
    for j in [1, 2, 3, 4, 5]:
        tbl[0, j].set_facecolor('#1565C0')
    for j in [7, 8]:
        tbl[0, j].set_facecolor('#F57C00')
    # 行着色
    for i, rc in enumerate(row_colors, start=1):
        for j in range(len(cols)):
            tbl[i, j].set_facecolor(rc)
            if j == 6:
                tbl[i, j].set_facecolor('white')
    axC.set_title('C. 逐年指标对比 (左: L=1.5 CPPI | 右: B&H 年化与MaxDD)',
                  fontsize=12, fontweight='bold', loc='left', pad=20)

    # ================= D. 大跌场景仓位 =================
    axD = fig.add_subplot(gs[2, 1])
    axD.axis('off')
    axD.set_title('D. 黄金大跌场景仓位变化 (见下方2×2)', fontsize=12,
                  fontweight='bold', loc='left', pad=20)

    gs_scen = gs[3, :].subgridspec(1, 4, wspace=0.22)
    for i, (name, sdata) in enumerate(D["scenarios"].items()):
        ax = fig.add_subplot(gs_scen[0, i])
        c = sdata["close"]
        p = sdata["pos"]
        ax.plot(c.index, c.values, color='#F57C00', linewidth=1.4, label='金价(左轴)')
        ax.set_ylabel('金价', color='#F57C00', fontsize=9)
        ax.tick_params(axis='y', labelcolor='#F57C00', labelsize=8)
        ax2 = ax.twinx()
        ax2.step(p.index, p.values, where='post', color='#1565C0', linewidth=1.8, label='仓位(右轴)')
        ax2.set_ylabel('仓位', color='#1565C0', fontsize=9)
        ax2.set_ylim(-0.05, 1.9)
        ax2.tick_params(axis='y', labelcolor='#1565C0', labelsize=8)
        # 标注区间跌幅
        if len(c) > 2:
            peak_idx = c.values.argmax()
            trough_after = c.values[peak_idx:].min()
            drop = (trough_after / c.values[peak_idx] - 1) * 100
            ax.set_title(f'{name}\n区间金价{drop:+.1f}%', fontsize=9.5, fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_fontsize(7.5)

    fig.suptitle('黄金策略综合介绍 — 分层集成 + Δtips + CPPI (无前视O2O · shift2 · 5bps)',
                 fontsize=17, fontweight='bold', y=0.995)

    plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ 综合介绍图已保存: {OUT}")


if __name__ == "__main__":
    main()
