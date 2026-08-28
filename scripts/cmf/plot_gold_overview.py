"""scripts/cmf/plot_gold_overview.py — 黄金策略综合介绍图.

布局:
  顶部: 笼统文字说明 (不涉及策略细节)
  A. 全周期净值 + 业绩指标框 + 测试区间底色切分
  B1. Walk-Forward滚动验证 | B2. 2026纯OOS
  C. 逐年区间收益对比表 | D. 大跌场景仓位 2×2

数据来自缓存: reports/gold/overview_data.pkl
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

# 区间切分定义 (A图底色)
SEGMENTS = [
    ("2010-01-01", "2019-12-31", "滚动验证区间", "#E3F2FD"),   # 浅蓝
    ("2020-01-01", "2025-12-31", "样本内测试区间", "#FFF8E1"),  # 浅黄
    ("2026-01-01", "2026-08-27", "纯样本外区间", "#E8F5E9"),    # 浅绿
]

HEADER_TEXT = (
    "本报告为黄金量化配置策略的业绩概览。策略基于多信号融合与动态风险控制框架，"
    "在严格控制回撤的前提下捕捉黄金中长期趋势收益。"
    "以下展示策略全周期业绩表现、滚动验证结果、样本外测试表现，"
    "以及历史极端行情下的仓位管理情况。回测均考虑交易成本与数据可得性约束。"
)


def setup_cjk(font_name):
    import matplotlib.font_manager as fm
    for fpath in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
        try:
            fm.fontManager.addfont(fpath)
        except Exception:
            pass
    matplotlib.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False


def pct(v, digits=1, sign=False):
    s = f"{v*100:+.{digits}f}%" if sign else f"{v*100:.{digits}f}%"
    return s


def main():
    with open(CACHE, "rb") as f:
        D = pickle.load(f)

    setup_cjk("Noto Sans CJK JP")
    nav_15 = D["nav_15"]
    nav_base = D["nav_base"]
    nav_bh = D["nav_bh"]
    m15 = D["metrics_15"]
    mbh = D["metrics_bh"]
    ydf = D["yearly"]

    fig = plt.figure(figsize=(20, 27))
    gs = GridSpec(4, 2, figure=fig, height_ratios=[0.06, 2.6, 1.4, 1.75],
                  hspace=0.30, wspace=0.18)

    # ================= 顶部说明文字 =================
    axH = fig.add_subplot(gs[0, :])
    axH.axis('off')
    axH.text(0.5, 0.75, HEADER_TEXT, transform=axH.transAxes,
             fontsize=12.5, ha='center', va='center', wrap=True,
             color='#37474F', linespacing=1.8,
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#ECEFF1',
                       edgecolor='#90A4AE', alpha=0.9, linewidth=1.2))

    # ================= A. 全周期净值 + 区间切分 =================
    axA = fig.add_subplot(gs[1, :])

    # 区间底色
    for s, e, label, color in SEGMENTS:
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        axA.axvspan(s_ts, e_ts, alpha=0.45, color=color, zorder=0)
        mid = s_ts + (e_ts - s_ts) / 2
        axA.text(mid, 0.06, label, transform=axA.get_xaxis_transform(),
                 fontsize=11, fontweight='bold', ha='center', va='bottom',
                 color='#455A64',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='#78909C', alpha=0.85, linewidth=1))

    axA.plot(nav_15.index, nav_15.values, label='L=1.5 CPPI (生产策略)',
             linewidth=2.2, color='#1565C0', zorder=3)
    axA.plot(nav_base.index, nav_base.values, label='基线 L=1',
             linewidth=1.5, color='#2E7D32', linestyle='-.', zorder=3)
    axA.plot(nav_bh.index, nav_bh.values, label='Buy&Hold',
             linewidth=1.8, color='#F57C00', alpha=0.85, zorder=3)

    # 回撤阴影
    ax1b = axA.twinx()
    dd15 = nav_15 / nav_15.cummax() - 1
    ax1b.fill_between(dd15.index, dd15.values * 100, 0, alpha=0.15, color='#EF5350', zorder=1)
    ax1b.set_ylabel('回撤 (%)', color='#EF5350', fontsize=11)
    ax1b.set_ylim(dd15.min() * 100 * 2.2, 8)
    ax1b.tick_params(axis='y', colors='#EF5350')

    axA.set_ylabel('累计净值', fontsize=12)
    axA.set_title('A. 全周期业绩表现 (无前视 Open-to-Open + shift2 + 5bps成本)',
                  fontsize=15, fontweight='bold', loc='left')
    axA.legend(loc='upper left', fontsize=11)
    axA.grid(True, alpha=0.3)
    axA.set_ylim(0.8, max(nav_15.max(), nav_bh.max()) * 1.18)

    # 指标标注框
    box = ("\n".join([
        "业绩指标对比",
        f"{'指标':<9}{'L=1.5 CPPI':>11}{'B&H':>9}",
        "─" * 31,
        f"{'年化收益':<9}{pct(m15['cagr'], 2):>11}{pct(mbh['cagr'], 2):>9}",
        f"{'Sharpe':<9}{m15['sharpe']:>11.2f}{mbh['sharpe']:>9.2f}",
        f"{'MaxDD':<9}{pct(m15['mdd'], 1):>11}{pct(mbh['mdd'], 1):>9}",
        f"{'Calmar':<9}{m15['calmar']:>11.2f}{mbh['calmar']:>9.2f}",
        f"{'总收益':<9}{pct(m15['total'], 0):>11}{pct(mbh['total'], 0):>9}",
    ]))
    axA.text(0.988, 0.965, box, transform=axA.transAxes, fontsize=10.5,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                       edgecolor='#1565C0', alpha=0.92, linewidth=1.5))

    # ================= B1. Walk-Forward =================
    axB1 = fig.add_subplot(gs[2, 0])
    wf = D["wf_nav"]
    axB1.plot(wf.index, wf.values, linewidth=2, color='#6A1B9A')
    axB1.fill_between(wf.index, 1.0, wf.values, alpha=0.12, color='#6A1B9A')
    axB1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
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
    axB1.set_title(f'B1. Walk-Forward滚动验证 (15折测试段拼接, CAGR={pct(wf_cagr, 2)}, 区间累计={pct(wf_ret, 0)})',
                   fontsize=12, fontweight='bold', loc='left')
    axB1.set_ylabel('累计净值')
    axB1.grid(True, alpha=0.3)
    axB1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axB1.xaxis.set_major_locator(mdates.YearLocator(2))
    for lbl in axB1.get_xticklabels():
        lbl.set_rotation(45)

    # ================= B2. 2026纯OOS =================
    axB2 = fig.add_subplot(gs[2, 1])
    nav26 = D["nav26"]
    c26 = D["close26"]
    axB2.plot(nav26.index, nav26.values, color='#1565C0', linewidth=2.2, label='策略净值')
    ax2b = axB2.twinx()
    ax2b.plot(c26.index, c26.values, color='#F57C00', linewidth=1.4, alpha=0.75, label='金价(右轴)')
    ax2b.set_ylabel('金价 (元/克)', color='#F57C00', fontsize=10)
    ax2b.tick_params(axis='y', colors='#F57C00')
    oos_ret = nav26.iloc[-1] / nav26.iloc[0] - 1
    crash_day = pd.Timestamp("2026-03-23")
    if crash_day in c26.index:
        axB2.axvline(x=crash_day, color='red', linestyle='--', alpha=0.6, linewidth=1.2)
        axB2.annotate('3/23金价单日-9.5%', xy=(crash_day, nav26.min()),
                      xytext=(crash_day - pd.Timedelta(days=38), nav26.min() * 0.99),
                      fontsize=9, color='red',
                      arrowprops=dict(arrowstyle='->', color='red', lw=1))
    axB2.set_title(f'B2. 2026纯样本外验证 (YTD区间收益={pct(oos_ret, 1)})',
                   fontsize=12, fontweight='bold', loc='left')
    axB2.set_ylabel('策略累计净值', color='#1565C0', fontsize=10)
    axB2.legend(loc='upper left', fontsize=9)
    axB2.grid(True, alpha=0.3)
    axB2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for lbl in axB2.get_xticklabels():
        lbl.set_rotation(45)

    # ================= C. 逐年区间收益对比表 (美化) =================
    axC = fig.add_subplot(gs[3, 0])
    axC.axis('off')

    cols = ['年份', '区间收益', '波动率', 'MaxDD', '夏普', 'Calmar', '',
            '区间收益 ', 'MaxDD ']
    cell_rows = []
    row_colors = []
    for year, r in ydf.iterrows():
        l15_ret = r['L15_ret']
        bh_ret = r['BH_ret']
        cell_rows.append([
            str(year),
            pct(l15_ret, 1, sign=True) if pd.notna(l15_ret) else '-',
            pct(r['L15_vol'], 1) if pd.notna(r['L15_vol']) else '-',
            pct(r['L15_mdd'], 1) if pd.notna(r['L15_mdd']) else '-',
            f"{r['L15_sharpe']:.2f}" if pd.notna(r['L15_sharpe']) else '-',
            f"{r['L15_calmar']:.2f}" if pd.notna(r['L15_calmar']) else '-',
            '',
            pct(bh_ret, 1, sign=True) if pd.notna(bh_ret) else '-',
            pct(r['BH_mdd'], 1) if pd.notna(r['BH_mdd']) else '-',
        ])
        if pd.notna(l15_ret) and pd.notna(bh_ret):
            row_colors.append('#E8F5E9' if l15_ret >= bh_ret else '#FFEBEE')
        else:
            row_colors.append('white')

    tbl = axC.table(cellText=cell_rows, colLabels=cols,
                    cellLoc='center', loc='center',
                    colWidths=[0.09, 0.13, 0.11, 0.12, 0.10, 0.11, 0.03, 0.13, 0.12])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.08)
    # 表头
    for j in range(len(cols)):
        cell = tbl[0, j]
        cell.set_facecolor('#263238')
        cell.set_text_props(color='white', fontweight='bold', fontsize=9)
        cell.set_height(0.09)
    # 左组表头蓝, 右组橙
    for j in [1, 2, 3, 4, 5]:
        tbl[0, j].set_facecolor('#1565C0')
    for j in [7, 8]:
        tbl[0, j].set_facecolor('#EF6C00')
    # 数据行: 斑马纹+胜负色, 收益正绿负红字
    for i, r in enumerate(cell_rows, start=1):
        base = row_colors[i-1]
        for j in range(len(cols)):
            cell = tbl[i, j]
            cell.set_facecolor(base if j != 6 else 'white')
            cell.set_height(0.072)
            if j in (1, 7) and i <= len(cell_rows):
                txt = r[j] if j < len(r) else ''
                if txt and txt != '-' and '%' in txt:
                    if '+' in txt:
                        cell.set_text_props(color='#2E7D32', fontsize=9)
                    else:
                        cell.set_text_props(color='#C62828', fontsize=9)
                else:
                    cell.set_text_props(fontsize=9)
            else:
                cell.set_text_props(fontsize=9)
    # 区分线
    axC.text(0.02, 1.03, '■ L=1.5 CPPI', transform=axC.transAxes,
             fontsize=10, color='#1565C0', fontweight='bold')
    axC.text(0.55, 1.03, '■ Buy&Hold', transform=axC.transAxes,
             fontsize=10, color='#EF6C00', fontweight='bold')
    axC.set_title('C. 逐年业绩对比 (区间收益, 非年化)',
                  fontsize=12, fontweight='bold', loc='left', pad=28)

    # ================= D. 大跌场景仓位 2×2 → 竖排4图 =================
    axD = fig.add_subplot(gs[3, :])
    axD.axis('off')
    axD.set_title('D. 黄金大跌场景下的仓位管理 (金价 + 策略仓位)', fontsize=12,
                  fontweight='bold', loc='left', pad=10)

    gs_scen = gs[3, :].subgridspec(1, 4, wspace=0.24)
    for i, (name, sdata) in enumerate(D["scenarios"].items()):
        ax = fig.add_subplot(gs_scen[0, i])
        c = sdata["close"]
        p = sdata["pos"]
        ax.plot(c.index, c.values, color='#F57C00', linewidth=1.5, label='金价(左轴)')
        ax.set_ylabel('金价', color='#F57C00', fontsize=9)
        ax.tick_params(axis='y', labelcolor='#F57C00', labelsize=8)
        ax2 = ax.twinx()
        ax2.step(p.index, p.values, where='post', color='#1565C0', linewidth=1.8, label='仓位(右轴)')
        ax2.set_ylabel('仓位', color='#1565C0', fontsize=9)
        ax2.set_ylim(-0.05, 1.9)
        ax2.tick_params(axis='y', labelcolor='#1565C0', labelsize=8)
        if len(c) > 2:
            peak_idx = c.values.argmax()
            trough_after = c.values[peak_idx:].min()
            drop = (trough_after / c.values[peak_idx] - 1) * 100
            ax.set_title(f'{name}\n区间金价{drop:+.1f}%', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_fontsize(7.5)

    fig.suptitle('黄金量化配置策略 — 业绩概览',
                 fontsize=18, fontweight='bold', y=0.998)

    plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ 综合介绍图已保存: {OUT}")


if __name__ == "__main__":
    main()
