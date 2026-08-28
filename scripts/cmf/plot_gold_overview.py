"""scripts/cmf/plot_gold_overview.py — 黄金策略综合介绍图 (重构布局).

布局 (简化、避免重叠):
  H. 顶部说明文字 (横跨整行)
  A. 全周期净值 + 业绩指标框 (下移, 不挡曲线)
  B1+B2 水平排 (左WF / 右2026 OOS)
  C. 逐年对比表 (窄, 让位给D)
  D. 大跌场景2x2 (加大)

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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

sys.path.insert(0, '/home/ll/Public/QuantNodes')

CACHE = "reports/gold/overview_data.pkl"
OUT = "reports/gold/gold_strategy_overview.png"

SEGMENTS = [
    ("2010-01-01", "2019-12-31", "滚动验证区间", "#E3F2FD"),
    ("2020-01-01", "2025-12-31", "样本内测试区间", "#FFF8E1"),
    ("2026-01-01", "2026-08-27", "纯样本外区间", "#E8F5E9"),
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
    mb = D["metrics_base"]
    ydf = D["yearly"]

    # 总体布局: 4行
    #   row0: 顶部说明 (小高度)
    #   row1: A图全周期净值 (主图, 大高度)
    #   row2: B1 + B2 (左右)
    #   row3: C表 + D场景 (左右, 高)
    fig = plt.figure(figsize=(20, 26))

    # ===== 顶部说明 =====
    axH = fig.add_axes([0.06, 0.965, 0.88, 0.025])
    axH.axis('off')
    axH.text(0.5, 0.5, HEADER_TEXT, transform=axH.transAxes,
             fontsize=11.5, ha='center', va='center', wrap=True,
             color='#37474F', linespacing=1.5,
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#ECEFF1',
                       edgecolor='#90A4AE', alpha=0.9, linewidth=1))

    # ===== A. 全周期净值 =====
    axA = fig.add_axes([0.06, 0.66, 0.88, 0.28])
    axA.set_title('A. 全周期业绩表现 (无前视 Open-to-Open + shift2 + 5bps成本)',
                 fontsize=14, fontweight='bold', loc='left')

    for s, e, label, color in SEGMENTS:
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        axA.axvspan(s_ts, e_ts, alpha=0.45, color=color, zorder=0)
        mid = s_ts + (e_ts - s_ts) / 2
        axA.text(mid, 1.02, label, transform=axA.get_xaxis_transform(),
                 fontsize=10.5, fontweight='bold', ha='center', va='bottom',
                 color='#455A64',
                 bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                           edgecolor='#78909C', alpha=0.85, linewidth=1))

    axA.plot(nav_15.index, nav_15.values, label='L=1.5 CPPI (生产策略)',
             linewidth=2.0, color='#1565C0', zorder=3)
    axA.plot(nav_base.index, nav_base.values, label='基线 L=1',
             linewidth=1.4, color='#2E7D32', linestyle='-.', zorder=3)
    axA.plot(nav_bh.index, nav_bh.values, label='Buy&Hold',
             linewidth=1.6, color='#F57C00', alpha=0.85, zorder=3)

    # 指标框下移到底部, 不挡曲线峰值
    box = "\n".join([
        "业绩指标对比 (CAGR / Sharpe / MaxDD / Calmar / 总收益)",
        f"{'指标':<9}{'L=1.5':>10}{'L=1基线':>10}{'B&H':>10}",
        "─" * 39,
        f"{'年化':<9}{pct(m15['cagr'], 2):>10}{pct(mb['cagr'], 2):>10}{pct(mbh['cagr'], 2):>10}",
        f"{'Sharpe':<9}{m15['sharpe']:>10.2f}{mb['sharpe']:>10.2f}{mbh['sharpe']:>10.2f}",
        f"{'MaxDD':<9}{pct(m15['mdd'], 1):>10}{pct(mb['mdd'], 1):>10}{pct(mbh['mdd'], 1):>10}",
        f"{'Calmar':<9}{m15['calmar']:>10.2f}{mb['calmar']:>10.2f}{mbh['calmar']:>10.2f}",
        f"{'总收益':<9}{pct(m15['total'], 0):>10}{pct(mb['total'], 0):>10}{pct(mbh['total'], 0):>10}",
    ])
    axA.text(0.015, 0.04, box, transform=axA.transAxes, fontsize=9.5,
             verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                       edgecolor='#1565C0', alpha=0.92, linewidth=1.5))

    ax1b = axA.twinx()
    dd15 = nav_15 / nav_15.cummax() - 1
    ax1b.fill_between(dd15.index, dd15.values * 100, 0, alpha=0.12, color='#EF5350', zorder=1)
    ax1b.set_ylabel('回撤 (%)', color='#EF5350', fontsize=10)
    ax1b.set_ylim(dd15.min() * 100 * 2.5, 5)
    ax1b.tick_params(axis='y', colors='#EF5350')
    axA.set_ylabel('累计净值', fontsize=11)
    axA.legend(loc='upper left', fontsize=10)
    axA.grid(True, alpha=0.3)
    axA.set_ylim(0.8, max(nav_15.max(), nav_bh.max()) * 1.18)

    # ===== B1 (左) + B2 (右) =====
    axB1 = fig.add_axes([0.06, 0.46, 0.42, 0.16])
    axB2 = fig.add_axes([0.52, 0.46, 0.42, 0.16])

    # B1: WF拼接
    wf = D["wf_nav"]
    axB1.plot(wf.index, wf.values, linewidth=2, color='#6A1B9A')
    axB1.fill_between(wf.index, 1.0, wf.values, alpha=0.10, color='#6A1B9A')
    axB1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    for year in range(2013, 2027):
        try:
            b = pd.Timestamp(f"{year}-01-01")
            if wf.index[0] <= b <= wf.index[-1]:
                axB1.axvline(x=b, color='gray', linestyle=':', alpha=0.3, linewidth=0.7)
        except Exception:
            pass
    wf_ret = wf.iloc[-1] / wf.iloc[0] - 1
    n_y = len(wf) / 252
    wf_cagr = (wf.iloc[-1]) ** (1 / n_y) - 1
    axB1.set_title(f'B1. Walk-Forward滚动验证 (15折拼接)\nCAGR={pct(wf_cagr, 2)}, 区间累计={pct(wf_ret, 0)}',
                   fontsize=11, fontweight='bold', loc='left')
    axB1.set_ylabel('累计净值', fontsize=9)
    axB1.grid(True, alpha=0.3)
    axB1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axB1.xaxis.set_major_locator(mdates.YearLocator(2))
    for lbl in axB1.get_xticklabels():
        lbl.set_rotation(45); lbl.set_fontsize(8)

    # B2: 2026纯OOS
    nav26 = D["nav26"]
    c26 = D["close26"]
    axB2.plot(nav26.index, nav26.values, color='#1565C0', linewidth=2.2, label='策略净值')
    ax2b = axB2.twinx()
    ax2b.plot(c26.index, c26.values, color='#F57C00', linewidth=1.4, alpha=0.75, label='金价')
    ax2b.set_ylabel('金价 (元/克)', color='#F57C00', fontsize=9)
    ax2b.tick_params(axis='y', colors='#F57C00')
    oos_ret = nav26.iloc[-1] / nav26.iloc[0] - 1
    crash_day = pd.Timestamp("2026-03-23")
    if crash_day in c26.index:
        axB2.axvline(x=crash_day, color='red', linestyle='--', alpha=0.6, linewidth=1.2)
        axB2.annotate('3/23\n金价-9.5%', xy=(crash_day, nav26.min()),
                      xytext=(crash_day - pd.Timedelta(days=42), nav26.min() * 0.99),
                      fontsize=8.5, color='red',
                      arrowprops=dict(arrowstyle='->', color='red', lw=1))
    axB2.set_title(f'B2. 2026纯样本外验证\nYTD区间收益={pct(oos_ret, 1)}',
                   fontsize=11, fontweight='bold', loc='left')
    axB2.set_ylabel('策略累计净值', color='#1565C0', fontsize=9)
    axB2.legend(loc='upper left', fontsize=8)
    axB2.grid(True, alpha=0.3)
    axB2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for lbl in axB2.get_xticklabels():
        lbl.set_rotation(45); lbl.set_fontsize(8)

    # ===== C. 逐年对比表 =====
    axC = fig.add_axes([0.06, 0.30, 0.42, 0.14])
    axC.axis('off')
    cols = ['年份', '区间收益', '波动率', 'MaxDD', 'Calmar', '',
            '区间收益 ', 'MaxDD ']
    cell_rows = []
    row_colors = []
    for year, r in ydf.iterrows():
        lr = r['L15_ret']; br = r['BH_ret']
        cell_rows.append([
            str(year),
            pct(lr, 1, sign=True) if pd.notna(lr) else '-',
            pct(r['L15_vol'], 1) if pd.notna(r['L15_vol']) else '-',
            pct(r['L15_mdd'], 1) if pd.notna(r['L15_mdd']) else '-',
            f"{r['L15_calmar']:.2f}" if pd.notna(r['L15_calmar']) else '-',
            '',
            pct(br, 1, sign=True) if pd.notna(br) else '-',
            pct(r['BH_mdd'], 1) if pd.notna(r['BH_mdd']) else '-',
        ])
        if pd.notna(lr) and pd.notna(br):
            row_colors.append('#E8F5E9' if lr >= br else '#FFEBEE')
        else:
            row_colors.append('white')

    tbl = axC.table(cellText=cell_rows, colLabels=cols,
                    cellLoc='center', loc='center',
                    colWidths=[0.10, 0.16, 0.13, 0.14, 0.13, 0.04, 0.16, 0.14])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.06)
    for j in range(len(cols)):
        cell = tbl[0, j]
        cell.set_facecolor('#263238')
        cell.set_text_props(color='white', fontweight='bold', fontsize=9)
        cell.set_height(0.10)
    for j in [1, 2, 3, 4]:
        tbl[0, j].set_facecolor('#1565C0')
    for j in [6, 7]:
        tbl[0, j].set_facecolor('#EF6C00')
    for i, r in enumerate(cell_rows, start=1):
        base = row_colors[i-1]
        for j in range(len(cols)):
            cell = tbl[i, j]
            cell.set_facecolor(base if j != 5 else 'white')
            cell.set_height(0.07)
            if j in (1, 6) and i <= len(cell_rows):
                txt = cell_rows[i-1][j]
                if txt and txt != '-' and '%' in txt:
                    if '+' in txt:
                        cell.set_text_props(color='#2E7D32', fontsize=9)
                    else:
                        cell.set_text_props(color='#C62828', fontsize=9)
                else:
                    cell.set_text_props(fontsize=9)
            else:
                cell.set_text_props(fontsize=9)
    axC.text(0.02, 1.04, '■ L=1.5 CPPI  ■ B&H',
             transform=axC.transAxes, fontsize=10,
             color='#1565C0', fontweight='bold')
    axC.set_title('C. 逐年业绩对比 (区间收益)',
                  fontsize=11, fontweight='bold', loc='left', pad=30)

    # ===== D. 大跌场景 (4图, 2×2) =====
    axD_title = fig.add_axes([0.52, 0.30, 0.42, 0.14])
    axD_title.axis('off')
    axD_title.set_title('D. 大跌场景下的仓位管理', fontsize=11,
                       fontweight='bold', loc='left', pad=10)

    inner_gs = GridSpec(2, 2, figure=fig, left=0.52, right=0.94, top=0.43, bottom=0.30,
                        wspace=0.30, hspace=0.55)
    for i, (name, sdata) in enumerate(D["scenarios"].items()):
        ax = fig.add_subplot(inner_gs[i // 2, i % 2])
        c = sdata["close"]
        p = sdata["pos"]
        ax.plot(c.index, c.values, color='#F57C00', linewidth=1.3, label='金价(左轴)')
        ax.set_ylabel('金价', color='#F57C00', fontsize=8)
        ax.tick_params(axis='y', labelcolor='#F57C00', labelsize=7)
        ax2 = ax.twinx()
        ax2.step(p.index, p.values, where='post', color='#1565C0', linewidth=1.5, label='仓位')
        ax2.set_ylabel('仓位', color='#1565C0', fontsize=8)
        ax2.set_ylim(-0.05, 1.9)
        ax2.tick_params(axis='y', labelcolor='#1565C0', labelsize=7)
        if len(c) > 2:
            peak_idx = c.values.argmax()
            trough_after = c.values[peak_idx:].min()
            drop = (trough_after / c.values[peak_idx] - 1) * 100
            ax.set_title(f'{name}\n金价区间{drop:+.1f}%', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30); lbl.set_fontsize(7)

    fig.suptitle('黄金量化配置策略 — 业绩概览',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ 综合介绍图已保存: {OUT}")


if __name__ == "__main__":
    main()
