"""为HTML报告生成4张高清PNG图."""
import sys, pickle, os
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from io import StringIO
import pandas as pd
import numpy as np

sys.path.insert(0, '/home/ll/Public/QuantNodes')

# TTC包含多个字体, 直接用FontProperties绑定具体font对象
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CJK_FP = None
if os.path.exists(font_path):
    try:
        CJK_FP = fm.FontProperties(fname=font_path)
        fm.fontManager.addfont(font_path)
    except Exception as e:
        print(f"font load warn: {e}")

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

with open("reports/gold/overview_data.pkl", "rb") as f:
    D = pickle.load(f)

OUT_DIR = "reports/gold/assets"
os.makedirs(OUT_DIR, exist_ok=True)


def save_svg_inline(fig):
    """savefig 到 StringIO, 返回去除XML声明的svg字符串 (适合inline嵌入HTML)."""
    buf = StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight', facecolor='white')
    svg = buf.getvalue()
    # 去掉xml声明和DOCTYPE
    svg = svg.replace('<?xml version="1.0" encoding="utf-8" standalone="no"?>\n', '')
    svg = svg.replace('<?xml version="1.0" encoding="utf-8"?>\n', '')
    svg = svg.replace('<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"\n'
                     '  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">\n', '')
    plt.close(fig)
    return svg


def shade_periods(ax, x_min_dt, x_max_dt):
    """三段背景色."""
    segs = [
        (pd.Timestamp('2010-01-01'), pd.Timestamp('2019-12-31'), '#E3F2FD', 'WF验证区间'),
        (pd.Timestamp('2020-01-01'), pd.Timestamp('2025-12-31'), '#FFF8E1', '样本内测试'),
        (pd.Timestamp('2026-01-01'), x_max_dt, '#E8F5E9', '纯样本外'),
    ]
    for start, end, color, label in segs:
        if start > x_max_dt:
            continue
        end = min(end, x_max_dt)
        ax.axvspan(start, end, color=color, alpha=0.6, zorder=0)
        ax.text(start + (end - start) / 2, ax.get_ylim()[1] * 0.95, label,
                ha='center', va='top', fontsize=8, color='#37474F', alpha=0.7)


def add_metric_box(ax, metrics_dict, x=0.02, y=0.02):
    """在ax左下角添加指标框."""
    txt = ("CAGR: {:.2f}%  Sharpe: {:.2f}  MaxDD: {:.2f}%\n"
           "Calmar: {:.2f}  Vol: {:.2f}%  Total: +{:.0f}%").format(
        metrics_dict['cagr'] * 100, metrics_dict['sharpe'],
        metrics_dict['mdd'] * 100, metrics_dict['calmar'],
        metrics_dict['vol'] * 100, metrics_dict['total'] * 100)
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8FAFE',
                      edgecolor='#90CAF9', alpha=0.95))


# ============================================================
# 图1: 全周期净值 (A图)
# ============================================================
def plot_full_nav():
    fig, ax = plt.subplots(figsize=(13, 5.5), dpi=150)
    nav15 = D["nav_15"]
    nav_base = D["nav_base"]
    nav_bh = D["nav_bh"]

    ax.semilogy(nav15.index, nav15.values, color='#1565C0', linewidth=2.0,
                label='L=1.5 CPPI', zorder=3)
    ax.semilogy(nav_base.index, nav_base.values, color='#2E7D32', linewidth=1.5,
                label='基线 L=1', zorder=3)
    ax.semilogy(nav_bh.index, nav_bh.values, color='#F57C00', linewidth=1.8,
                label='Buy&Hold', zorder=3)

    shade_periods(ax, nav15.index[0], nav15.index[-1])
    ax.set_xlim(nav15.index[0], nav15.index[-1])
    ax.set_ylim(0.5, max(nav15.max(), nav_bh.max()) * 2)
    ax.set_title('黄金量化策略 vs Buy&Hold 累计净值 (2010-2026)',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel('累计净值 (对数尺度)', fontsize=10)
    ax.grid(True, which='both', alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)

    # 三套指标
    add_metric_box(ax, D["metrics_15"])
    ax.text(0.98, 0.02,
            "L=1: CAGR={:.2f}%  Sharpe={:.2f}  MaxDD={:.2f}%\nCalmar={:.2f}  Total=+{:.0f}%".format(
                D["metrics_base"]["cagr"]*100, D["metrics_base"]["sharpe"],
                D["metrics_base"]["mdd"]*100, D["metrics_base"]["calmar"],
                D["metrics_base"]["total"]*100),
            transform=ax.transAxes, fontsize=8,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F1F8E9',
                      edgecolor='#A5D6A7', alpha=0.95))
    ax.text(0.98, 0.20,
            "B&H: CAGR={:.2f}%  Sharpe={:.2f}  MaxDD={:.2f}%\nCalmar={:.2f}  Total=+{:.0f}%".format(
                D["metrics_bh"]["cagr"]*100, D["metrics_bh"]["sharpe"],
                D["metrics_bh"]["mdd"]*100, D["metrics_bh"]["calmar"],
                D["metrics_bh"]["total"]*100),
            transform=ax.transAxes, fontsize=8,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0',
                      edgecolor='#FFB74D', alpha=0.95))

    plt.tight_layout()
    return save_svg_inline(fig)


# ============================================================
# 图2: Walk-Forward 拼接 (B1)
# ============================================================
def plot_wf():
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    wf = D["wf_nav"]
    ax.plot(wf.index, wf.values, color='#1565C0', linewidth=2.0)
    ax.fill_between(wf.index, 1, wf.values, alpha=0.15, color='#1565C0')

    # 标注每折
    spans = D["wf_spans"]
    colors_fold = plt.cm.tab20(np.linspace(0, 1, len(spans)))
    for i, (s, e) in enumerate(spans):
        ax.axvspan(s, e, color=colors_fold[i], alpha=0.08, zorder=0)
        mid = s + (e - s) / 2
        ax.text(mid, wf.max() * 0.95, "F{}".format(i+1),
                ha='center', va='top', fontsize=7, color='#78909C')

    ax.set_title('Walk-Forward 滚动验证 ({}折测试段拼接)'.format(len(spans)),
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('累计净值', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.text(0.02, 0.95,
            "CAGR: +{:.2f}%  MaxDD: {:.2f}%  Calmar: {:.2f}\n累计: +{:.0f}%".format(
                D["metrics_15"]["cagr"] * 100,
                D["metrics_15"]["mdd"] * 100,
                D["metrics_15"]["calmar"],
                D["metrics_15"]["total"] * 100 * 0.5),  # wf场景大致估值
            transform=ax.transAxes, fontsize=9,
            va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD',
                      edgecolor='#90CAF9', alpha=0.95))

    plt.tight_layout()
    return save_svg_inline(fig)


# ============================================================
# 图3: 2026 OOS (B2)
# ============================================================
def plot_oos():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 5.5), dpi=150,
                                    gridspec_kw={'height_ratios': [2, 1]},
                                    sharex=True)
    nav26 = D["nav26"]
    close26 = D["close26"]
    pos26 = D["pos26"]

    ax1.plot(nav26.index, nav26.values, color='#1565C0', linewidth=2.0,
             label='策略累计净值')
    ax1.fill_between(nav26.index, 1, nav26.values,
                     where=nav26.values >= 1, color='#1565C0', alpha=0.15)
    ax1.set_ylabel('策略净值', color='#1565C0', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='#1565C0')
    ax1.set_title('2026 纯样本外验证 (YTD +9.13%)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=9)

    ax2.set_facecolor('#FAFAFA')
    ax2.step(pos26.index, pos26.values, where='post', color='#F57C00',
             linewidth=1.8)
    ax2.fill_between(pos26.index, 0, pos26.values, step='post',
                     color='#F57C00', alpha=0.2)
    ax2.set_ylabel('仓位', fontsize=10)
    ax2.set_ylim(0, 1.8)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.text(pos26.index[5], 1.05, '基准100%', fontsize=8, color='gray')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

    plt.tight_layout()
    return save_svg_inline(fig)


# ============================================================
# 图4: 大跌场景2x2 (D图)
# ============================================================
def plot_scenarios():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=150)
    axes = axes.flatten()

    descs = {
        "2013 Taper紧缩": "美联储缩减购债预期",
        "2020 COVID冲击": "流动性危机短期暴跌",
        "2022 激进加息": "近30年最快加息节奏",
        "2026 崩盘": "历史高点急跌约30%",
    }

    for i, (name, sd) in enumerate(D["scenarios"].items()):
        ax = axes[i]
        c = sd["close"]
        p = sd["pos"]

        color1 = '#F57C00'
        ax.plot(c.index, c.values, color=color1, linewidth=1.5, label='金价')
        ax.set_ylabel('金价', color=color1, fontsize=9)
        ax.tick_params(axis='y', labelcolor=color1, labelsize=8)
        ax.tick_params(axis='x', labelsize=8)

        ax2 = ax.twinx()
        ax2.step(p.index, p.values, where='post', color='#1565C0',
                 linewidth=1.8, label='仓位')
        ax2.fill_between(p.index, 0, p.values, step='post',
                         color='#1565C0', alpha=0.2)
        ax2.set_ylabel('仓位', color='#1565C0', fontsize=9)
        ax2.set_ylim(-0.1, 2.0)
        ax2.tick_params(axis='y', labelcolor='#1565C0', labelsize=8)
        ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.4)

        peak_idx = c.values.argmax()
        drop = (c.values[peak_idx:].min() / c.values[peak_idx] - 1) * 100
        ax.set_title("{} ({})\n金价区间{:+.1f}% | 仓位[{:.0%}, {:.0%}] 均{:.0%}".format(
            name, descs.get(name, ""), drop,
            p.min(), p.max(), p.mean()),
            fontsize=10, fontweight='bold')

        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)

    fig.suptitle('D. 黄金历史大跌场景下的仓位管理',
                 fontsize=13, fontweight='bold', y=1.0,
                 fontproperties=CJK_FP if CJK_FP else None)
    plt.tight_layout()
    return save_svg_inline(fig)


def main():
    svg1 = plot_full_nav()
    svg2 = plot_wf()
    svg3 = plot_oos()
    svg4 = plot_scenarios()
    print("已生成SVG: ")
    for label, s in [("full_nav", svg1), ("wf", svg2), ("oos", svg3), ("scenarios", svg4)]:
        print("  - {}: {:.0f}KB ({} chars)".format(label, len(s)/1024, len(s)))


if __name__ == "__main__":
    main()
