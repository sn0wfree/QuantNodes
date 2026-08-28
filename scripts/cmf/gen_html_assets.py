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
import warnings

sys.path.insert(0, '/home/ll/Public/QuantNodes')

warnings.filterwarnings("ignore", message=".*missing from current font.*")

# TTC包含多个字体, 显式绑定Regular (含完整中文字形)
CJK_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CJK_FP = None
if os.path.exists(CJK_PATH):
    try:
        fm.fontManager.addfont(CJK_PATH)
        CJK_FP = fm.FontProperties(fname=CJK_PATH)
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
        plt.rcParams['font.family'] = 'sans-serif'
    except Exception as e:
        print(f"font load warn: {e}")

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


def force_cjk_on_axes(ax):
    """强制所有text/label使用CJK字体."""
    if CJK_FP is None:
        return
    # ax.texts 只包含 ax.text() 创建的, 不包含 annotate 的Annotation对象
    for text in ax.texts:
        text.set_fontproperties(CJK_FP)
    # 遍历figure的所有artist, 找Text/Annotation
    for child in ax.get_children():
        from matplotlib.text import Text, Annotation
        if isinstance(child, (Text, Annotation)):
            child.set_fontproperties(CJK_FP)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(CJK_FP)
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel(), fontproperties=CJK_FP)
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel(), fontproperties=CJK_FP)
    if ax.get_title():
        ax.set_title(ax.get_title(), fontproperties=CJK_FP)
    if ax.get_legend():
        for text in ax.get_legend().get_texts():
            text.set_fontproperties(CJK_FP)


def shade_periods(ax, x_min_dt, x_max_dt, ylim_top=None):
    """三段背景色."""
    if ylim_top is None:
        ylim_top = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 10.0
    segs = [
        (pd.Timestamp('2010-01-01'), pd.Timestamp('2019-12-31'), '#E3F2FD', '样本内测试'),
        (pd.Timestamp('2020-01-01'), pd.Timestamp('2025-12-31'), '#FFF8E1', '滚动验证'),
        (pd.Timestamp('2026-01-01'), x_max_dt, '#E8F5E9', '纯样本外'),
    ]
    for start, end, color, label in segs:
        if start > x_max_dt:
            continue
        end_c = min(end, x_max_dt)
        ax.axvspan(start, end_c, color=color, alpha=0.6, zorder=0)
    # 顶部图外标注 (label + 年份) — 用fig.text绝对位置避免重叠
    if x_max_dt >= pd.Timestamp('2019-06-01'):
        ax.text(pd.Timestamp('2014-12-31'), ylim_top * 0.55, '样本内测试 2010-2019',
                transform=ax.transData,
                ha='center', va='center', fontsize=9, color='#37474F',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8E1',
                          edgecolor='#FFE082', alpha=0.85), zorder=2)
    if x_max_dt >= pd.Timestamp('2022-12-01'):
        ax.text(pd.Timestamp('2022-12-31'), ylim_top * 0.55, '滚动验证 2020-2025',
                transform=ax.transData,
                ha='center', va='center', fontsize=9, color='#37474F',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD',
                          edgecolor='#90CAF9', alpha=0.85), zorder=2)
    if x_max_dt >= pd.Timestamp('2026-04-01'):
        ax.text(pd.Timestamp('2026-04-01'), ylim_top * 0.55, '纯样本外 2026',
                transform=ax.transData,
                ha='center', va='center', fontsize=9, color='#37474F',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                          edgecolor='#A5D6A7', alpha=0.85), zorder=2)


def add_metric_box(ax, metrics_dict, x=0.02, y=0.02):
    """在ax左下角添加指标框."""
    txt = ("L=1.5 CPPI\n"
           "CAGR:   {:.2f}%\n"
           "Sharpe: {:.2f}\n"
           "MaxDD:  {:.2f}%\n"
           "Calmar: {:.2f}\n"
           "Vol:    {:.2f}%\n"
           "Total:  +{:.0f}%").format(
        metrics_dict['cagr'] * 100, metrics_dict['sharpe'],
        metrics_dict['mdd'] * 100, metrics_dict['calmar'],
        metrics_dict['vol'] * 100, metrics_dict['total'] * 100)
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8FAFE',
                      edgecolor='#90CAF9', alpha=0.95))


# ============================================================
# 图1: 全周期净值 (A图) — 完整复刻对比曲线
# ============================================================
def plot_full_nav():
    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=150)
    nav15 = D["nav_15"]
    nav_base = D["nav_base"]
    nav_bh = D["nav_bh"]

    # 计算 ylim_top (提前供 shade_periods 使用)
    ylim_top = max(nav15.max(), nav_bh.max()) * 2.5

    # 三段底色 + 区间标注
    shade_periods(ax, nav15.index[0], nav15.index[-1], ylim_top=ylim_top)

    # 三条曲线 (粗细差异显著, 易于对比)
    ax.semilogy(nav15.index, nav15.values, color='#1565C0', linewidth=2.4,
                label='L=1.5 CPPI (本策略)', zorder=5)
    ax.semilogy(nav_base.index, nav_base.values, color='#2E7D32', linewidth=1.6,
                linestyle='--', label='基线 L=1', zorder=4)
    ax.semilogy(nav_bh.index, nav_bh.values, color='#F57C00', linewidth=2.0,
                label='Buy&Hold', zorder=4)

    # 标记每段曲线终点的具体值
    end_date = nav15.index[-1]
    end_15 = nav15.iloc[-1]
    end_base = nav_base.iloc[-1]
    end_bh = nav_bh.iloc[-1]

    ax.annotate(f'L=1.5: {end_15:.1f}x', xy=(end_date, end_15),
                xytext=(8, 5), textcoords='offset points',
                fontsize=9, color='#1565C0', fontweight='bold')
    ax.annotate(f'L=1: {end_base:.1f}x', xy=(end_date, end_base),
                xytext=(8, -8), textcoords='offset points',
                fontsize=9, color='#2E7D32')
    ax.annotate(f'B&H: {end_bh:.1f}x', xy=(end_date, end_bh),
                xytext=(8, -18), textcoords='offset points',
                fontsize=9, color='#F57C00')

    # 标注关键事件
    # 1. 2013 Taper
    if len(nav_bh) > 0:
        ax.annotate('2013\nTaper紧缩', xy=(pd.Timestamp('2013-06-01'), 1.0),
                    fontsize=8, color='#C62828', ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE',
                              edgecolor='#EF9A9A', alpha=0.85))
        # 2. 2020 COVID
        ax.annotate('2020\nCOVID', xy=(pd.Timestamp('2020-03-15'), 1.2),
                    fontsize=8, color='#C62828', ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE',
                              edgecolor='#EF9A9A', alpha=0.85))
        # 3. 2022 加息
        ax.annotate('2022\n激进加息', xy=(pd.Timestamp('2022-09-01'), 1.8),
                    fontsize=8, color='#C62828', ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE',
                              edgecolor='#EF9A9A', alpha=0.85))
        # 4. 2026 OOS
        ax.annotate('2026\n崩盘', xy=(pd.Timestamp('2026-03-15'), 10),
                    fontsize=8, color='#C62828', ha='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE',
                              edgecolor='#EF9A9A', alpha=0.85))

    # 1x基准参考线 (放右上角避免与左下指标框重叠)
    ax.axhline(y=1.0, color='#90A4AE', linestyle=':', linewidth=0.8, alpha=0.6, zorder=1)
    ax.text(nav15.index[5], ylim_top * 0.5, '基准 1.0x', fontsize=8, color='#607D8B',
            ha='left', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      edgecolor='#B0BEC5', alpha=0.8))

    # 找每段最大回撤位置, 标注
    def find_mdd(nav):
        peak = nav.cummax()
        dd = (nav / peak - 1)
        mdd_idx = dd.idxmin()
        return mdd_idx, dd.min()

    for nav, color, name in [(nav15, '#1565C0', 'L=1.5'), (nav_bh, '#F57C00', 'B&H')]:
        mdd_idx, mdd_val = find_mdd(nav)
        if mdd_val < -0.1:
            ax.scatter([mdd_idx], [nav[mdd_idx]], color=color, s=30, zorder=6,
                       edgecolors='white', linewidths=1)
            ax.annotate(f'{name} MaxDD\n{mdd_val*100:.1f}%',
                        xy=(mdd_idx, nav[mdd_idx]),
                        xytext=(-15, -25), textcoords='offset points',
                        fontsize=7, color=color, ha='right',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor=color, alpha=0.85))

    ax.set_xlim(nav15.index[0], nav15.index[-1])
    ax.set_ylim(0.4, ylim_top)
    ax.set_title('黄金量化策略 vs Buy&Hold 累计净值 (2010-2026)',
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_ylabel('累计净值 (对数尺度)', fontsize=10)
    ax.grid(True, which='both', alpha=0.25)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # 图例: 顶部
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95,
              title='策略曲线', title_fontsize=10)

    # 三套指标框 (左下)
    m15 = D["metrics_15"]; mb = D["metrics_base"]; mbh = D["metrics_bh"]
    add_metric_box(ax, m15)
    ax.text(0.98, 0.02,
            "L=1基线: CAGR={:.2f}%  Sharpe={:.2f}  MaxDD={:.2f}%\nCalmar={:.2f}  Vol={:.2f}%  Total=+{:.0f}%".format(
                mb["cagr"]*100, mb["sharpe"], mb["mdd"]*100, mb["calmar"],
                mb["vol"]*100, mb["total"]*100),
            transform=ax.transAxes, fontsize=8,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F1F8E9',
                      edgecolor='#A5D6A7', alpha=0.95))
    ax.text(0.98, 0.20,
            "Buy&Hold: CAGR={:.2f}%  Sharpe={:.2f}  MaxDD={:.2f}%\nCalmar={:.2f}  Vol={:.2f}%  Total=+{:.0f}%".format(
                mbh["cagr"]*100, mbh["sharpe"], mbh["mdd"]*100, mbh["calmar"],
                mbh["vol"]*100, mbh["total"]*100),
            transform=ax.transAxes, fontsize=8,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0',
                      edgecolor='#FFB74D', alpha=0.95))

    plt.tight_layout()
    force_cjk_on_axes(ax)
    return save_svg_inline(fig)


# ============================================================
# 图2: Walk-Forward 拼接 (B1) — 完整复刻
# ============================================================
def plot_wf():
    fig, ax = plt.subplots(figsize=(8.5, 5), dpi=150)
    wf = D["wf_nav"]
    spans = D["wf_spans"]

    # 折段底色 (15折)
    colors_fold = plt.cm.tab20(np.linspace(0, 1, len(spans)))
    for i, (s, e) in enumerate(spans):
        ax.axvspan(s, e, color=colors_fold[i], alpha=0.08, zorder=0)

    # 拼接曲线
    ax.plot(wf.index, wf.values, color='#1565C0', linewidth=2.4, label='WF拼接净值', zorder=4)
    ax.fill_between(wf.index, 1, wf.values, alpha=0.15, color='#1565C0', zorder=2)

    # 折标注 (每折中心标号)
    for i, (s, e) in enumerate(spans):
        mid = s + (e - s) / 2
        ax.text(mid, wf.max() * 1.05, "F{}".format(i + 1),
                ha='center', va='bottom', fontsize=7, color='#546E7A',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor='#B0BEC5', alpha=0.85), zorder=3)

    # 1x基准线
    ax.axhline(y=1.0, color='#90A4AE', linestyle=':', linewidth=0.8, alpha=0.6, zorder=1)
    ax.text(wf.index[5], 1.05, '基准 1.0x', fontsize=8, color='#607D8B')

    # 终点标注
    end_date = wf.index[-1]
    end_val = wf.iloc[-1]
    ax.annotate(f'终点: {end_val:.2f}x',
                xy=(end_date, end_val),
                xytext=(8, 5), textcoords='offset points',
                fontsize=10, color='#1565C0', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD',
                          edgecolor='#1565C0', alpha=0.95), zorder=5)

    # 找MaxDD点
    peak = wf.cummax()
    dd = (wf / peak - 1)
    mdd_idx = dd.idxmin()
    mdd_val = dd.min()
    if mdd_val < -0.05:
        ax.scatter([mdd_idx], [wf[mdd_idx]], color='#C62828', s=50, zorder=6,
                   edgecolors='white', linewidths=1.5)
        ax.annotate(f'WF MaxDD\n{mdd_val*100:.1f}%',
                    xy=(mdd_idx, wf[mdd_idx]),
                    xytext=(-15, -30), textcoords='offset points',
                    fontsize=8, color='#C62828', ha='right', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='#C62828', alpha=0.9), zorder=6)

    # 指标框 (右上)
    total_ret = (wf.iloc[-1] - 1) * 100
    n_years = (wf.index[-1] - wf.index[0]).days / 365.25
    wf_cagr = ((wf.iloc[-1]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    ax.text(0.98, 0.98,
            "WF拼接表现\n"
            "CAGR:   +{:.2f}%\n"
            "MaxDD:  {:.2f}%\n"
            "Calmar: {:.2f}\n"
            "区间累计: +{:.0f}%\n"
            "测试折数: {}".format(
                wf_cagr, mdd_val * 100,
                wf_cagr / abs(mdd_val * 100) if mdd_val != 0 else 0,
                total_ret, len(spans)),
            transform=ax.transAxes, fontsize=9,
            ha='right', va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD',
                      edgecolor='#90CAF9', alpha=0.95), zorder=5)

    # 关键事件标注 (与A图一致)
    if len(wf) > 0:
        if wf.index[0] <= pd.Timestamp('2020-03-15') <= wf.index[-1]:
            ax.annotate('2020 COVID', xy=(pd.Timestamp('2020-03-15'), wf[wf.index <= '2020-03-15'].iloc[-1] if len(wf[wf.index <= '2020-03-15']) > 0 else 1.2),
                        xytext=(0, 25), textcoords='offset points',
                        fontsize=7, color='#C62828', ha='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE',
                                  edgecolor='#EF9A9A', alpha=0.85),
                        arrowprops=dict(arrowstyle='->', color='#C62828', alpha=0.6))
        if wf.index[0] <= pd.Timestamp('2022-09-01') <= wf.index[-1]:
            ax.annotate('2022 加息', xy=(pd.Timestamp('2022-09-01'), 2.5),
                        xytext=(0, 20), textcoords='offset points',
                        fontsize=7, color='#C62828', ha='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE',
                                  edgecolor='#EF9A9A', alpha=0.85),
                        arrowprops=dict(arrowstyle='->', color='#C62828', alpha=0.6))

    ax.set_title('B1. Walk-Forward 滚动验证 ({}折测试段拼接)'.format(len(spans)),
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_ylabel('累计净值', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)

    plt.tight_layout()
    force_cjk_on_axes(ax)
    return save_svg_inline(fig)


# ============================================================
# 图3: 2026 OOS (B2) — 完整复刻
# ============================================================
def plot_oos():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.5), dpi=150,
                                    gridspec_kw={'height_ratios': [2, 1]},
                                    sharex=True)
    nav26 = D["nav26"]
    close26 = D["close26"]
    pos26 = D["pos26"]

    # 策略净值
    ax1.plot(nav26.index, nav26.values, color='#1565C0', linewidth=2.4,
             label='策略累计净值', zorder=4)
    ax1.fill_between(nav26.index, 1, nav26.values,
                     where=nav26.values >= 1, color='#1565C0', alpha=0.15, zorder=2)
    ax1.set_ylabel('策略净值', color='#1565C0', fontsize=10)
    ax1.tick_params(axis='y', labelcolor='#1565C0')

    # 1x基准线
    ax1.axhline(y=1.0, color='#90A4AE', linestyle=':', linewidth=0.8, alpha=0.6, zorder=1)
    ax1.text(nav26.index[5], 1.02, '基准 1.0x', fontsize=8, color='#607D8B')

    # 标注关键事件
    if len(nav26) > 0:
        # 崩盘日 (2026-03-15附近找到最低点)
        crash_date = pd.Timestamp('2026-03-15')
        if crash_date in nav26.index:
            crash_nav = nav26[crash_date]
        else:
            # 找最近的日期
            closest = nav26.index[nav26.index >= crash_date]
            if len(closest) > 0:
                crash_date = closest[0]
                crash_nav = nav26[crash_date]
            else:
                crash_date = nav26.index[0]
                crash_nav = nav26.iloc[0]
        ax1.scatter([crash_date], [crash_nav], color='#C62828', s=50, zorder=6,
                   edgecolors='white', linewidths=1.5)
        ax1.annotate(f'崩盘日\n{crash_date.strftime("%Y-%m-%d")}',
                    xy=(crash_date, crash_nav),
                    xytext=(-20, -35), textcoords='offset points',
                    fontsize=8, color='#C62828', ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE',
                              edgecolor='#EF9A9A', alpha=0.9),
                    arrowprops=dict(arrowstyle='->', color='#C62828', alpha=0.7), zorder=6)

        # 最高点标注
        peak_idx = nav26.values.argmax()
        peak_date = nav26.index[peak_idx]
        peak_val = nav26.values[peak_idx]
        ax1.scatter([peak_date], [peak_val], color='#2E7D32', s=50, zorder=6,
                   edgecolors='white', linewidths=1.5)
        ax1.annotate(f'峰值: {peak_val:.2f}x\n{peak_date.strftime("%Y-%m-%d")}',
                    xy=(peak_date, peak_val),
                    xytext=(8, 15), textcoords='offset points',
                    fontsize=8, color='#2E7D32', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                              edgecolor='#A5D6A7', alpha=0.9), zorder=6)

        # 终点标注
        end_date = nav26.index[-1]
        end_val = nav26.iloc[-1]
        ax1.annotate(f'YTD: {end_val-1:+.1%}\n{end_date.strftime("%Y-%m-%d")}',
                    xy=(end_date, end_val),
                    xytext=(8, 5), textcoords='offset points',
                    fontsize=9, color='#1565C0', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#E3F2FD',
                              edgecolor='#1565C0', alpha=0.95), zorder=6)

    # 指标框 (右上)
    total_ret = (nav26.iloc[-1] - 1) * 100
    n_days = len(nav26)
    n_years = n_days / 252
    cagr = ((nav26.iloc[-1]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0
    ax1.text(0.98, 0.98,
            "2026 OOS表现\n"
            "YTD收益: {:+.1f}%\n"
            "测试天数: {}天\n"
            "区间CAGR: {:.1f}%".format(total_ret, n_days, cagr),
            transform=ax1.transAxes, fontsize=9,
            ha='right', va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E3F2FD',
                      edgecolor='#90CAF9', alpha=0.95), zorder=5)

    ax1.set_title('B2. 2026 纯样本外验证 (YTD {:+.1f}%)'.format(total_ret),
                 fontsize=13, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.95)

    # 仓位管理 (下图)
    ax2.set_facecolor('#FAFAFA')
    ax2.step(pos26.index, pos26.values, where='post', color='#F57C00',
             linewidth=2.0, label='策略仓位', zorder=4)
    ax2.fill_between(pos26.index, 0, pos26.values, step='post',
                     color='#F57C00', alpha=0.2, zorder=2)
    ax2.set_ylabel('仓位', fontsize=10)
    ax2.set_ylim(0, 1.8)
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    ax2.text(pos26.index[5], 1.05, '基准100%', fontsize=8, color='gray')

    # 标注仓位大幅变动
    if len(pos26) > 1:
        pos_diff = pos26.diff().abs()
        # 找到仓位变动超过0.3的点
        big_changes = pos_diff[pos_diff > 0.3]
        if len(big_changes) > 0:
            for idx in big_changes.index[:3]:  # 最多标注3个
                val = pos26[idx]
                ax2.scatter([idx], [val], color='#C62828', s=40, zorder=6,
                           edgecolors='white', linewidths=1)
                ax2.annotate(f'{idx.strftime("%m-%d")}\n仓位{val:.0%}',
                            xy=(idx, val),
                            xytext=(0, 15), textcoords='offset points',
                            fontsize=7, color='#C62828', ha='center',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE',
                                      edgecolor='#EF9A9A', alpha=0.85), zorder=6)

    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.95)

    plt.tight_layout()
    force_cjk_on_axes(ax1)
    force_cjk_on_axes(ax2)
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

        # 标注金价峰值和谷底
        peak_date = c.index[peak_idx]
        peak_val = c.values[peak_idx]
        ax.scatter([peak_date], [peak_val], color='#C62828', s=40, zorder=6,
                   edgecolors='white', linewidths=1)
        ax.annotate(f'峰值\n{peak_val:.1f}',
                    xy=(peak_date, peak_val),
                    xytext=(8, 10), textcoords='offset points',
                    fontsize=7, color='#C62828', ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFEBEE',
                              edgecolor='#EF9A9A', alpha=0.85), zorder=6)

        # 找到金价最低点
        trough_idx = c.values.argmin()
        trough_date = c.index[trough_idx]
        trough_val = c.values[trough_idx]
        ax.scatter([trough_date], [trough_val], color='#2E7D32', s=40, zorder=6,
                   edgecolors='white', linewidths=1)
        ax.annotate(f'谷底\n{trough_val:.1f}',
                    xy=(trough_date, trough_val),
                    xytext=(8, -15), textcoords='offset points',
                    fontsize=7, color='#2E7D32', ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9',
                              edgecolor='#A5D6A7', alpha=0.85), zorder=6)

        # 标注仓位最大变动
        if len(p) > 1:
            pos_diff = p.diff().abs()
            big_changes = pos_diff[pos_diff > 0.3]
            if len(big_changes) > 0:
                for idx in big_changes.index[:2]:  # 最多标注2个
                    val = p[idx]
                    ax2.scatter([idx], [val], color='#1565C0', s=30, zorder=6,
                               edgecolors='white', linewidths=1)
                    ax2.annotate(f'{idx.strftime("%m-%d")}\n仓位{val:.0%}',
                                xy=(idx, val),
                                xytext=(0, 10), textcoords='offset points',
                                fontsize=7, color='#1565C0', ha='center',
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='#E3F2FD',
                                          edgecolor='#90CAF9', alpha=0.85), zorder=6)

        ax.grid(True, alpha=0.25)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)

    fig.suptitle('D. 黄金历史大跌场景下的仓位管理',
                 fontsize=13, fontweight='bold', y=1.0,
                 fontproperties=CJK_FP if CJK_FP else None)
    plt.tight_layout()
    for a in axes:
        force_cjk_on_axes(a)
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
