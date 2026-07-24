# coding=utf-8
"""v9 周期拆解 — 滚动窗口 + 带通滤波方式.

直接用带通滤波提取特定频段, 比 VMD 更直观.

用法:
    python3.11 scripts/v9/v9_cycle_decompose.py
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

from scipy.signal import butter, filtfilt


def load_data():
    data_dir = REPO / "data" / "high_freq_macro"
    y_weekly = pd.read_parquet(data_dir / "v7_10_Y_weekly.parquet")
    hs300 = y_weekly.mean(axis=1).fillna(0)
    nav = (1 + hs300).cumprod()
    log_nav = np.log(nav)
    return hs300, nav, log_nav


def bandpass_filter(signal, low_weeks, high_weeks, fs=1.0, order=3):
    """带通滤波器, 提取特定周期范围的分量."""
    nyq = 0.5 * fs
    low = (1.0 / high_weeks) / nyq
    high = (1.0 / low_weeks) / nyq
    low = max(low, 0.001)
    high = min(high, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def analyze_cycles(log_nav, hs300):
    """用带通滤波拆解周期."""
    print("=" * 70)
    print("沪深300 周期拆解 — 带通滤波法")
    print("=" * 70)
    print(f"  数据: {log_nav.index[0].strftime('%Y-%m-%d')} ~ {log_nav.index[-1].strftime('%Y-%m-%d')}")
    print(f"  周数: {len(log_nav)}")

    # 定义周期频段 (单位: 周)
    cycles = [
        {'name': 'Minor (超短波)', 'typical': '9-12月', 'low': 30, 'high': 60, 'color': '#06b6d4'},
        {'name': 'Kitchin (库存周期)', 'typical': '3-5年', 'low': 120, 'high': 260, 'color': '#3b82f6'},
        {'name': 'Juglar (设备周期)', 'typical': '7-11年', 'low': 350, 'high': 570, 'color': '#10b981'},
        {'name': 'Trend (长期趋势)', 'typical': '>15年', 'low': 600, 'high': 2000, 'color': '#f59e0b'},
    ]

    # 对每个周期做带通滤波
    results = []
    for c in cycles:
        try:
            filtered = bandpass_filter(log_nav.values, c['low'], c['high'])
            series = pd.Series(filtered, index=log_nav.index)
            current = series.iloc[-1]
            amplitude = series.iloc[-52:].std()
            trend = 'up' if series.iloc[-1] > series.iloc[-26] else 'down'
            results.append({**c, 'series': series, 'current': current,
                           'amplitude': amplitude, 'trend': trend})
            print(f"  {c['name']} ({c['typical']}): 振幅={amplitude:.4f}, 当前={current:+.4f}, 趋势={'↑' if trend=='up' else '↓'}")
        except Exception as e:
            print(f"  {c['name']}: 滤波失败 ({e})")

    # 综合判定
    up = sum(1 for r in results if r['trend'] == 'up')
    down = sum(1 for r in results if r['trend'] == 'down')
    if up >= 3:
        verdict = "多周期共振向上 → 看多"
        color = '#22c55e'
    elif down >= 3:
        verdict = "多周期共振向下 → 看空"
        color = '#ef4444'
    else:
        verdict = "周期分化 → 中性/观望"
        color = '#f59e0b'
    print(f"\n  上行: {up}, 下行: {down}")
    print(f"  {verdict}")

    return results, verdict, color


def plot_cycles(log_nav, results, verdict, verdict_color, output_dir):
    """绘制周期拆解图."""
    n = len(results) + 1
    fig, axes = plt.subplots(n, 1, figsize=(16, 3*n), sharex=True)

    ax0 = axes[0]
    ax0.plot(log_nav.index, log_nav.values, color='#1e293b', linewidth=1.5, label='log(NAV)')
    ax0.set_title('沪深300 等权 — log(NAV)', fontsize=12, fontweight='bold')
    ax0.legend(loc='upper left')
    ax0.grid(True, alpha=0.3)

    for i, r in enumerate(results):
        ax = axes[i + 1]
        trend_color = '#22c55e' if r['trend'] == 'up' else '#ef4444'
        trend_cn = '↑上行' if r['trend'] == 'up' else '↓下行'

        ax.plot(r['series'].index, r['series'].values, color=r['color'], linewidth=1.5)
        ax.fill_between(r['series'].index, 0, r['series'].values,
                        where=r['series'].values > 0, alpha=0.3, color=r['color'])
        ax.fill_between(r['series'].index, 0, r['series'].values,
                        where=r['series'].values < 0, alpha=0.3, color='#ef4444')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        ax.set_title(
            f'{r["name"]} ({r["typical"]}) — 振幅={r["amplitude"]:.4f} {trend_cn}',
            fontsize=11, fontweight='bold', color=trend_color
        )
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'v9 周期拆解 — {verdict}', fontsize=14, fontweight='bold',
                 color=verdict_color, y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "cycle_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  {output_dir / 'cycle_decomposition.png'}")


def plot_dashboard(results, output_dir):
    """相位仪表盘."""
    fig, axes = plt.subplots(1, len(results), figsize=(4*len(results), 3.5))
    if len(results) == 1:
        axes = [axes]
    for i, r in enumerate(results):
        ax = axes[i]
        trend_color = '#22c55e' if r['trend'] == 'up' else '#ef4444'
        trend_cn = '上行' if r['trend'] == 'up' else '下行'
        s = r['series']
        last_52 = s.iloc[-52:]
        ax.fill_between(range(len(last_52)), 0, last_52.values,
                        where=last_52.values > 0, alpha=0.4, color=trend_color)
        ax.fill_between(range(len(last_52)), 0, last_52.values,
                        where=last_52.values < 0, alpha=0.4, color='#ef4444')
        ax.plot(range(len(last_52)), last_52.values, color=trend_color, linewidth=2)
        ax.set_title(f'{r["name"]}\n{r["typical"]} | {trend_cn}',
                     fontsize=10, fontweight='bold', color=trend_color)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('各周期当前相位', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / "phase_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {output_dir / 'phase_dashboard.png'}")


def main():
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v9"
    output_dir.mkdir(parents=True, exist_ok=True)

    hs300, nav, log_nav = load_data()
    results, verdict, color = analyze_cycles(log_nav, hs300)

    plot_cycles(log_nav, results, verdict, color, output_dir)
    plot_dashboard(results, output_dir)

    report = []
    report.append("# 沪深300 周期拆解报告")
    report.append(f"> 数据: {log_nav.index[0].strftime('%Y-%m-%d')} ~ {log_nav.index[-1].strftime('%Y-%m-%d')}")
    report.append(f"> 方法: 带通滤波 (Butterworth)")
    report.append("")
    report.append("## 各周期识别")
    report.append("")
    report.append("| 周期 | 典型时长 | 振幅 | 当前值 | 趋势 |")
    report.append("|------|---------|------|--------|------|")
    for r in results:
        trend_cn = '↑上行' if r['trend'] == 'up' else '↓下行'
        report.append(f"| {r['name']} | {r['typical']} | {r['amplitude']:.4f} | {r['current']:+.4f} | {trend_cn} |")
    report.append("")
    report.append(f"## 综合判定")
    report.append(f"**{verdict}**")
    report.append("")
    report.append("![周期分解](cycle_decomposition.png)")
    report.append("![相位仪表盘](phase_dashboard.png)")

    report_path = output_dir / "cycle_decomposition_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"  {report_path}")

    print("\n" + "=" * 70)
    print("完成!")


if __name__ == "__main__":
    main()