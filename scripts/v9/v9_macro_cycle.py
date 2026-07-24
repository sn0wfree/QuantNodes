# coding=utf-8
"""v9 宏观周期拆解 — 用 GDP/CPI/PPI 代理指标.

数据: v7_6_X_macro_weekly.parquet (2008-2026, 941 周)

用法:
    python3.11 scripts/v9/v9_macro_cycle.py
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
from matplotlib.gridspec import GridSpec
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

from scipy.signal import butter, filtfilt


def load_macro_data():
    data_dir = REPO / "data" / "high_freq_macro"
    df = pd.read_parquet(data_dir / "v7_6_X_macro_weekly.parquet")
    return df


def bandpass_filter(signal, low_weeks, high_weeks, fs=1.0, order=3):
    nyq = 0.5 * fs
    low = (1.0 / high_weeks) / nyq
    high = (1.0 / low_weeks) / nyq
    low = max(low, 0.001)
    high = min(high, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)


def get_phase(series, window=26):
    if len(series.dropna()) < window:
        return 'unknown'
    recent = series.dropna().iloc[-window:]
    slope = np.polyfit(np.arange(len(recent)), recent.values, 1)[0]
    std = recent.std()
    threshold = std / len(recent) * 0.05 if std > 0 else 1e-10
    if slope > threshold:
        return 'up'
    elif slope < -threshold:
        return 'down'
    return 'flat'


def analyze_macro_cycles(df):
    """对每个宏观指标做带通滤波分析."""
    print("=" * 70)
    print("宏观周期拆解分析 (2008-2026)")
    print("=" * 70)
    print(f"  数据范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"  总周数: {len(df)}")

    # 宏观指标及其经济含义
    indicators = {
        '宏观增长因子': {'name': '经济增长', 'unit': 'GDP代理', 'color': '#3b82f6'},
        '宏观通胀因子_生活端': {'name': 'CPI通胀', 'unit': '生活端', 'color': '#ef4444'},
        '宏观通胀因子_生产端': {'name': 'PPI通胀', 'unit': '生产端', 'color': '#f59e0b'},
        '无风险收益率': {'name': '无风险利率', 'unit': '利率', 'color': '#10b981'},
        '信用利差因子': {'name': '信用利差', 'unit': '信用', 'color': '#8b5cf6'},
        'real_rate': {'name': '实际利率', 'unit': '实际利率', 'color': '#06b6d4'},
    }

    # 周期频段定义 (单位: 周)
    cycle_defs = [
        ('Minor', '超短波', 13, 52, '9-12月'),
        ('Kitchin', '库存周期', 100, 260, '3-5年'),
        ('Juglar', '设备周期', 260, 520, '5-10年'),
    ]

    results = {}
    for col, meta in indicators.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) < 100:
            continue

        # Z-score 标准化
        z = (series - series.mean()) / (series.std() + 1e-10)

        # HP 滤波分离趋势
        from statsmodels.tsa.filters.hp_filter import hpfilter
        try:
            cycle_hp, trend_hp = hpfilter(z, lamb=100)
        except:
            cycle_hp = z - z.rolling(52).mean()
            trend_hp = z.rolling(52).mean()

        # 带通滤波
        cycles = {}
        for cname, cn_name, low, high, typical in cycle_defs:
            try:
                filtered = bandpass_filter(cycle_hp.values, low, high)
                fs = pd.Series(filtered, index=cycle_hp.index)
                current = fs.iloc[-1]
                amp = fs.iloc[-52:].std()
                phase = get_phase(fs, window=26)
                cycles[cname] = {
                    'name': cn_name, 'typical': typical,
                    'series': fs, 'current': current,
                    'amplitude': amp, 'phase': phase,
                }
            except Exception as e:
                pass

        results[col] = {'meta': meta, 'cycles': cycles, 'z': z, 'trend': trend_hp}

    return results, indicators, cycle_defs


def print_results(results, indicators, cycle_defs):
    """打印分析结果."""
    print("\n[各宏观指标周期拆解]")
    for col, res in results.items():
        meta = res['meta']
        z_current = res['z'].iloc[-1]
        trend_phase = get_phase(res['trend'], window=52)
        trend_cn = '↑上行' if trend_phase == 'up' else ('↓下行' if trend_phase == 'down' else '→平')

        print(f"\n  {meta['name']} ({meta['unit']}):")
        print(f"    Z-score 当前: {z_current:+.2f} ({'高于均值' if z_current > 0 else '低于均值'}), 长期趋势: {trend_cn}")

        for cname, cdata in res['cycles'].items():
            phase_cn = '↑' if cdata['phase'] == 'up' else ('↓' if cdata['phase'] == 'down' else '→')
            print(f"    {cdata['name']}({cdata['typical']}): "
                  f"振幅={cdata['amplitude']:.4f}, 当前={cdata['current']:+.4f}, 趋势={phase_cn}")

    # 美林时钟判定
    print("\n[美林时钟判定]")
    growth = results.get('宏观增长因子', {})
    cpi = results.get('宏观通胀因子_生活端', {})
    g_trend = get_phase(growth.get('trend', pd.Series()), window=52) if growth else 'unknown'
    c_trend = get_phase(cpi.get('trend', pd.Series()), window=52) if cpi else 'unknown'

    g_cn = '上行' if g_trend == 'up' else ('下行' if g_trend == 'down' else '平')
    c_cn = '上行' if c_trend == 'up' else ('下行' if c_trend == 'down' else '平')

    if g_trend == 'up' and c_trend == 'down':
        phase = 'Recovery (复苏)'
        emoji = '🟢'
        alloc = '股票 > 债券'
    elif g_trend == 'up' and c_trend == 'up':
        phase = 'Overheat (过热)'
        emoji = '🟡'
        alloc = '商品 > 股票'
    elif g_trend == 'down' and c_trend == 'up':
        phase = 'Stagflation (滞胀)'
        emoji = '🔴'
        alloc = '现金 > 商品'
    else:
        phase = 'Recession (衰退)'
        emoji = '🔵'
        alloc = '债券 > 股票'

    print(f"  经济增长趋势: {g_cn}")
    print(f"  通胀趋势: {c_cn}")
    print(f"  {emoji} 当前阶段: {phase}")
    print(f"  推荐配置: {alloc}")

    # 综合评分
    print("\n[综合评分]")
    total_score = 0
    n_indicators = 0
    for col, res in results.items():
        for cname, cdata in res['cycles'].items():
            if cdata['phase'] == 'up':
                total_score += 10
            elif cdata['phase'] == 'down':
                total_score -= 10
            n_indicators += 1
    print(f"  周期趋势分: {total_score:+d} (基于 {n_indicators} 个指标-周期组合)")
    if total_score > 20:
        print("  综合: 多周期共振向上 → 看多")
    elif total_score < -20:
        print("  综合: 多周期共振向下 → 看空")
    else:
        print("  综合: 周期分化 → 中性/观望")

    return results


def plot_macro_cycles(results, cycle_defs, output_dir):
    """绘制宏观周期图."""
    n_indicators = len(results)
    n_cycles = len(cycle_defs)

    fig, axes = plt.subplots(n_indicators + 1, 1, figsize=(16, 3*(n_indicators+1)), sharex=True)

    # 总览: Z-score
    ax0 = axes[0]
    for col, res in results.items():
        meta = res['meta']
        ax0.plot(res['z'].index, res['z'].values, color=meta['color'], linewidth=1.2, label=meta['name'], alpha=0.7)
    ax0.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax0.set_title('宏观指标 Z-score (标准化值)', fontsize=12, fontweight='bold')
    ax0.legend(loc='upper left', ncol=3, fontsize=9)
    ax0.grid(True, alpha=0.3)

    # 每个指标的周期拆解
    for i, (col, res) in enumerate(results.items()):
        meta = res['meta']
        ax = axes[i + 1]

        # 绘制 HP 趋势
        ax.plot(res['trend'].index, res['trend'].values, color=meta['color'],
                linewidth=2, alpha=0.7, label=f'{meta["name"]} 趋势')

        # 绘制各周期分量
        for cname, cdata in res['cycles'].items():
            phase_color = '#22c55e' if cdata['phase'] == 'up' else '#ef4444'
            ax.plot(cdata['series'].index, cdata['series'].values,
                    color=phase_color, linewidth=0.8, alpha=0.5)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.set_title(f'{meta["name"]}', fontsize=11, fontweight='bold', color=meta['color'])
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('v9 宏观周期拆解 (2008-2026)', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / "macro_cycles.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  {output_dir / 'macro_cycles.png'}")


def plot_phase_dashboard(results, cycle_defs, output_dir):
    """相位仪表盘."""
    n_cols = len(results)
    fig, axes = plt.subplots(2, (n_cols + 1) // 2, figsize=(4 * ((n_cols + 1) // 2), 7))
    axes = axes.flatten()

    for i, (col, res) in enumerate(results.items()):
        if i >= len(axes):
            break
        ax = axes[i]
        meta = res['meta']

        # 绘制最近 104 周的趋势
        trend = res['trend'].iloc[-104:]
        up = trend.diff().mean() > 0
        trend_color = '#22c55e' if up else '#ef4444'
        trend_cn = '上行' if up else '下行'

        ax.plot(range(len(trend)), trend.values, color=trend_color, linewidth=2)
        ax.fill_between(range(len(trend)), 0, trend.values,
                        where=trend.values > 0, alpha=0.3, color=trend_color)
        ax.fill_between(range(len(trend)), 0, trend.values,
                        where=trend.values < 0, alpha=0.3, color='#ef4444')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f'{meta["name"]}\n{trend_cn}', fontsize=10, fontweight='bold', color=trend_color)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('宏观指标长期趋势', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / "macro_phase_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  {output_dir / 'macro_phase_dashboard.png'}")


def main():
    output_dir = REPO / "reports" / "momentum_etf_rotation" / "v9"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_macro_data()
    results, indicators, cycle_defs = analyze_macro_cycles(df)
    results = print_results(results, indicators, cycle_defs)

    plot_macro_cycles(results, cycle_defs, output_dir)
    plot_phase_dashboard(results, cycle_defs, output_dir)

    print("\n" + "=" * 70)
    print("完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()