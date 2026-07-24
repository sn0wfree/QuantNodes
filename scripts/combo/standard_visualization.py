"""3 张关键可视化 (PNG).

1. 标准 sharpe_heatmap: 9 区间 × 21 策略 (红蓝色彩)
2. nav_curves_4strats: 4 策略 NAV 曲线 18-26
3. scenario_sharpe_bars: 8 场景 Top5 策略 柱状
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

REPO = Path('.')
OUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "combo"
FIGS_DIR = OUT_DIR / "figs"
FIGS_DIR.mkdir(exist_ok=True)

# Use English-safe font
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
mpl.rcParams['axes.unicode_minus'] = False

# Per figure size config
plt.rcParams['figure.dpi'] = 110
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# 中文策略名 → 短名映射 (用于图表标签)
SHORT_NAMES = {
    'v0.0 baseline': 'v0.0',
    'v0.1 +VT': 'v0.1',
    'v0.2 +TF': 'v0.2',
    'v1.0 locked': 'v1.0',
    'v3 (52 池)': 'v3',
    'v4 style': 'v4-s',
    'v4 factor': 'v4-f',
    'v5 量价': 'v5',
    'v5.1 量价 (逆波动)': 'v5.1',
    'v7.10 TV-PR (标准化+CV)': 'v7.10',
    'v6 全风控': 'v6',
    'v8 method_b (有未来)': 'v8-mB⚠',
    'v8 prob 3state 10bp': 'v8-3s',
    'v8 prob 2state 10bp': 'v8-2s',
    'v8 per-asset 5bp': 'v8-pa',
    'v9 银河方案-动态仓位': 'v9-dyn',
    'v9 银河因子配置': 'v9-gxy',
    'v9 等权基准': 'v9-eq',
    '⭐ v8+v9 macro 5bp (NEW)': '⭐v8+v9',
    'v8+v9 macro 10bp (NEW)': 'v8+v9-10',
    '沪深300 (benchmark)': 'CSI300',
}

PERIODS_ORDER = ['Full Sample', 'OOS 22-26', '2018 贸易战大跌', '2019 春燥行情',
                 '2020 疫情牛', '2021 白马崩盘', '2022 慢熊', '2023 慢熊延续', '2024 政策牛']


def plot_sharpe_heatmap():
    """21 策略 × 9 区间 Sharpe 热图."""
    df = pd.read_csv(OUT_DIR / "standard_comparison_wide.csv")

    strategies = df['Strategy'].tolist()
    short_lbls = [SHORT_NAMES.get(s, s[:10]) for s in strategies]

    matrix = np.zeros((len(PERIODS_ORDER), len(strategies)))
    for i, p in enumerate(PERIODS_ORDER):
        col = f"{p}_Sharpe"
        if col in df.columns:
            matrix[i] = df[col].fillna(0).values

    # 红蓝: 负蓝, 正红. clip 到 [-2, 3]
    vmax, vmin = 3.0, -2.0
    fig, ax = plt.subplots(figsize=(18, 7))
    im = ax.imshow(matrix, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(short_lbls, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(PERIODS_ORDER)))
    ax.set_yticklabels(PERIODS_ORDER, fontsize=10)
    ax.set_title('21 Strategies × 9 Periods: Sharpe Heatmap (Red = Positive, Blue = Negative)',
                 fontsize=13, pad=12)

    # Add text annotations
    for i in range(len(PERIODS_ORDER)):
        for j in range(len(strategies)):
            v = matrix[i, j]
            color = 'white' if abs(v) > 1.5 else 'black'
            text = f'{v:+.2f}' if abs(v) >= 0.005 else '·'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=7)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label('Sharpe Ratio', fontsize=10)

    # Highlight NEW strategy
    new_idx = strategies.index('⭐ v8+v9 macro 5bp (NEW)')
    ax.add_patch(plt.Rectangle((new_idx - 0.5, -0.5), 1, len(PERIODS_ORDER),
                                fill=False, edgecolor='gold', linewidth=2.5, zorder=10))

    plt.tight_layout()
    out_path = FIGS_DIR / "standard_sharpe_heatmap.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → {out_path}")


def plot_nav_curves_4strats():
    """4 关键策略 NAV 曲线 2018-2026."""
    navs = {}
    navs['v7.10 TV-PR (5bp)'] = pd.read_parquet(OUT_DIR / "unified_v1v5_navs_calA.parquet")['v7.10 TV-PR (标准化+CV)']
    navs['v1.0 locked'] = pd.read_parquet(OUT_DIR / "unified_v1v5_navs_calA.parquet")['v1.0 locked']
    navs['v8 per-asset 5bp'] = pd.read_parquet(OUT_DIR / "v8_per_asset_C1_5bp.parquet").iloc[:, 0] 
    navs['⭐ v8+v9 macro 5bp'] = pd.read_parquet(OUT_DIR / "v9_macro_best_C5.parquet").iloc[:, 0]

    # Benchmark
    bench_price = pd.read_parquet('data/high_freq_macro/v9_benchmark_沪深300.parquet')['沪深300指数']
    bench_ret = bench_price.pct_change()
    bench_nav = (1 + bench_ret).cumprod()
    bench_nav.iloc[0] = 1.0

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = {'v1.0 locked': 'tab:gray', 'v7.10 TV-PR (5bp)': 'tab:green',
              'v8 per-asset 5bp': 'tab:blue', '⭐ v8+v9 macro 5bp': 'tab:red',
              '沪深300': 'lightgray'}

    for name in ['v1.0 locked', 'v7.10 TV-PR (5bp)', 'v8 per-asset 5bp', '⭐ v8+v9 macro 5bp']:
        seg = navs[name].loc['2018-01-03':'2026-05-29'].dropna()
        n_years = (seg.index[-1] - seg.index[0]).days / 365.25
        ann_ret = (seg.iloc[-1] / seg.iloc[0]) ** (1 / n_years) - 1
        rets = seg.pct_change().dropna()
        vol = rets.std() * np.sqrt(252)
        sharpe = ann_ret / vol
        ax.plot(seg.index, seg.values / seg.iloc[0], label=f'{name} (Sharpe={sharpe:.2f}, AR={ann_ret:.1%})',
                color=colors[name], linewidth=1.8, alpha=0.95)

    # Benchmark
    bench_seg = bench_nav.loc['2018-01-03':'2026-05-29'].dropna()
    ax.plot(bench_seg.index, bench_seg.values, label='沪深300', color='lightgray', linewidth=1.0, alpha=0.6, linestyle='--')

    # Highlight key events
    events = [
        ('2022-01-01', '2022 慢熊开始', 'gray'),
        ('2022-10-31', '2022 熊底', 'gray'),
        ('2024-09-24', '924 政策红利', 'red'),
        ('2024-10-09', '924 后回吐', 'red'),
        ('2025-01-01', '2025 慢牛开始', 'green'),
    ]
    for d, label, c in events:
        if d in pd.to_datetime(pd.Series([d])).iloc[0].strftime('%Y-%m-%d'):
            dd = pd.Timestamp(d)
            ax.axvline(dd, color=c, linestyle=':', alpha=0.4, linewidth=1)
            ax.annotate(label, xy=(dd, 1.0), xytext=(dd, 1.05),
                        fontsize=8, color=c, ha='left', rotation=0)

    ax.set_title('4 Key Strategies NAV Curves (2018-01 ~ 2026-05, Rebased to 1.0)', fontsize=13)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative NAV (log scale suggested)')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIGS_DIR / "nav_curves_4strats.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  → {out_path}")


def plot_scenario_sharpe_bars():
    """7 场景下 Top5 策略 Sharpe 柱状图."""
    df = pd.read_csv(OUT_DIR / "standard_comparison_wide.csv")

    # 排除 OOS 22-26 (它是完整区间不是子场景)
    scenarios = ['2018 贸易战大跌', '2019 春燥行情', '2020 疫情牛', '2021 白马崩盘',
                '2022 慢熊', '2023 慢熊延续', '2024 政策牛']

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    axes = axes.flatten()

    strategies = df['Strategy'].tolist()
    colors_map = {
        '⭐ v8+v9 macro 5bp (NEW)': 'tab:red',
        'v8 per-asset 5bp': 'tab:blue',
        'v7.10 TV-PR (标准化+CV)': 'tab:green',
        'v1.0 locked': 'tab:gray',
        'v8 method_b (有未来)': 'tab:purple',
        'v9 银河方案-动态仓位': 'tab:orange',
        '沪深300 (benchmark)': 'lightgray',
    }

    for i, sc in enumerate(scenarios):
        ax = axes[i]
        col = f"{sc}_Sharpe"
        if col not in df.columns:
            ax.set_title(f'{sc}\n(no data)')
            continue
        sub = df[['Strategy', col]].dropna().sort_values(col, ascending=True)
        # 取 Top 5 + Bottom 5 (如果超过 10)
        if len(sub) > 10:
            show = pd.concat([sub.head(5), sub.tail(5)])
        else:
            show = sub
        labels = [SHORT_NAMES.get(s, s[:12]) for s in show['Strategy']]
        bar_colors = [colors_map.get(s, 'lightblue') for s in show['Strategy']]

        bars = ax.barh(labels, show[col].values, color=bar_colors, alpha=0.85)
        ax.set_title(sc, fontsize=10)
        ax.axvline(0, color='black', linewidth=0.6)
        ax.grid(True, axis='x', alpha=0.3)
        # Annotations
        for j, (label, value) in enumerate(zip(labels, show[col].values)):
            ax.text(value, j, f'{value:+.2f}', va='center', fontsize=7,
                    ha='left' if value >= 0 else 'right')

    axes[7].axis('off')
    axes[7].text(0.5, 0.5,
                  'RED = ⭐v8+v9 macro\nBLUE = v8 per-asset\nGREEN = v7.10\nGRAY = v1.0 / 沪深300',
                  ha='center', va='center', fontsize=11, transform=axes[7].transAxes)

    plt.suptitle('21 Strategies × 7 Scenarios: Top/Bottom Sharpe Comparison',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = FIGS_DIR / "scenario_sharpe_bars.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {out_path}")


def main():
    print("=== 生成 3 张关键可视化 ===")

    print("\n[1/3] Sharpe heatmap...")
    plot_sharpe_heatmap()

    print("\n[2/3] NAV curves for 4 key strategies...")
    plot_nav_curves_4strats()

    print("\n[3/3] Scenario Sharpe bars...")
    plot_scenario_sharpe_bars()

    print(f"\n✅ 3 张 PNG 已保存到 {FIGS_DIR}/")


if __name__ == "__main__":
    main()
