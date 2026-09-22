#!/usr/bin/env python3
# coding=utf-8
"""共振门控业绩曲线 (净值化): 从已导出 NAV 重绘.

净值口径: 各配置统一起点 = 1.0000 (2019-01-02), 线性轴.
输出: docs/allweather_resonance_nav.png + 年末净值表.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'SimHei', 'Droid Sans Fallback']
rcParams['axes.unicode_minus'] = False
ANN = 252


def main():
    df = pd.read_csv('docs/allweather_resonance_performance.csv', index_col=0,
                     parse_dates=True).sort_index()
    # 净值化: 每列起点归一到 1.0000
    nav = df.div(df.iloc[0], axis=1)
    colors = ['#888888', '#d62728', '#ff7f0e', '#2ca02c']

    fig, axes = plt.subplots(2, 2, figsize=(16, 9.5))
    # 1 净值曲线 (线性)
    ax = axes[0, 0]
    for (lbl, s), c in zip(nav.items(), colors):
        ax.plot(s.index, s.values, label=lbl, color=c, lw=1.7)
    ax.set_title('净值曲线 (起点 = 1.0000, 线性)')
    ax.set_ylabel('单位净值')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # 2 回撤
    ax = axes[0, 1]
    for (lbl, s), c in zip(nav.items(), colors):
        ax.plot(s.index, (s / s.cummax() - 1) * 100, label=lbl, color=c, lw=1.2)
    ax.set_title('回撤 (%)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # 3 滚动 252 日夏普
    ax = axes[1, 0]
    for (lbl, s), c in zip(nav.items(), colors):
        r = s.pct_change()
        rs = (r.rolling(252).mean() * ANN) / (r.rolling(252).std() * np.sqrt(ANN))
        ax.plot(rs.index, rs, label=lbl, color=c, lw=1.2)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_title('滚动 252 日夏普')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # 4 年末净值 (柱)
    ax = axes[1, 1]
    years = sorted(set(nav.index.year))
    width = 0.2
    x = np.arange(len(years))
    ye = {}
    for i, ((lbl, s), c) in enumerate(zip(nav.items(), colors)):
        vals = []
        for y in years:
            sy = s[s.index.year == y]
            vals.append(sy.iloc[-1] if len(sy) > 5 else np.nan)
        ye[lbl] = vals
        ax.bar(x + (i - 1.5) * width, vals, width, label=lbl, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.axhline(1.0, color='k', lw=0.6, ls='--')
    ax.set_title('年末单位净值')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('docs/allweather_resonance_nav.png', dpi=130)
    print('已保存: docs/allweather_resonance_nav.png')

    # 年末净值表
    tbl = pd.DataFrame(ye, index=years).T
    tbl['期末'] = [nav[c].iloc[-1] for c in nav.columns]
    tbl.index.name = '配置'
    print('\n年末单位净值 (起点 1.0000):')
    print(tbl.round(4).to_string())
    tbl.to_csv('docs/allweather_resonance_nav_yearend.csv', float_format='%.4f')
    print('\n已保存: docs/allweather_resonance_nav_yearend.csv')

    # 完整净值序列导出
    nav.to_csv('docs/allweather_resonance_nav.csv', float_format='%.6f')
    print('已保存: docs/allweather_resonance_nav.csv (完整净值序列)')


if __name__ == '__main__':
    main()
