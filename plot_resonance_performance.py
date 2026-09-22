#!/usr/bin/env python3
# coding=utf-8
"""共振门控业绩曲线: NAV / 回撤 / 滚动夏普 / 分年收益."""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from QuantNodes.strategy.allweather.d_pipeline import run_d_sleeve
from QuantNodes.strategy.allweather.volume_loader import load_volume_oi_panel
from QuantNodes.strategy.allweather.class_erc import ASSET_CLASSES

rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'SimHei', 'Droid Sans Fallback']
rcParams['axes.unicode_minus'] = False
ANN = 252

NEW = ('etf_510500', 'etf_588000', 'etf_159920', 'etf_513520',
       'etf_511260', 'etf_159985')


def load():
    panel = pd.read_parquet('data/allweather/panel_v5.parquet').sort_index()
    panel.index = pd.to_datetime(panel.index)
    v7 = pd.read_parquet('data/high_freq_macro/v7_index_v2.parquet')
    cash = v7['中债1-3年国债财富指数'].dropna()
    cash.index = pd.to_datetime(cash.index)
    cash_ret = cash.pct_change(fill_method=None)
    csi = v7['中证1000'].dropna()
    csi.index = pd.to_datetime(csi.index)
    panel['csi_1000'] = csi.reindex(panel.index).ffill()
    ref41 = [c for c in panel.columns if c != 'csi_1000' and c not in NEW]
    return panel.loc[:'2026-06-30'], ref41, cash_ret


def run(panel, cols, bond, cash_ret, vt, ml, **kw):
    sub = panel[cols]
    cls = {k: [c for c in v if c in cols] for k, v in ASSET_CLASSES.items()}
    for c, k in [('etf_510500', 'equity_etf_dom'), ('etf_588000', 'equity_etf_dom'),
                 ('etf_159920', 'equity_etf_off'), ('etf_513520', 'equity_etf_off'),
                 ('etf_511260', 'bond_rate'), ('etf_159985', 'agri_oilseed'),
                 ('csi_1000', 'equity_etf_dom')]:
        if c in cols and c not in [x for v in cls.values() for x in v]:
            cls.setdefault(k, []).append(c)
    cls = {k: v for k, v in cls.items() if v}
    vol, oi = load_volume_oi_panel(sub, cache_path=None)
    res = run_d_sleeve(sub, bond_cap=0.40, window=252, rebalance_freq='W-FRI',
                       cov_estimator='lw', neg_trend_mode='trend',
                       neg_trend_preset='balanced', include_rapid=True,
                       momentum_beta=0.20, volume=vol, oi=oi, asset_classes=cls,
                       cash_ret=cash_ret, target_vol=vt, max_leverage=ml,
                       financing_rate=0.02, bond_cols=bond, **kw)
    nav = res.nav.dropna()
    return nav[nav.index >= '2019-01-01']


def main():
    panel, ref41, cash_ret = load()
    p2b = ['fut_t', 'fut_tf', 'fut_ts', 'etf_511260',
           'etf_510300', 'etf_510500', 'etf_588000', 'etf_159915', 'csi_1000',
           'etf_513100', 'etf_513500', 'etf_513520', 'etf_159920',
           'fut_au', 'fut_cu', 'fut_sc', 'fut_m', 'etf_159985']
    BC = ['fut_t', 'fut_tf', 'fut_ts', 'etf_511260']
    cfgs = [
        ('REF41 基线 vt=15%', ref41, None, 0.15, 2.5, {}),
        ('REF41 共振门控 vt=15%', ref41, None, 0.15, 2.5, dict(resonance_k=0.7)),
        ('REF41 共振门控 vt=20%', ref41, None, 0.20, 3.0, dict(resonance_k=0.7)),
        ('P2b宽基18 基线 vt=15%', p2b, BC, 0.15, 2.5, {}),
    ]
    navs = {}
    for lbl, cols, bond, vt, ml, kw in cfgs:
        navs[lbl] = run(panel, cols, bond, cash_ret, vt, ml, **kw)
        print(f"  {lbl}: 完成 ({len(navs[lbl])} 行)")

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    colors = ['#888888', '#d62728', '#ff7f0e', '#2ca02c']
    # 1 NAV (对数)
    ax = axes[0, 0]
    for (lbl, nav), c in zip(navs.items(), colors):
        ax.plot(nav.index, nav / nav.iloc[0], label=lbl, color=c, lw=1.6)
    ax.set_yscale('log')
    ax.set_title('净值曲线 (对数, 2019-01 起 = 1.0)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # 2 回撤
    ax = axes[0, 1]
    for (lbl, nav), c in zip(navs.items(), colors):
        dd = (nav / nav.cummax() - 1) * 100
        ax.plot(dd.index, dd, label=lbl, color=c, lw=1.2)
    ax.set_title('回撤 (%)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # 3 滚动 252 日夏普
    ax = axes[1, 0]
    for (lbl, nav), c in zip(navs.items(), colors):
        r = nav.pct_change()
        rs = (r.rolling(252).mean() * ANN) / (r.rolling(252).std() * np.sqrt(ANN))
        ax.plot(rs.index, rs, label=lbl, color=c, lw=1.2)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_title('滚动 252 日夏普')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # 4 分年收益
    ax = axes[1, 1]
    years = sorted(set(navs[list(navs)[0]].index.year))
    width = 0.2
    x = np.arange(len(years))
    for i, ((lbl, nav), c) in enumerate(zip(navs.items(), colors)):
        yr = []
        for y in years:
            s = nav[nav.index.year == y]
            yr.append((s.iloc[-1] / s.iloc[0] - 1) * 100 if len(s) > 5 else np.nan)
        ax.bar(x + (i - 1.5) * width, yr, width, label=lbl, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_title('分年收益 (%)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    out = 'docs/allweather_resonance_performance.png'
    plt.savefig(out, dpi=130)
    print(f"已保存: {out}")
    # 同时导出曲线数据
    pd.DataFrame(navs).to_csv('docs/allweather_resonance_performance.csv',
                              float_format='%.6f')
    print("已保存: docs/allweather_resonance_performance.csv")


if __name__ == '__main__':
    main()
