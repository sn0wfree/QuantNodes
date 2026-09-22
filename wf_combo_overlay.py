#!/usr/bin/env python3
# coding=utf-8
"""P1-B + 共振门控 组合配置 walk-forward 复验 (§5.42).

参数空间: (correlation_k, resonance_k) ∈ {(0,0),(1.0,0),(0,0.7),(0.5,0.7),
          (1.0,0.7),(1.0,0.5)}; 其余固定 (θ_corr=0.60, AR3 60/0.70/0.40/0.50,
          dd_floor=-0.03). 扩张 IS 选参 (最大化 Sharpe) → OOS 逐年评估.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.d_pipeline import run_d_sleeve
from QuantNodes.strategy.allweather.volume_loader import load_volume_oi_panel
from QuantNodes.strategy.allweather.class_erc import ASSET_CLASSES

ANN = 252


def metr(nav):
    nav = nav.dropna()
    r = nav.pct_change(fill_method=None).dropna()
    if len(r) < 20:
        return np.nan, np.nan, np.nan
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    vol = r.std() * np.sqrt(ANN)
    return ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1,
            r.mean() * ANN / vol if vol > 0 else np.nan,
            (nav / nav.cummax() - 1).min())


def main():
    panel = pd.read_parquet('data/allweather/panel_v5.parquet').sort_index()
    panel.index = pd.to_datetime(panel.index)
    v7 = pd.read_parquet('data/high_freq_macro/v7_index_v2.parquet')
    cash = v7['中债1-3年国债财富指数'].dropna()
    cash.index = pd.to_datetime(cash.index)
    cash_ret = cash.pct_change(fill_method=None)
    new = ('etf_510500', 'etf_588000', 'etf_159920', 'etf_513520',
           'etf_511260', 'etf_159985', 'csi_1000')
    ref41 = [c for c in panel.columns if c not in new]
    panel = panel.loc[:'2026-06-30'][ref41]
    vol, oi = load_volume_oi_panel(panel, cache_path=None)

    combos = [(0.0, 0.0, '无叠加层'), (1.0, 0.0, '仅P1-B'),
              (0.0, 0.7, '仅共振'), (0.5, 0.7, 'P1-B(0.5)+共振'),
              (1.0, 0.7, 'P1-B(1.0)+共振'), (1.0, 0.5, 'P1-B(1.0)+共振(0.5)')]
    navs, base_nav = {}, None
    for ck, rk, lbl in combos:
        res = run_d_sleeve(
            panel, bond_cap=0.40, window=252, rebalance_freq='W-FRI',
            cov_estimator='lw', neg_trend_mode='trend',
            neg_trend_preset='balanced', include_rapid=True,
            momentum_beta=0.20, volume=vol, oi=oi,
            asset_classes=dict(ASSET_CLASSES), cash_ret=cash_ret,
            target_vol=0.15, max_leverage=2.5, financing_rate=0.02,
            correlation_k=ck, correlation_threshold=0.60,
            resonance_k=rk, resonance_dd_floor=-0.03)
        navs[lbl] = res.nav.dropna()
        c, s, d = metr(navs[lbl][navs[lbl].index >= '2019-01-01'])
        print(f"  {lbl:<22} 全样本 CAGR {c*100:>6.2f}% Sh {s:>5.2f} "
              f"MaxDD {d*100:>7.2f}%")
        if lbl == '无叠加层':
            base_nav = navs[lbl]

    print(f"\n{'OOS年':>7}{'IS选出':>22}{'门控 Sh/DD':>18}{'基线 Sh/DD':>18}")
    sel = []
    for y in ['2022', '2023', '2024', '2025', '2026']:
        is_end = f'{int(y)-1}-12-31'
        a, b = f'{y}-01-01', f'{y}-06-30' if y == '2026' else f'{y}-12-31'
        best, bs = None, -99
        for lbl, nav in navs.items():
            _, s, _ = metr(nav[nav.index <= is_end])
            if np.isfinite(s) and s > bs:
                bs, best = s, lbl
        gc, gs, gd = metr(navs[best].loc[a:b])
        bc, bs2, bd = metr(base_nav.loc[a:b])
        sel.append((gs, bs2, gd, bd))
        print(f"{y:>7}{best:>22}{f'{gs:.2f}/{gd*100:.1f}%':>18}"
              f"{f'{bs2:.2f}/{bd*100:.1f}%':>18}")
    print(f"{'均值':>7}{'':>22}"
          f"{f'{np.mean([x[0] for x in sel]):.2f}/{np.mean([x[2] for x in sel])*100:.1f}%':>18}"
          f"{f'{np.mean([x[1] for x in sel]):.2f}/{np.mean([x[3] for x in sel])*100:.1f}%':>18}")

    # 固定组合配置的 OOS
    for lbl in ('P1-B(1.0)+共振', '仅共振'):
        ss, ds = [], []
        for y in ['2022', '2023', '2024', '2025', '2026']:
            a = f'{y}-01-01'
            b = f'{y}-06-30' if y == '2026' else f'{y}-12-31'
            _, s, d = metr(navs[lbl].loc[a:b])
            ss.append(s)
            ds.append(d)
        print(f"固定 {lbl:<18} OOS 均值 Sh {np.nanmean(ss):.2f} "
              f"DD {np.nanmean(ds)*100:.1f}%")


if __name__ == '__main__':
    main()
