#!/usr/bin/env python3
# coding=utf-8
"""资产池对照实验 (A+B 路线图 §阶段3).

5 池对照, 同一 D v3 管线 + 保证金口径融资成本, 公共窗口 2018-01→2026-06-30.
评估: CAGR/Sharpe/Vol/MaxDD/Calmar + 分段 + 商品占比 + 有效资产数 N_eff.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.d_pipeline import run_d_sleeve
from QuantNodes.strategy.allweather.volume_loader import load_volume_oi_panel
from QuantNodes.strategy.allweather.class_erc import ASSET_CLASSES

ANN = 252
END = '2026-06-30'

# 新增 ETF 类别归属
NEW_CLS = {
    'etf_510500': 'equity_etf_dom', 'etf_510050': 'equity_etf_dom',
    'etf_588000': 'equity_etf_dom', 'etf_159928': 'equity_etf_dom',
    'etf_510880': 'equity_etf_dom', 'etf_512880': 'equity_etf_dom',
    'etf_513050': 'equity_etf_off', 'etf_159920': 'equity_etf_off',
    'etf_510900': 'equity_etf_off', 'etf_513520': 'equity_etf_off',
    'etf_518880': 'precious', 'etf_511260': 'bond_rate',
    'etf_512400': 'nonferrous', 'etf_159985': 'agri_oilseed',
    'csi_1000': 'equity_etf_dom',
}
COMMODITY_CLASSES = {'precious', 'nonferrous', 'ferrous',
                     'energy_chem', 'agri_oilseed', 'agri_soft'}

ALIGN11 = ['fut_t', 'fut_ts', 'fut_cu', 'fut_sc', 'fut_m', 'fut_au',
           'etf_512890', 'etf_513100', 'etf_513500', 'etf_159915', 'etf_510300']


def build_pools(panel: pd.DataFrame) -> dict[str, list[str]]:
    new_cols = [c for c in NEW_CLS if c != 'csi_1000']
    v4_cols = [c for c in panel.columns
               if c != 'csi_1000' and c not in new_cols]
    p0 = ALIGN11 + ['csi_1000']
    p1 = p0 + ['etf_510500', 'etf_588000', 'etf_513520', 'etf_159920', 'etf_510880']
    p2 = ['fut_t', 'fut_tf', 'fut_ts', 'etf_511260',
          'etf_510300', 'etf_510500', 'etf_588000', 'etf_159915',
          'etf_513100', 'etf_513500', 'etf_513520', 'etf_159920',
          'fut_au', 'etf_518880',
          'fut_cu', 'fut_sc', 'fut_m', 'etf_159985']
    drop = ['fut_j', 'fut_jm', 'fut_l', 'fut_pp', 'fut_bu', 'fut_p', 'fut_y']
    p3 = [c for c in v4_cols if c not in drop]
    return {'REF41': v4_cols, 'P0对齐12': p0, 'P1对齐+补充17': p1,
            'P2宽基核心18': p2, 'P3精简34': p3}


def make_classes(panel, cols):
    cls = {k: [c for c in v if c in cols] for k, v in ASSET_CLASSES.items()}
    for c, k in NEW_CLS.items():
        if c in cols and c not in [x for v in cls.values() for x in v]:
            cls.setdefault(k, []).append(c)
    return {k: v for k, v in cls.items() if v}


def metrics(nav):
    nav = nav.dropna()
    r = nav.pct_change(fill_method=None).dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(ANN)
    sh = r.mean() * ANN / vol if vol > 0 else np.nan
    dd = (nav / nav.cummax() - 1).min()
    calmar = cagr / abs(dd) if dd < 0 else np.nan
    return cagr, sh, vol, dd, calmar


def main():
    panel = pd.read_parquet('data/allweather/panel_v5.parquet').sort_index()
    panel.index = pd.to_datetime(panel.index)
    v7 = pd.read_parquet('data/high_freq_macro/v7_index_v2.parquet')
    cash = v7['中债1-3年国债财富指数'].dropna()
    cash.index = pd.to_datetime(cash.index)
    cash_ret = cash.pct_change(fill_method=None)
    csi = v7['中证1000'].dropna()
    csi.index = pd.to_datetime(csi.index)
    panel['csi_1000'] = csi.reindex(panel.index).ffill()
    panel = panel.loc[:END]

    pools = build_pools(panel)
    BOND_SETS = {'REF41': None, 'P0对齐12': None, 'P1对齐+补充17': None,
                 'P2宽基核心18': ['fut_t', 'fut_tf', 'fut_ts', 'etf_511260'],
                 'P3精简34': None}
    rows = []
    navs = {}
    print(f"公共窗口: {panel.index[0].date()} → {panel.index[-1].date()}")
    print(f"{'池':<16}{'N':>4}{'CAGR':>8}{'Sh':>6}{'Vol':>7}{'MaxDD':>8}"
          f"{'Calmar':>7}{'商品%':>7}{'N_eff':>7}")
    for name, cols in pools.items():
        cols = [c for c in cols if c in panel.columns]
        sub = panel[cols]
        cls = make_classes(panel, cols)
        vol, oi = load_volume_oi_panel(sub, cache_path=None)
        res = run_d_sleeve(sub, bond_cap=0.40, window=252, rebalance_freq='W-FRI',
                           cov_estimator='lw', neg_trend_mode='trend',
                           neg_trend_preset='balanced', include_rapid=True,
                           momentum_beta=0.20, volume=vol, oi=oi,
                           asset_classes=cls, cash_ret=cash_ret,
                           target_vol=0.085, max_leverage=2.5,
                           bond_cols=BOND_SETS.get(name))
        nav = res.nav.dropna()
        navs[name] = nav
        cagr, sh, volann, dd, calmar = metrics(nav)
        wbar = res.weights.mean()
        top5 = wbar.sort_values(ascending=False).head(5)
        print(f'   [{name}] top5 平均权重: ' +
              ', '.join(f'{k}={v:.3f}' for k, v in top5.items()))
        comm = sum(wbar[c] for c in cols if cls and any(
            c in v for k, v in cls.items() if k in COMMODITY_CLASSES))
        neff = 1.0 / (wbar ** 2).sum()
        print(f"{name:<16}{len(cols):>4}{cagr*100:>7.2f}%{sh:>6.2f}"
              f"{volann*100:>6.2f}%{dd*100:>7.2f}%{calmar:>7.2f}"
              f"{comm*100:>6.1f}%{neff:>7.1f}")
        rows.append({'pool': name, 'N': len(cols), 'CAGR': cagr, 'Sharpe': sh,
                     'Vol': volann, 'MaxDD': dd, 'Calmar': calmar,
                     'commodity_w': comm, 'N_eff': neff})

    out = pd.DataFrame(rows)
    out.to_csv('data/allweather/pool_comparison.csv', index=False)
    pd.DataFrame(navs).to_csv('data/allweather/pool_comparison_nav.csv',
                              float_format='%.8f')
    print("\n已保存: data/allweather/pool_comparison.csv + _nav.csv")


if __name__ == '__main__':
    main()
