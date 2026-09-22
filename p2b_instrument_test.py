#!/usr/bin/env python3
# coding=utf-8
"""P2b 工具化对照: 中证1000 指数代理 → 可交易 512100 ETF (§5.46).

对比 P2b 宽基核心18 池中 csi_1000 (指数) vs etf_512100 (可交易 ETF) 的业绩差异.
512100 来自 OHLCV 复权面板 (已处理 2022-09-05 份额折算 +176%).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.d_pipeline import run_d_sleeve
from QuantNodes.strategy.allweather.volume_loader import load_volume_oi_panel
from QuantNodes.strategy.allweather.class_erc import ASSET_CLASSES

ANN = 252


def load():
    panel = pd.read_parquet('data/allweather/panel_v5.parquet').sort_index()
    panel.index = pd.to_datetime(panel.index)
    v7 = pd.read_parquet('data/high_freq_macro/v7_index_v2.parquet')
    v7.index = pd.to_datetime(v7.index)
    cash = v7['中债1-3年国债财富指数'].dropna()
    cash_ret = cash.pct_change(fill_method=None)
    ohlcv = pd.read_parquet(
        'data/real/etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet')
    ohlcv.index = pd.to_datetime(ohlcv.index)
    panel['etf_512100'] = ohlcv[('512100', 'close')].reindex(panel.index)
    csi = v7['中证1000'].dropna()
    panel['csi_1000'] = csi.reindex(panel.index).ffill()
    return panel.loc[:'2026-06-30'], cash_ret


def metr(nav):
    nav = nav.dropna()
    r = nav.pct_change(fill_method=None).dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    vol = r.std() * np.sqrt(ANN)
    return {'CAGR': (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1,
            'Sharpe': r.mean() * ANN / vol if vol > 0 else np.nan,
            'Vol': vol, 'MaxDD': (nav / nav.cummax() - 1).min()}


def build_pool(panel, csi_col):
    A = ['fut_t', 'fut_tf', 'fut_ts', 'etf_511260']
    B = ['etf_513100', 'etf_513500', 'etf_513520', 'etf_159920']
    C = ['fut_cu', 'fut_sc', 'fut_m', 'etf_159985']
    return A + [csi_col] + B + ['fut_au'] + C


def run(panel, cols, cash_ret, vt=0.15, ml=2.5):
    cls = {k: [c for c in v if c in cols] for k, v in ASSET_CLASSES.items()}
    for c, k in [('etf_510500', 'equity_etf_dom'), ('etf_588000', 'equity_etf_dom'),
                 ('etf_159920', 'equity_etf_off'), ('etf_513520', 'equity_etf_off'),
                 ('etf_511260', 'bond_rate'), ('etf_159985', 'agri_oilseed'),
                 ('csi_1000', 'equity_etf_dom'), ('etf_512100', 'equity_etf_dom')]:
        if c in cols and c not in [x for v in cls.values() for x in v]:
            cls.setdefault(k, []).append(c)
    cls = {k: v for k, v in cls.items() if v}
    sub = panel[cols]
    vol, oi = load_volume_oi_panel(sub, cache_path=None)
    bond = [c for c in ['fut_t', 'fut_tf', 'fut_ts', 'etf_511260'] if c in cols]
    res = run_d_sleeve(sub, bond_cap=0.40, window=252, rebalance_freq='W-FRI',
                       cov_estimator='lw', neg_trend_mode='trend',
                       neg_trend_preset='balanced', include_rapid=True,
                       momentum_beta=0.20, volume=vol, oi=oi, asset_classes=cls,
                       cash_ret=cash_ret, target_vol=vt, max_leverage=ml,
                       financing_rate=0.02, bond_cols=bond)
    nav = res.nav.dropna()
    return nav[nav.index >= '2019-01-01']


def main():
    panel, cash_ret = load()
    print("P2b 宽基核心18 工具化对照 (vt=15%, 融资2%)")
    print(f"{'变体':<26}{'CAGR':>8}{'Sharpe':>7}{'Vol':>8}{'MaxDD':>8}")
    out = {}
    for lbl, col in [('csi_1000 (指数代理)', 'csi_1000'),
                     ('etf_512100 (可交易 ETF)', 'etf_512100')]:
        cols = build_pool(panel, col)
        nav = run(panel, cols, cash_ret)
        out[lbl] = nav
        m = metr(nav)
        print(f"{lbl:<26}{m['CAGR']*100:>7.2f}%{m['Sharpe']:>7.2f}"
              f"{m['Vol']*100:>7.2f}%{m['MaxDD']*100:>7.2f}%")
    pd.DataFrame(out).to_csv('data/allweather/p2b_instrument_nav.csv',
                             float_format='%.6f')
    print("\n已保存: data/allweather/p2b_instrument_nav.csv")


if __name__ == '__main__':
    main()
