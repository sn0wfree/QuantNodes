#!/usr/bin/env python3
# coding=utf-8
"""Carry 多空 sleeve 测试 (§5.48).

经典商品/FI carry 因子实现:
  1. 信号 = carry / vol (风险调整展期收益), 横截面标准化
  2. 多空组合: 权重 ∝ z-score, 美元中性 (Σw=0), gross ≤ 1
  3. 作为独立 sleeve 与 D 组合按权重合成
评估: sleeve 独立表现 + 与 D 组合后 Sharpe 提升.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.d_pipeline import run_d_sleeve
from QuantNodes.strategy.allweather.volume_loader import load_volume_oi_panel
from QuantNodes.strategy.allweather.class_erc import ASSET_CLASSES

ANN = 252
NEW = ('etf_510500', 'etf_588000', 'etf_159920', 'etf_513520',
       'etf_511260', 'etf_159985', 'csi_1000', 'etf_512100')


def metr(nav):
    nav = nav.dropna()
    r = nav.pct_change(fill_method=None).dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    vol = r.std() * np.sqrt(ANN)
    return {'CAGR': (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1,
            'Sharpe': r.mean() * ANN / vol if vol > 0 else np.nan,
            'Vol': vol, 'MaxDD': (nav / nav.cummax() - 1).min()}


def build_carry_sleeve(panel, carry, vol_win=60, freq='ME', gross_cap=1.0):
    """carry/vol 横截面 z-score → 美元中性多空权重 (调仓日重置)."""
    ret = panel.pct_change(fill_method=None)
    vol = ret.rolling(vol_win, min_periods=20).std()
    sig = carry.reindex(panel.index).ffill() / vol.replace(0, np.nan)
    cols = [c for c in sig.columns if c in panel.columns]
    sig = sig[cols]
    # 调仓掩码
    if freq == 'ME':
        rb = panel.index.to_series().groupby(
            [panel.index.year, panel.index.month]).transform('max') == \
            panel.index.to_series()
    else:
        rb = pd.Series(True, index=panel.index)
    w = pd.DataFrame(0.0, index=panel.index, columns=cols)
    for i, d in enumerate(panel.index):
        if i > 0 and not rb.iloc[i]:
            w.iloc[i] = w.iloc[i - 1]
            continue
        s = sig.loc[d].dropna()
        if len(s) < 6:
            continue
        z = (s - s.mean()) / s.std() if s.std() > 0 else s * 0
        z = z.clip(-2, 2)
        ww = z / z.abs().sum() * gross_cap if z.abs().sum() > 0 else z
        w.loc[d, ww.index] = ww
    rate = pd.Series({c: 0.0001 for c in cols})
    port = (w.shift(1) * ret[cols]).sum(axis=1) \
        - (w.diff().abs() * rate).sum(axis=1).fillna(0)
    return port.dropna(), w


def main():
    panel = pd.read_parquet('data/allweather/panel_v5.parquet').sort_index()
    panel.index = pd.to_datetime(panel.index)
    v7 = pd.read_parquet('data/high_freq_macro/v7_index_v2.parquet')
    v7.index = pd.to_datetime(v7.index)
    cash_ret = v7['中债1-3年国债财富指数'].dropna().pct_change(fill_method=None)
    carry = pd.read_parquet('data/allweather/carry_daily.parquet')
    carry.index = pd.to_datetime(carry.index)
    ref41 = [c for c in panel.columns if c not in NEW]
    P = panel.loc[:'2026-06-30'][ref41]

    # D 组合 (ME + 调优参数)
    vol, oi = load_volume_oi_panel(P, cache_path=None)
    res = run_d_sleeve(P, bond_cap=0.40, window=252, rebalance_freq='ME',
                       cov_estimator='lw', neg_trend_mode='trend',
                       neg_trend_preset='aggressive', include_rapid=True,
                       momentum_beta=0.20, volume=vol, oi=oi,
                       asset_classes=dict(ASSET_CLASSES), cash_ret=cash_ret,
                       target_vol=0.15, max_leverage=2.5, financing_rate=0.02,
                       correlation_k=1.0, correlation_threshold=0.60,
                       resonance_k=0.9, resonance_dd_floor=-0.05)
    nav_d = res.nav.dropna()
    nav_d = nav_d[nav_d.index >= '2019-01-01']
    m = metr(nav_d)
    print(f"D 组合 (ME+调优): CAGR {m['CAGR']*100:.2f}% "
          f"Sh {m['Sharpe']:.2f} Vol {m['Vol']*100:.2f}% MaxDD {m['MaxDD']*100:.2f}%")

    # Carry sleeve (独立)
    cs, w = build_carry_sleeve(P, carry, freq='ME')
    cs = cs.reindex(nav_d.index).fillna(0.0)
    nav_c = (1 + cs).cumprod()
    m = metr(nav_c)
    print(f"Carry sleeve 独立 : CAGR {m['CAGR']*100:.2f}% "
          f"Sh {m['Sharpe']:.2f} Vol {m['Vol']*100:.2f}% MaxDD {m['MaxDD']*100:.2f}%")

    # 合成
    r_d = nav_d.pct_change().fillna(0.0)
    print(f"\n{'合成权重':<24}{'CAGR':>8}{'Sharpe':>8}{'Vol':>8}{'MaxDD':>8}")
    for wc in (0.0, 0.1, 0.2, 0.3, 0.5):
        nav = (1 + (1 - wc) * r_d + wc * cs).cumprod()
        m = metr(nav)
        print(f"{'D ' + str(round(1-wc,2)) + ' + Carry ' + str(wc):<24}"
              f"{m['CAGR']*100:>7.2f}%{m['Sharpe']:>8.2f}"
              f"{m['Vol']*100:>7.2f}%{m['MaxDD']*100:>8.2f}%")
    # 信号正交性: carry vs 动量 横截面秩相关
    mom = P.pct_change(60, fill_method=None)
    cc = carry.reindex(P.index)
    rs = []
    for d in cc.index[-800:]:
        a, b = cc.loc[d], mom.loc[d]
        m2 = a.notna() & b.notna()
        if m2.sum() >= 8:
            rs.append(a[m2].rank().corr(b[m2].rank()))
    print(f"\ncarry vs 60日动量 横截面秩相关: {np.nanmean(rs):+.3f} "
          f"(n={len(rs)}天, 近0=正交)")


if __name__ == '__main__':
    main()
