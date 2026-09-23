#!/usr/bin/env python3
# coding=utf-8
"""Carry (展期收益) 倾斜 overlay 测试 (§5.48).

在生产候选 (REF41 + ME + P1-B + 共振) 权重上叠加横截面 carry 倾斜:
  w_i × (1 + β·score_i), score = 2×(carry 横截面秩分 − 0.5) ∈ [-1,1]
  PIT: carry 信号 shift(1); 倾斜后按总敞口重缩放 (不改变杠杆).
对照: 基线 / 多个 β / carry 过滤变体.
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


def load():
    panel = pd.read_parquet('data/allweather/panel_v5.parquet').sort_index()
    panel.index = pd.to_datetime(panel.index)
    v7 = pd.read_parquet('data/high_freq_macro/v7_index_v2.parquet')
    v7.index = pd.to_datetime(v7.index)
    cash_ret = v7['中债1-3年国债财富指数'].dropna().pct_change(fill_method=None)
    carry = pd.read_parquet('data/allweather/carry_daily.parquet')
    carry.index = pd.to_datetime(carry.index)
    ref41 = [c for c in panel.columns if c not in NEW]
    return panel.loc[:'2026-06-30'][ref41], cash_ret, carry


def sim(w, ret, cash_ret):
    rate = pd.Series({c: (0.0001 if c.startswith('fut_') else 0.0005)
                      for c in w.columns})
    port = (w.shift(1) * ret).sum(axis=1)
    port -= (w.diff().abs() * rate).sum(axis=1).fillna(0)
    port -= (w.sum(axis=1).shift(1) - 1.0).clip(lower=0) * 0.02 / ANN
    port += (1.0 - w.sum(axis=1).shift(1)).clip(lower=0) \
        * cash_ret.reindex(w.index).ffill().fillna(0)
    nav = (1 + port.fillna(0)).cumprod()
    return nav[nav.index >= '2019-01-01']


def metr(nav):
    nav = nav.dropna()
    r = nav.pct_change(fill_method=None).dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    vol = r.std() * np.sqrt(ANN)
    return {'CAGR': (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1,
            'Sharpe': r.mean() * ANN / vol if vol > 0 else np.nan,
            'MaxDD': (nav / nav.cummax() - 1).min()}


def apply_carry_tilt(w, carry, beta):
    """w × (1+β·score), 保持总敞口 (按行重缩放)."""
    sc = carry.reindex(w.index).ffill()
    sc = (sc.rank(axis=1, pct=True) - 0.5) * 2.0      # ∈[-1,1]
    sc = sc.shift(1).clip(-1, 1)                      # PIT
    ww = w.copy()
    for c in [x for x in w.columns if x in sc.columns]:
        ww[c] = w[c] * (1.0 + beta * sc[c].fillna(0.0))
    gross = w.sum(axis=1)
    rowsum = ww.sum(axis=1).replace(0, np.nan)
    return ww.mul(gross / rowsum, axis=0).fillna(0.0)


def main():
    panel, cash_ret, carry = load()
    print(f"carry_daily: {carry.shape}, 覆盖至 {carry.index.max().date()}")
    vol, oi = load_volume_oi_panel(panel, cache_path=None)
    res = run_d_sleeve(panel, bond_cap=0.40, window=252, rebalance_freq='ME',
                       cov_estimator='lw', neg_trend_mode='trend',
                       neg_trend_preset='balanced', include_rapid=True,
                       momentum_beta=0.20, volume=vol, oi=oi,
                       asset_classes=dict(ASSET_CLASSES), cash_ret=cash_ret,
                       target_vol=0.15, max_leverage=2.5, financing_rate=0.02,
                       correlation_k=1.0, correlation_threshold=0.60,
                       resonance_k=0.7)
    w = res.weights
    ret = panel.pct_change(fill_method=None).reindex(w.index)

    print(f"\n{'配置':<24}{'CAGR':>8}{'Sharpe':>7}{'MaxDD':>8}")
    m = metr(sim(w, ret, cash_ret))
    print(f"{'基线 (无 carry 倾斜)':<24}{m['CAGR']*100:>7.2f}%"
          f"{m['Sharpe']:>7.2f}{m['MaxDD']*100:>8.2f}%")
    for beta in (0.2, 0.4, 0.6, 0.8):
        ww = apply_carry_tilt(w, carry, beta)
        m = metr(sim(ww, ret, cash_ret))
        print(f"{'carry tilt β=' + str(beta):<24}{m['CAGR']*100:>7.2f}%"
              f"{m['Sharpe']:>7.2f}{m['MaxDD']*100:>8.2f}%")


if __name__ == '__main__':
    main()
