#!/usr/bin/env python3
# coding=utf-8
"""Carry sleeve 内部优化测试 (§5.51).

维度:
  1. 调仓频率: ME / 2W / W-FRI
  2. 换手控制: 部分调整 (alpha) / 无交易带 (band)
  3. 权重方案: z-score / rank / sign
  4. 信号 clip: 1 / 2 / 3
  5. 信号平滑: carry 的 N 日 MA
  6. 离散度择时: 按横截面 carry 离散度缩放总敞口
评估: sleeve 独立 Sharpe/换手 + 与 D 组合 (wc=0.2) 的 Sharpe/MaxDD.
"""
from __future__ import annotations

import sys

sys.path.insert(0, '.')
import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

from QuantNodes.strategy.allweather.class_erc import ASSET_CLASSES  # noqa: E402
from QuantNodes.strategy.allweather.d_pipeline import run_d_sleeve  # noqa: E402
from QuantNodes.strategy.allweather.volume_loader import (   # noqa: E402
    load_volume_oi_panel)

ANN = 252
NEW = ('etf_510500', 'etf_588000', 'etf_159920', 'etf_513520',
       'etf_511260', 'etf_159985', 'csi_1000', 'etf_512100')


def metr(port):
    port = port.dropna()
    nav = (1 + port).cumprod()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    v = port.std() * np.sqrt(ANN)
    return {'CAGR': (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1,
            'Sharpe': port.mean() * ANN / v if v > 0 else np.nan,
            'Vol': v, 'MaxDD': (nav / nav.cummax() - 1).min()}


def period_end_mask(idx, rule):
    if rule == 'ME':
        key = [idx.year, idx.month]
    elif rule == 'W':
        iso = idx.isocalendar()
        key = [iso.year.values, iso.week.values]
    elif rule == '2W':
        iso = idx.isocalendar()
        key = [iso.year.values, iso.week.values // 2]
    else:
        raise ValueError(rule)
    s = pd.Series(idx, index=idx)
    return s.groupby(key).transform('max') == s


def build_sleeve(panel, sig, vol_win=60, gross=1.0, freq='ME', rate=0.0001,
                 scheme='zscore', clip=2.0, alpha=1.0, band=0.0,
                 smooth=1):
    """信号 → 美元中性多空权重 → 净收益; 返回 (port, w, 年换手)."""
    ret = panel.pct_change(fill_method=None)
    vol = ret.rolling(vol_win, min_periods=20).std()
    s = sig.reindex(panel.index).ffill() / vol.replace(0, np.nan)
    if smooth > 1:
        s = s.rolling(smooth, min_periods=1).mean()
    cols = [c for c in s.columns if c in panel.columns]
    s = s[cols]
    rb = period_end_mask(panel.index, freq)
    w = pd.DataFrame(0.0, index=panel.index, columns=cols)
    for i, d in enumerate(panel.index):
        if i > 0 and not rb.iloc[i]:
            w.iloc[i] = w.iloc[i - 1]
            continue
        sv = s.loc[d].dropna()
        if len(sv) < 6:
            continue
        if scheme == 'zscore':
            z = (sv - sv.mean()) / sv.std() if sv.std() > 0 else sv * 0
            z = z.clip(-clip, clip)
        elif scheme == 'rank':
            z = (sv.rank(pct=True) - 0.5) * 2.0 * clip
        elif scheme == 'sign':
            z = np.sign(sv - sv.median()) * 1.0
        else:
            raise ValueError(scheme)
        target = z / z.abs().sum() * gross if z.abs().sum() > 0 else z
        if i > 0:
            prev = w.iloc[i - 1]
            if band > 0 and (target - prev).abs().sum() < band * gross:
                w.iloc[i] = prev
                continue
            target = prev + alpha * (target - prev)
        w.loc[d, target.index] = target
    r = pd.Series(rate, index=cols)
    port = (w.shift(1) * ret[cols]).sum(axis=1) \
        - (w.diff().abs() * r).sum(axis=1).fillna(0.0)
    turnover = w.diff().abs().sum(axis=1).sum() / \
        ((w.index[-1] - w.index[0]).days / 365.25)
    return port.dropna(), w, turnover


def main():
    panel = pd.read_parquet('data/allweather/panel_v5.parquet').sort_index()
    panel.index = pd.to_datetime(panel.index)
    v7 = pd.read_parquet('data/high_freq_macro/v7_index_v2.parquet')
    v7.index = pd.to_datetime(v7.index)
    cash_ret = v7['中债1-3年国债财富指数'].dropna().pct_change(fill_method=None)
    carry = pd.read_parquet('data/allweather/carry_daily.parquet')
    carry.index = pd.to_datetime(carry.index)
    P = panel.loc[:'2026-06-30'][[c for c in panel.columns if c not in NEW]]

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
    r_d = nav_d.pct_change().fillna(0.0)
    idx = r_d.index
    print(f"D 基线: Sharpe {metr(r_d)['Sharpe']:.2f}")

    cfgs = [
        ('基线 ME/z2', dict(freq='ME', scheme='zscore', clip=2.0)),
        ('2W', dict(freq='2W', scheme='zscore', clip=2.0)),
        ('W-FRI', dict(freq='W', scheme='zscore', clip=2.0)),
        ('部分调整 a=0.5', dict(freq='ME', scheme='zscore', clip=2.0, alpha=0.5)),
        ('无交易带 0.10', dict(freq='ME', scheme='zscore', clip=2.0, band=0.10)),
        ('无交易带 0.20', dict(freq='ME', scheme='zscore', clip=2.0, band=0.20)),
        ('rank 权重', dict(freq='ME', scheme='rank', clip=2.0)),
        ('sign 权重', dict(freq='ME', scheme='sign', clip=2.0)),
        ('clip=1', dict(freq='ME', scheme='zscore', clip=1.0)),
        ('clip=3', dict(freq='ME', scheme='zscore', clip=3.0)),
        ('平滑 5d', dict(freq='ME', scheme='zscore', clip=2.0, smooth=5)),
        ('平滑 20d', dict(freq='ME', scheme='zscore', clip=2.0, smooth=20)),
    ]
    print(f"\n{'配置':<18}{'sleeve Sh':>10}{'换手/年':>9}"
          f"{'组合 Sh':>9}{'CAGR':>8}{'MaxDD':>8}")
    for nm, kw in cfgs:
        port, w, to = build_sleeve(P, carry, **kw)
        sm = metr(port)
        cs = port.reindex(idx).fillna(0.0)
        nav = (1 + 0.8 * r_d + 0.2 * cs).cumprod()
        cm = metr(nav.pct_change().fillna(0.0))
        print(f"{nm:<18}{sm['Sharpe']:>10.2f}{to:>9.1f}"
              f"{cm['Sharpe']:>9.2f}{cm['CAGR']*100:>7.2f}%"
              f"{cm['MaxDD']*100:>8.2f}%")

    # 离散度择时
    print(f"\n{'离散度择时':<18}{'sleeve Sh':>10}{'换手/年':>9}"
          f"{'组合 Sh':>9}{'CAGR':>8}{'MaxDD':>8}")
    port0, w0, to0 = build_sleeve(P, carry, freq='ME', scheme='zscore', clip=2.0)
    ret = P.pct_change(fill_method=None)
    volc = ret.rolling(60, min_periods=20).std()
    sig = carry.reindex(P.index).ffill() / volc.replace(0, np.nan)
    disp = sig.std(axis=1)
    for pct in (0.5, 0.7):
        thr = disp.expanding(min_periods=252).quantile(pct).shift(1)
        scale = pd.Series(1.0, index=idx)
        d2 = disp.reindex(idx).shift(1)
        t2 = thr.reindex(idx)
        scale[(d2 < t2)] = 0.5           # 低离散度 → 半仓
        pc = port0.reindex(idx).fillna(0.0) * scale
        sm = metr(pc)
        nav = (1 + 0.8 * r_d + 0.2 * pc).cumprod()
        cm = metr(nav.pct_change().fillna(0.0))
        print(f"离散度 P{int(pct*100)} 半仓  {sm['Sharpe']:>10.2f}"
              f"{to0:>9.1f}{cm['Sharpe']:>9.2f}{cm['CAGR']*100:>7.2f}%"
              f"{cm['MaxDD']*100:>8.2f}%")


if __name__ == '__main__':
    main()
