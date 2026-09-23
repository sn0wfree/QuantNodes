#!/usr/bin/env python3
# coding=utf-8
"""IV / 波动率风险溢价 (VRP) alpha 测试 (§5.52).

信号候选 (横截面, 月频美元中性多空):
  1. IV 水平      — 多高 IV / 空低 IV 及其反向
  2. VRP = IV−RV  — 波动率风险溢价 (RV = 20d 已实现波动) 及反向
  3. ΔIV(20d)     — IV 变化
评估: sleeve 独立 + 与 D 组合 (wc=0.1/0.2).
数据: scripts/extract_iv.py → data/allweather/iv_daily.parquet (6 产品).
"""
from __future__ import annotations

import sys

sys.path.insert(0, '.')
import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

import alpha_factors_test as A                              # noqa: E402
from QuantNodes.strategy.allweather.class_erc import ASSET_CLASSES  # noqa: E402
from QuantNodes.strategy.allweather.d_pipeline import run_d_sleeve  # noqa: E402
from QuantNodes.strategy.allweather.volume_loader import (   # noqa: E402
    load_volume_oi_panel)

ANN = 252


def metr(port):
    port = port.dropna()
    nav = (1 + port).cumprod()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    v = port.std() * np.sqrt(ANN)
    return {'CAGR': (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1,
            'Sharpe': port.mean() * ANN / v if v > 0 else np.nan,
            'Vol': v, 'MaxDD': (nav / nav.cummax() - 1).min()}


def main():
    panel = pd.read_parquet('data/allweather/panel_v5.parquet').sort_index()
    panel.index = pd.to_datetime(panel.index)
    v7 = pd.read_parquet('data/high_freq_macro/v7_index_v2.parquet')
    v7.index = pd.to_datetime(v7.index)
    cash_ret = v7['中债1-3年国债财富指数'].dropna().pct_change(fill_method=None)
    iv = pd.read_parquet('data/allweather/iv_daily.parquet')
    iv.index = pd.to_datetime(iv.index)
    P = panel.loc[:'2026-06-30'][[c for c in panel.columns if c not in A.NEW]]

    cols = [c for c in iv.columns if c in P.columns]
    print(f"IV 产品 ({len(cols)}): {cols}")
    rv = (P[cols].pct_change(fill_method=None)
          .rolling(20, min_periods=10).std() * np.sqrt(ANN))
    vrp = iv[cols] - rv

    signals = {
        'IV 多高波': iv[cols],
        'IV 空高波': -iv[cols],
        'VRP 多高溢价': vrp,
        'VRP 反向': -vrp,
        'ΔIV(20d)': iv[cols] - iv[cols].shift(20),
    }
    print(f"\n{'信号':<16}{'CAGR':>8}{'Sharpe':>8}{'Vol':>8}{'MaxDD':>8}")
    sleeves = {}
    for nm, sig in signals.items():
        port = A.sleeve_from_signal(P, sig)
        sleeves[nm] = port
        m = metr(port)
        print(f"{nm:<16}{m['CAGR']*100:>7.2f}%{m['Sharpe']:>8.2f}"
              f"{m['Vol']*100:>7.2f}%{m['MaxDD']*100:>8.2f}%")

    # 与 IV 平均相关性 (是否冗余)
    print("\n相关性 (日收益):")
    sl = pd.DataFrame(sleeves)
    print(sl.corr().round(3).to_string())

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
    print(f"\nD 基线: Sharpe {metr(r_d)['Sharpe']:.2f}")
    print(f"{'合成':<26}{'CAGR':>8}{'Sharpe':>8}{'Vol':>8}{'MaxDD':>8}")
    for nm in signals:
        alpha = sleeves[nm].reindex(idx).fillna(0.0)
        for wc in (0.1, 0.2):
            nav = (1 + (1 - wc) * r_d + wc * alpha).cumprod()
            m = metr(nav.pct_change().fillna(0.0))
            print(f"{nm + ' @' + str(wc):<26}{m['CAGR']*100:>7.2f}%"
                  f"{m['Sharpe']:>8.2f}{m['Vol']*100:>7.2f}%"
                  f"{m['MaxDD']*100:>8.2f}%")


if __name__ == '__main__':
    main()
