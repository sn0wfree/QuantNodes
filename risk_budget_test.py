#!/usr/bin/env python3
# coding=utf-8
"""D + Carry 联合风险预算测试 (§5.50).

对比固定权重 (80/20) vs 数据驱动联合风险预算:
  1. fixed      固定 wc
  2. erc        滚动 ERC (等风险贡献; 2 资产 = 逆波动)
  3. inv_vol    滚动逆波动
  4. max_sharpe 滚动切点组合 (Σ⁻¹μ, 多头约束)
  5. min_var    滚动最小方差
每类 ± 目标波动率缩放 (vol targeting).
评估 Sharpe/CAGR/MaxDD + walk-forward.
"""
from __future__ import annotations

import sys

sys.path.insert(0, '.')
import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

from QuantNodes.strategy.allweather.carry_sleeve import (    # noqa: E402
    build_carry_sleeve, month_end_mask)
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
    if len(port) < 20:
        return {'CAGR': np.nan, 'Sharpe': np.nan, 'Vol': np.nan, 'MaxDD': np.nan}
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    v = port.std() * np.sqrt(ANN)
    return {'CAGR': (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1,
            'Sharpe': port.mean() * ANN / v if v > 0 else np.nan,
            'Vol': v, 'MaxDD': (nav / nav.cummax() - 1).min()}


def erc_weights(cov):
    n = cov.shape[0]
    if n == 2:
        inv = 1.0 / np.sqrt(np.diag(cov))
        return inv / inv.sum()
    w = np.ones(n) / n
    for _ in range(2000):
        mrc = cov @ w
        rc = w * mrc
        w = w * (rc.mean() / np.where(rc > 0, rc, np.nan))
        w = np.nan_to_num(w, nan=1.0 / n)
        w = w / w.sum()
    return w


def inv_vol_weights(cov):
    inv = 1.0 / np.sqrt(np.diag(cov))
    return inv / inv.sum()


def max_sharpe_weights(cov, mu):
    w = np.linalg.pinv(cov) @ mu
    w = np.clip(w, 0, None)
    return w / w.sum() if w.sum() > 0 else np.ones(len(mu)) / len(mu)


def min_var_weights(cov):
    inv = np.linalg.pinv(cov) @ np.ones(cov.shape[0])
    inv = np.clip(inv, 0, None)
    return inv / inv.sum() if inv.sum() > 0 else np.ones(cov.shape[0]) / cov.shape[0]


def rolling_weights(ret, method, window=252, min_periods=120):
    """滚动权重 (月频调仓, 只用 t 之前数据)."""
    rb = month_end_mask(ret.index)
    w = pd.DataFrame(np.nan, index=ret.index, columns=ret.columns)
    for i, _t in enumerate(ret.index):
        if i > 0 and not rb.iloc[i]:
            w.iloc[i] = w.iloc[i - 1]
            continue
        hist = ret.iloc[max(0, i - window):i]
        if len(hist) < min_periods:
            w.iloc[i] = 1.0 / ret.shape[1]
            continue
        cov = hist.cov().values
        if method == 'erc':
            w.iloc[i] = erc_weights(cov)
        elif method == 'inv_vol':
            w.iloc[i] = inv_vol_weights(cov)
        elif method == 'max_sharpe':
            w.iloc[i] = max_sharpe_weights(cov, hist.mean().values)
        elif method == 'min_var':
            w.iloc[i] = min_var_weights(cov)
    return w


def vol_target(port, target=0.10, window=60, max_lev=2.5, fin=0.02):
    rv = port.rolling(window, min_periods=20).std() * np.sqrt(ANN)
    lev = (target / rv).clip(upper=max_lev).shift(1).fillna(1.0)
    return port * lev - (lev - 1.0).clip(lower=0) * fin / ANN


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
    cs, _ = build_carry_sleeve(P, carry, freq='ME')
    cs = cs.reindex(r_d.index).fillna(0.0)
    ret = pd.DataFrame({'D': r_d, 'carry': cs})

    print(f"D 单独 : {metr(r_d)}")
    print(f"carry  : {metr(cs)}")
    rho = ret['D'].corr(ret['carry'])
    print(f"相关 D vs carry: {rho:.3f}")
    print(f"年化波动 D {r_d.std()*np.sqrt(ANN)*100:.2f}% / "
          f"carry {cs.std()*np.sqrt(ANN)*100:.2f}%")
    print("→ ERC(2资产)=逆波动 ≈ "
          f"{(1/(r_d.std()))/(1/r_d.std()+1/cs.std())*100:.0f}% D / "
          f"{(1/cs.std())/(1/r_d.std()+1/cs.std())*100:.0f}% carry")

    methods = {'fixed20': None, 'fixed30': None, 'erc': 'erc',
               'inv_vol': 'inv_vol', 'max_sharpe': 'max_sharpe',
               'min_var': 'min_var'}
    print(f"\n{'方法':<26}{'CAGR':>8}{'Sharpe':>8}{'Vol':>8}{'MaxDD':>8}")
    results = {}
    for nm, m in methods.items():
        if nm.startswith('fixed'):
            wc = 0.2 if nm == 'fixed20' else 0.3
            w = pd.DataFrame({'D': 1 - wc, 'carry': wc}, index=ret.index)
        else:
            w = rolling_weights(ret, m)
        port = (w.shift(1) * ret).sum(axis=1).dropna()
        results[nm] = port
        mt = metr(port)
        print(f"{nm:<26}{mt['CAGR']*100:>7.2f}%{mt['Sharpe']:>8.2f}"
              f"{mt['Vol']*100:>7.2f}%{mt['MaxDD']*100:>8.2f}%")
        # vol-target 10%
        pv = vol_target(port, target=0.10)
        mt = metr(pv)
        results[nm + '+vt10'] = pv
        print(f"{nm + ' +vt10':<26}{mt['CAGR']*100:>7.2f}%{mt['Sharpe']:>8.2f}"
              f"{mt['Vol']*100:>7.2f}%{mt['MaxDD']*100:>8.2f}%")

    # 平均 ERC 权重诊断
    w_erc = rolling_weights(ret, 'erc')
    print(f"\nERC 平均权重: D {w_erc['D'].mean():.3f} / "
          f"carry {w_erc['carry'].mean():.3f}")

    # Walk-forward: IS 选方法 (Sharpe 最大, 达标约束), OOS 评估
    print("\n=== Walk-forward (IS 选方法, OOS 评估) ===")
    print(f"{'OOS年':>7}{'IS选出':>16}{'OOS Sh/CAGR/DD':>22}"
          f"{'基线fixed20 Sh':>18}")
    for y in ['2022', '2023', '2024', '2025', '2026']:
        is_end = f'{int(y)-1}-12-31'
        a = f'{y}-01-01'
        b = f'{y}-06-30' if y == '2026' else f'{y}-12-31'
        best, bs = None, -99
        for nm in results:
            p = results[nm]
            mt = metr(p[p.index <= is_end])
            if mt['CAGR'] >= 0.10 and mt['MaxDD'] >= -0.10 \
                    and np.isfinite(mt['Sharpe']) and mt['Sharpe'] > bs:
                bs, best = mt['Sharpe'], nm
        if best is None:
            for nm in results:
                p = results[nm]
                mt = metr(p[p.index <= is_end])
                if np.isfinite(mt['Sharpe']) and mt['Sharpe'] > bs:
                    bs, best = mt['Sharpe'], nm
        pc = results[best].loc[a:b]
        base = results['fixed20'].loc[a:b]
        mc, mb = metr(pc), metr(base)
        cs_ = f"{mc['Sharpe']:.2f}/{mc['CAGR']*100:5.1f}%/{mc['MaxDD']*100:6.1f}%"
        bs_ = f"{mb['Sharpe']:.2f}"
        print(f"{y:>7}{best:>16}{cs_:>22}{bs_:>18}")


if __name__ == '__main__':
    main()
