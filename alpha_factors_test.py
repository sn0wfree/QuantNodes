#!/usr/bin/env python3
# coding=utf-8
"""多真 alpha 因子 sleeve 测试 (§5.49).

因子 (全部 PIT, 月频调仓, 美元中性多空):
  1. carry         展期收益 (carry/vol z-score)         [已有]
  2. basis_mom     基差动量 Δcarry(20d)/vol             [Boons-Prado 2019]
  3. seasonality   季节性 (同月历史平均收益, 扩张窗口)
各因子独立 + 等权合成 + 与 D 组合, 评估 Sharpe 提升.
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


def _month_end_mask(idx):
    s = pd.Series(idx, index=idx)
    return (s.groupby([idx.year, idx.month]).transform('max') == s)


def sleeve_from_signal(panel, signal, vol_win=60, gross=1.0, freq='ME'):
    """信号 → 美元中性多空权重 (调仓日重置) → 净收益序列."""
    ret = panel.pct_change(fill_method=None)
    vol = ret.rolling(vol_win, min_periods=20).std()
    sig = signal.reindex(panel.index).ffill() / vol.replace(0, np.nan)
    cols = [c for c in sig.columns if c in panel.columns]
    sig = sig[cols]
    if freq == 'ME':
        rb = _month_end_mask(panel.index)
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
        if z.abs().sum() > 0:
            w.loc[d, z.index] = z / z.abs().sum() * gross
    rate = pd.Series({c: 0.0001 for c in cols})
    port = (w.shift(1) * ret[cols]).sum(axis=1) \
        - (w.diff().abs() * rate).sum(axis=1).fillna(0)
    return port.dropna()


def seasonality_signal(panel):
    """同月历史平均日收益 (扩张窗口, PIT)."""
    ret = panel.pct_change(fill_method=None)
    out = pd.DataFrame(np.nan, index=panel.index, columns=panel.columns)
    for m in range(1, 13):
        sub = ret[ret.index.month == m]
        # 按年分组 → 截至上一年同月的累计均值 (PIT)
        yr = sub.index.year
        for y in sorted(set(yr)):
            hist = sub[yr < y]
            if len(hist) < 40:
                continue
            out.loc[sub.index[yr == y], :] = hist.mean().values
    return out


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

    sigs = {
        'carry': carry,
        'basis_mom': carry - carry.shift(20),          # Δcarry (20日)
        'seasonality': seasonality_signal(P),
    }
    sleeves = {}
    print("=== 因子 sleeve 独立表现 (月频美元中性多空) ===")
    print(f"{'因子':<16}{'CAGR':>8}{'Sharpe':>8}{'Vol':>8}{'MaxDD':>8}")
    for nm, s in sigs.items():
        port = sleeve_from_signal(P, s)
        sleeves[nm] = port
        m = metr((1 + port).cumprod())
        print(f"{nm:<16}{m['CAGR']*100:>7.2f}%{m['Sharpe']:>8.2f}"
              f"{m['Vol']*100:>7.2f}%{m['MaxDD']*100:>8.2f}%")

    # 因子相关性
    print("\n因子相关 (日收益):")
    sl = pd.DataFrame(sleeves)
    print(sl.corr().round(3).to_string())

    # D 组合基线
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
    idx = nav_d.index
    m = metr(nav_d)
    print(f"\nD 组合基线: CAGR {m['CAGR']*100:.2f}% Sh {m['Sharpe']:.2f} "
          f"Vol {m['Vol']*100:.2f}% MaxDD {m['MaxDD']*100:.2f}%")

    combos = {
        'carry': ['carry'],
        'carry+basis_mom': ['carry', 'basis_mom'],
        'carry+season': ['carry', 'seasonality'],
        '全部三因子': ['carry', 'basis_mom', 'seasonality'],
    }
    print(f"\n{'合成':<28}{'CAGR':>8}{'Sharpe':>8}{'Vol':>8}{'MaxDD':>8}")
    for nm, keys in combos.items():
        for wc in (0.2, 0.3):
            alpha = sl[keys].reindex(idx).fillna(0).mean(axis=1)
            nav = (1 + (1 - wc) * r_d + wc * alpha).cumprod()
            m = metr(nav)
            print(f"{nm + ' @' + str(wc):<28}{m['CAGR']*100:>7.2f}%"
                  f"{m['Sharpe']:>8.2f}{m['Vol']*100:>7.2f}%{m['MaxDD']*100:>8.2f}%")


if __name__ == '__main__':
    main()
