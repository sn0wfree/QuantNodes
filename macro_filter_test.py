#!/usr/bin/env python3
# coding=utf-8
"""宏观环境边际过滤消融 (§5.36): DXY / VIX / US10Y 边际信号 → 敞口缩放.

基线: 41 资产 D v3 全栈 vt=8.5% ml=2.5 (§5.35/杠杆阶梯口径).
设计 A: 组合级乘数 (composite → 总敞口).
设计 B: 类别级异质乘数 (因子→类别定向缩放, 同 D.1 哲学).
防前视: 宏观数据 shift(1); 信号缺失 ffill 5 日后中性 0.
诚实门: Sharpe/MaxDD 显著改善才保留.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.d_pipeline import run_d_sleeve
from QuantNodes.strategy.allweather.volume_loader import load_volume_oi_panel
from QuantNodes.strategy.allweather.class_erc import ASSET_CLASSES

ANN = 252
FIN = 0.02
K = 0.30          # 宏观过滤最大削减幅度 (risk-off 时最多 -30% 敞口)
FFILL_LIMIT = 5

# ── 数据 ──
panel = pd.read_parquet('data/allweather/panel_v4.parquet').sort_index()
v7 = pd.read_parquet('data/high_freq_macro/v7_index_v2.parquet')
cash = v7['中债1-3年国债财富指数'].dropna()
cash.index = pd.to_datetime(cash.index)
cash_ret = cash.pct_change(fill_method=None)


def load_macro():
    dxy = pd.read_parquet('data/high_freq_macro/macro_dxy_daily_v2.parquet')['dxy']
    dxy.index = pd.to_datetime(dxy.index)
    vix = pd.read_parquet('data/high_freq_macro/macro_vix_daily.parquet')['vix']
    vix.index = pd.to_datetime(vix.index)
    # US10Y: date 是列 → 修为索引
    us = pd.read_parquet('data/high_freq_macro/us_10y_yield_fred.parquet')
    if 'date' in us.columns:
        us['date'] = pd.to_datetime(us['date'])
        us = us.set_index('date')
    us10 = us['us_10y_yield'].dropna()
    us10.index = pd.to_datetime(us10.index)
    return dxy.sort_index(), vix.sort_index(), us10.sort_index()


def zsig(s, win, zwin=60):
    """win 日边际变化的 z-score → tanh 压缩 ∈ (-1,1)."""
    chg = s.diff(win) if s.index.is_monotonic_increasing else np.nan
    m = chg.rolling(zwin).mean()
    sd = chg.rolling(zwin).std()
    z = (chg - m) / sd.replace(0, np.nan)
    return np.tanh(0.5 * z)


def build_signals(dates):
    dxy, vix, us10 = load_macro()
    sig_dxy = zsig(dxy, 20)
    sig_us = zsig(us10, 20)

    v5 = zsig(vix, 5)
    lvl = vix.rolling(252, min_periods=60).rank(pct=True) - 0.5
    sig_vix = (0.6 * v5 + 0.4 * lvl).clip(-1, 1)

    out = {}
    for key, s in [('dxy', sig_dxy), ('vix', sig_vix), ('us10', sig_us)]:
        s = s.shift(1)  # 防前视: T-1 收盘信息 T 日生效
        s = s.reindex(dates).ffill(limit=FFILL_LIMIT).fillna(0.0)
        out[key] = s
    return out


# 因子 → 类别映射 (设计 B): 因子为正 (risk-off) 时定向削减这些类别
CLASS_MAP = {
    'dxy': ['nonferrous', 'energy_chem', 'agri_oilseed', 'agri_soft'],
    'vix': ['equity_etf_dom', 'equity_etf_off', 'equity_fut'],
    'us10': ['bond_rate', 'equity_etf_dom'],
}
# precious 不在映射里: VIX 飙升黄金常为对冲受益, 不削


def apply_portfolio_filter(w, score, k=K):
    """设计 A: 组合级乘数 m = 1 - k·clip(score,0,1)."""
    m = 1.0 - k * score.clip(lower=0.0, upper=1.0)
    return w.mul(m, axis=0)


def apply_class_filter(w, signals, classes, k=K):
    """设计 B: 类别级异质乘数 (逐类 1 - k·clip(sig,0,1))."""
    col_cls = {}
    for cls, members in classes.items():
        for c in members:
            col_cls[c] = cls
    w_out = w.copy()
    for key, sig in signals.items():
        m = 1.0 - k * sig.clip(lower=0.0, upper=1.0)
        affected = [c for c in w.columns if col_cls.get(c) in CLASS_MAP.get(key, [])]
        if affected:
            w_out[affected] = w_out[affected].mul(m, axis=0)
    return w_out


def eval_weights(w, label, seg_print=True):
    """重建收益: 成本 + 融资 + G5-lite 现金腿."""
    r = panel.pct_change(fill_method=None)
    port = (w.shift(1) * r).sum(axis=1)
    rate = pd.Series({c: (0.0001 if c.startswith('fut_') else 0.0005)
                      for c in panel.columns})
    port = port - (w.diff().abs() * rate).sum(axis=1).fillna(0.0)
    gross = w.sum(axis=1)
    port = port - (gross.shift(1) - 1.0).clip(lower=0.0) * FIN / ANN
    idle = (1.0 - w.shift(1).sum(axis=1)).clip(lower=0.0)
    port = port + idle * cash_ret.reindex(w.index).ffill().fillna(0.0)
    nav = (1 + port.fillna(0)).cumprod()
    nav = nav[nav.index >= '2019-01-01']  # 剔除渐进入池期

    ret = nav.pct_change(fill_method=None).dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    vol = ret.std() * np.sqrt(ANN)
    sh = ret.mean() * ANN / vol if vol > 0 else np.nan
    dd = (nav / nav.cummax() - 1).min()
    segs = {}
    for name, lo, hi in [('18-20', '2018', '2021'), ('21-23', '2021', '2024'),
                         ('24-26', '2024', '2027'), ('stress', '2025-09', None)]:
        seg = nav.loc[lo:hi]
        rr = seg.pct_change(fill_method=None).dropna()
        v = rr.std() * np.sqrt(ANN)
        segs[name] = (rr.mean() * ANN / v if v > 0 else np.nan,
                      (seg / seg.cummax() - 1).min())
    print(f"{label:<24}{cagr*100:>7.2f}% {sh:>5.2f} {vol*100:>5.1f}% {dd*100:>7.2f}% "
          f"| 18-20 {segs['18-20'][0]:>5.2f}/{segs['18-20'][1]*100:>6.2f}% "
          f"21-23 {segs['21-23'][0]:>5.2f}/{segs['21-23'][1]*100:>6.2f}% "
          f"24-26 {segs['24-26'][0]:>5.2f}/{segs['24-26'][1]*100:>6.2f}% "
          f"stress {segs['stress'][0]:>5.2f}/{segs['stress'][1]*100:>6.2f}%")
    return nav, cagr, sh, vol, dd


def main():
    vt, ml = 0.085, 2.5
    vol, oi = load_volume_oi_panel(panel, cache_path=None)
    CLS = dict(ASSET_CLASSES)
    print(f"基线: 41 资产 D v3 全栈 vt={vt:.1%} ml={ml} (2019 起, 含融资/成本/现金腿)")
    print(f"{'配置':<24}{'CAGR':>8}{'Sh':>6}{'Vol':>6}{'MaxDD':>7} | 分段 Sharpe/MaxDD")

    res = run_d_sleeve(panel, bond_cap=0.40, window=252, rebalance_freq='W-FRI',
                       cov_estimator='lw', neg_trend_mode='trend',
                       neg_trend_preset='balanced', include_rapid=True,
                       momentum_beta=0.20, volume=vol, oi=oi,
                       asset_classes=CLS, cash_ret=cash_ret,
                       target_vol=vt, max_leverage=ml)
    eval_weights(res.weights, '基线 (无宏观过滤)')

    signals = build_signals(res.weights.index)

    # 设计 A: 组合级
    for key in ['dxy', 'vix', 'us10']:
        w_f = apply_portfolio_filter(res.weights, signals[key])
        eval_weights(w_f, f'A: {key} 组合级')
    comp = pd.DataFrame(signals).mean(axis=1)
    eval_weights(apply_portfolio_filter(res.weights, comp), 'A: 三因子组合级')

    # 设计 B: 类别级
    for key in ['dxy', 'vix', 'us10']:
        w_f = apply_class_filter(res.weights, {key: signals[key]}, CLS)
        eval_weights(w_f, f'B: {key} 类别级')
    eval_weights(apply_class_filter(res.weights, signals, CLS), 'B: 三因子类别级')


if __name__ == '__main__':
    main()
