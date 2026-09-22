#!/usr/bin/env python3
# coding=utf-8
"""共振因子 → 回撤优化实验 (共振因子清单 §B1/B2 移植到本项目).

信号 (PIT: 扩张分位, shift(1)):
  AR3    吸收比率 = 60日相关阵前3主成分解释方差比 (Kritzman)
  密度    60日 |rho|>=0.35 无向边密度
应用: 组合级阈值门控降杠杆 m = 1 - k*max(0,(pct-theta)/(1-theta))
主战场: P2b vt=20% ml=3.0 (回撤痛点) + REF41 对照
诚实门: 同 MaxDD 下 CAGR 是否高于"直接降 vt"对照臂
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
END = '2026-06-30'

NEW_CLS = {
    'etf_510500': 'equity_etf_dom', 'etf_588000': 'equity_etf_dom',
    'etf_159920': 'equity_etf_off', 'etf_513520': 'equity_etf_off',
    'etf_511260': 'bond_rate', 'etf_159985': 'agri_oilseed',
    'csi_1000': 'equity_etf_dom',
}
P2B = ['fut_t', 'fut_tf', 'fut_ts', 'etf_511260',
       'etf_510300', 'etf_510500', 'etf_588000', 'etf_159915', 'csi_1000',
       'etf_513100', 'etf_513500', 'etf_513520', 'etf_159920',
       'fut_au', 'fut_cu', 'fut_sc', 'fut_m', 'etf_159985']
P2B_BOND = ['fut_t', 'fut_tf', 'fut_ts', 'etf_511260']


def load_data():
    panel = pd.read_parquet('data/allweather/panel_v5.parquet').sort_index()
    panel.index = pd.to_datetime(panel.index)
    v7 = pd.read_parquet('data/high_freq_macro/v7_index_v2.parquet')
    cash = v7['中债1-3年国债财富指数'].dropna()
    cash.index = pd.to_datetime(cash.index)
    cash_ret = cash.pct_change(fill_method=None)
    csi = v7['中证1000'].dropna()
    csi.index = pd.to_datetime(csi.index)
    panel['csi_1000'] = csi.reindex(panel.index).ffill()
    return panel.loc[:END], cash_ret


def make_classes(cols):
    cls = {k: [c for c in v if c in cols] for k, v in ASSET_CLASSES.items()}
    for c, k in NEW_CLS.items():
        if c in cols and c not in [x for v in cls.values() for x in v]:
            cls.setdefault(k, []).append(c)
    return {k: v for k, v in cls.items() if v}


def compute_ar3(ret, window=60, min_frac=0.5):
    """吸收比率: 60日相关阵前3主成分解释方差占比 (Kritzman)."""
    out = {}
    for i, d in enumerate(ret.index):
        if i < window:
            out[d] = np.nan
            continue
        sub = ret.iloc[i - window:i].dropna(axis=1, how='any')
        if sub.shape[1] < 3 or sub.shape[0] < max(30, sub.shape[1] * min_frac):
            out[d] = np.nan
            continue
        c = np.nan_to_num(sub.corr().to_numpy(), nan=0.0)
        np.fill_diagonal(c, 1.0)
        ev = np.clip(np.linalg.eigvalsh(c)[::-1], 0, None)
        out[d] = float(ev[:3].sum() / ev.sum()) if ev.sum() > 0 else np.nan
    return pd.Series(out).reindex(ret.index)


def compute_density(ret, window=60, thr=0.35):
    """网络密度: 60日 |rho|>=thr 无向边占比."""
    out = {}
    for i, d in enumerate(ret.index):
        if i < window:
            out[d] = np.nan
            continue
        sub = ret.iloc[i - window:i].dropna(axis=1, how='any')
        n = sub.shape[1]
        if n < 3:
            out[d] = np.nan
            continue
        c = np.abs(sub.corr().to_numpy())
        iu = np.triu_indices(n, k=1)
        out[d] = float((c[iu] >= thr).mean())
    return pd.Series(out).reindex(ret.index)


def pit_pct(s, min_periods=120):
    """扩张分位 (PIT, 无前视)."""
    return s.expanding(min_periods=min_periods).rank(pct=True)


def resonance_days(ret, window=252, p90=0.90, min_frac=0.5, same_dir=0.70):
    """共振日: 极端品种占比>=50% 且同向占比>=70% (分位 shift(1))."""
    absr = ret.abs()
    thr = absr.rolling(window, min_periods=60).quantile(p90).shift(1)
    extreme = absr > thr
    active = ret.notna()
    n_active = active.sum(axis=1).clip(lower=1)
    frac_ext = (extreme & active).sum(axis=1) / n_active
    pos_ext = ((ret > 0) & extreme & active).sum(axis=1)
    neg_ext = ((ret < 0) & extreme & active).sum(axis=1)
    tot_ext = (pos_ext + neg_ext).clip(lower=1)
    frac_same = pd.concat([pos_ext, neg_ext], axis=1).max(axis=1) / tot_ext
    is_res = (frac_ext >= min_frac) & (frac_same >= same_dir) & (extreme.sum(axis=1) > 0)
    return pd.DataFrame({'frac_ext': frac_ext, 'frac_same': frac_same,
                         'resonance': is_res})


def gate_from_pct(pct, k, theta):
    """阈值门控: m = 1 - k*max(0,(pct-theta)/(1-theta)); shift(1) 防前视."""
    p = pit_pct(pct).shift(1)
    eff = ((p - theta) / max(1 - theta, 1e-6)).clip(lower=0, upper=1)
    return (1.0 - k * eff).fillna(1.0)


def simulate(w_base, ret, cash_ret, gate=None):
    """门控缩放权重 → 组合收益 (名义口径 + 融资2% + 现金腿)."""
    w = w_base.copy()
    if gate is not None:
        m = gate.reindex(w.index).ffill().fillna(1.0).clip(0, 1)
        w = w.mul(m, axis=0)
    rate = pd.Series({c: (0.0001 if c.startswith('fut_') else 0.0005)
                      for c in w.columns})
    port = (w.shift(1) * ret).sum(axis=1)
    port -= (w.diff().abs() * rate).sum(axis=1).fillna(0.0)
    port -= (w.sum(axis=1).shift(1) - 1.0).clip(lower=0.0) * FIN / ANN
    idle = (1.0 - w.sum(axis=1).shift(1)).clip(lower=0.0)
    port += idle * cash_ret.reindex(w.index).ffill().fillna(0.0)
    nav = (1 + port.fillna(0)).cumprod()
    return nav[nav.index >= '2019-01-01']


def metrics(nav):
    nav = nav.dropna()
    r = nav.pct_change(fill_method=None).dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(ANN)
    dd = (nav / nav.cummax() - 1).min()
    return {'CAGR': cagr, 'Sharpe': r.mean() * ANN / vol if vol > 0 else np.nan,
            'Vol': vol, 'MaxDD': dd,
            'Calmar': cagr / abs(dd) if dd < 0 else np.nan}


def run_pool(panel, cols, bond_cols, cash_ret, vt, ml=3.0):
    sub = panel[cols]
    cls = make_classes(cols)
    vol, oi = load_volume_oi_panel(sub, cache_path=None)
    return run_d_sleeve(sub, bond_cap=0.40, window=252, rebalance_freq='W-FRI',
                        cov_estimator='lw', neg_trend_mode='trend',
                        neg_trend_preset='balanced', include_rapid=True,
                        momentum_beta=0.20, volume=vol, oi=oi,
                        asset_classes=cls, cash_ret=cash_ret,
                        target_vol=vt, max_leverage=ml, bond_cols=bond_cols)


def main():
    panel, cash_ret = load_data()
    ref41 = [c for c in panel.columns if c != 'csi_1000' and c not in NEW_CLS]
    print(f"窗口: {panel.index[0].date()} → {panel.index[-1].date()}")

    # ── 信号 + 效标 (P2b 池) ──
    ret_p2b = panel[P2B].pct_change(fill_method=None)
    ar3 = compute_ar3(ret_p2b)
    dens = compute_density(ret_p2b)
    comp = pit_pct(ar3) + pit_pct(dens)  # 合成 (各自 PIT 后相加)
    res_ev = resonance_days(ret_p2b)
    n_res = int(res_ev['resonance'].sum())
    print(f"\n共振日 (P2b 效标): {n_res} 日 ({n_res/len(res_ev)*100:.2f}%)")
    print(f"AR3 均值 {ar3.mean():.3f} | 密度均值 {dens.mean():.3f}")
    # AR3 对共振日的覆盖率 (P80/P90 阈值)
    for th in (0.80, 0.90):
        trig = pit_pct(ar3).shift(1) >= th
        cov = (trig & res_ev['resonance']).sum() / max(n_res, 1)
        print(f"  AR3>={th:.0%} 触发 {(trig).mean()*100:.0f}% 时间, 覆盖共振日 {cov*100:.0f}%")
    # 共振日 vs 非共振日的 AR3 分位
    rp = pit_pct(ar3).shift(1)
    print(f"  共振日 AR3 分位 {rp[res_ev['resonance']].mean():.2f} "
          f"vs 非共振 {rp[~res_ev['resonance']].mean():.2f}")
    rpn = pit_pct(dens).shift(1)
    print(f"  共振日 密度分位 {rpn[res_ev['resonance']].mean():.2f} "
          f"vs 非共振 {rpn[~res_ev['resonance']].mean():.2f}")

    # ── 基线 + 对照臂 ──
    print(f"\n{'='*88}")
    print("基线 vs 对照臂 (P2b, 名义口径+融资2%)")
    print(f"{'配置':<30}{'CAGR':>8}{'Sharpe':>7}{'Vol':>8}{'MaxDD':>8}{'Calmar':>7}")
    rows = []

    def show(lbl, nav):
        m = metrics(nav)
        print(f"{lbl:<30}{m['CAGR']*100:>7.2f}%{m['Sharpe']:>7.2f}"
              f"{m['Vol']*100:>7.2f}%{m['MaxDD']*100:>7.2f}%{m['Calmar']:>7.2f}")
        rows.append({'config': lbl, **m})
        return m

    base20 = run_pool(panel, P2B, P2B_BOND, cash_ret, 0.20, 3.0)
    show('P2b vt=20% (基线)', simulate(base20.weights, ret_p2b, cash_ret))
    for vt in (0.09, 0.10, 0.11, 0.12):
        r = run_pool(panel, P2B, P2B_BOND, cash_ret, vt, 2.5)
        show(f'对照: vt={vt:.0%} (无风控)', simulate(r.weights, ret_p2b, cash_ret))

    # ── 门控网格 ──
    print(f"\n{'='*88}")
    print("共振门控网格 (P2b vt=20% 基线 + 门控)")
    print(f"{'信号/参数':<30}{'CAGR':>8}{'Sharpe':>7}{'Vol':>8}{'MaxDD':>8}{'Calmar':>7}")
    for sig_name, sig in [('AR3', ar3), ('密度', dens), ('合成', comp)]:
        pct = sig if sig_name == '合成' else sig
        for theta in (0.70, 0.80, 0.90):
            for k in (0.3, 0.5, 0.7):
                g = gate_from_pct(pct, k, theta)
                nav = simulate(base20.weights, ret_p2b, cash_ret, gate=g)
                show(f'{sig_name} θ={theta:.0%} k={k}', nav)

    out = pd.DataFrame(rows)
    out.to_csv('data/allweather/resonance_overlay_test.csv', index=False)
    print("\n已保存: data/allweather/resonance_overlay_test.csv")

    # ── REF41 对照 (P1-B 有效场景, 看 AR3 是否同向) ──
    print(f"\n{'='*88}")
    print("REF41 对照 (AR3 门控, vt=20%)")
    ret_ref = panel[ref41].pct_change(fill_method=None)
    ar3_ref = compute_ar3(ret_ref)
    base_ref = run_pool(panel, ref41, None, cash_ret, 0.20, 3.0)
    show('REF41 vt=20% 基线', simulate(base_ref.weights, ret_ref, cash_ret))
    for theta in (0.70, 0.80, 0.90):
        for k in (0.3, 0.5, 0.7):
            g = gate_from_pct(ar3_ref, k, theta)
            show(f'REF41 AR3 θ={theta:.0%} k={k}',
                 simulate(base_ref.weights, ret_ref, cash_ret, gate=g))


if __name__ == '__main__':
    main()
