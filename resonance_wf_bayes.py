#!/usr/bin/env python3
# coding=utf-8
"""共振门控: walk-forward 复验 + 贝叶斯信号质量分析.

Part A 贝叶斯:
  - 各状态单元的 P(坏窗|cell) Beta-Binomial 后验 + 95% CI + P(p>基准)
  - 贝叶斯因子 (H1: 信号有信息 vs H0: 与基准同率)
  - 后验概率驱动的仓位 (替代硬阈值)
Part B Walk-forward:
  - 扩张 IS 选参 (最大化 Sharpe) → OOS 逐年评估
  - 参数稳定性 (各折最优参数是否漂移)
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy.stats import beta as Beta
from scipy.special import betaln

import resonance_factor_test as R

ANN = 252


def load(panel_name='REF41'):
    panel, cash_ret = R.load_data()
    ref41 = [c for c in panel.columns if c != 'csi_1000' and c not in R.NEW_CLS]
    if panel_name == 'REF41':
        cols, bond = ref41, None
    else:
        cols, bond = R.P2B, R.P2B_BOND
    ret = panel[cols].pct_change(fill_method=None)
    return panel, cols, bond, ret, cash_ret


def hyst(p, hi, lo, k):
    st = np.ones(len(p))
    on = False
    for i, v in enumerate(p.values):
        if not on and np.isfinite(v) and v >= hi:
            on = True
        elif on and np.isfinite(v) and v <= lo:
            on = False
        st[i] = (1.0 - k) if on else 1.0
    return pd.Series(st, index=p.index)


def metr(nav):
    nav = nav.dropna()
    r = nav.pct_change(fill_method=None).dropna()
    if len(r) < 20:
        return np.nan, np.nan, np.nan
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    vol = r.std() * np.sqrt(ANN)
    return ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1,
            r.mean() * ANN / vol if vol > 0 else np.nan,
            (nav / nav.cummax() - 1).min())


# ── Part A: 贝叶斯 ──

def post(k_bad, n, p0, strength=50):
    """Beta-Binomial 后验 (先验中心 p0, 强度 strength)."""
    a0, b0 = p0 * strength, (1 - p0) * strength
    a, b = a0 + k_bad, b0 + (n - k_bad)
    mean = a / (a + b)
    lo, hi = Beta.ppf([0.025, 0.975], a, b)
    p_gt = 1 - Beta.cdf(p0, a, b)
    return mean, lo, hi, p_gt, a, b


def bayes_factor(k_bad, n, p0):
    """log BF10: H1 (p~Beta(a0,b0) 弱信息) vs H0 (p=p0)."""
    a0, b0 = 1, 1
    log_ml1 = betaln(a0 + k_bad, b0 + n - k_bad) - betaln(a0, b0)
    eps = 1e-9
    pp = min(max(p0, eps), 1 - eps)
    log_ml0 = k_bad * np.log(pp) + (n - k_bad) * np.log(1 - pp)
    return log_ml1 - log_ml0


def bayes_analysis(panel, cols, bond, ret, cash_ret, label):
    base = R.run_pool(panel, cols, bond, cash_ret, 0.20, 3.0)
    nav = R.simulate(base.weights, ret, cash_ret)
    fwd20 = nav.shift(-20) / nav - 1
    bad = fwd20 < fwd20.quantile(0.20)
    p0 = float(bad.dropna().mean())
    print(f"\n=== Part A 贝叶斯: {label} (基准率 p0={p0:.3f}) ===")
    print(f"{'状态单元':<28}{'n':>6}{'P(坏)':>7}{'后验均值':>9}{'95%CI':>18}"
          f"{'P(p>p0)':>9}{'logBF10':>9}")
    ar3 = R.compute_ar3(ret)
    p_ar3 = R.pit_pct(ar3).shift(1)
    dens = R.compute_density(ret)
    p_den = R.pit_pct(dens).shift(1)
    vol20 = nav.pct_change().rolling(20).std()
    p_vol = vol20.expanding(120).rank(pct=True).shift(1)
    csad = (ret.sub(ret.mean(axis=1), axis=0)).abs().mean(axis=1)
    p_csad = R.pit_pct(csad).shift(1)
    cells = {
        'AR3≥P80': (p_ar3 >= 0.80),
        'AR3≥P70': (p_ar3 >= 0.70),
        'AR3≥P80 ∩ 波动≥P80': (p_ar3 >= 0.80) & (p_vol >= 0.80),
        'AR3≥P80 ∩ CSAD≥P80': (p_ar3 >= 0.80) & (p_csad >= 0.80),
        'AR3<P30': (p_ar3 < 0.30),
        '密度≥P80': (p_den >= 0.80),
    }
    idx = bad.dropna().index
    for name, m in cells.items():
        mm = m.reindex(idx).fillna(False)
        n = int(mm.sum())
        if n < 10:
            continue
        kb = int(bad.reindex(idx)[mm].sum())
        mn, lo, hi, pgt, _, _ = post(kb, n, p0)
        bf = bayes_factor(kb, n, p0)
        print(f"{name:<28}{n:>6}{kb/n:>7.3f}{mn:>9.3f}"
              f"{f'[{lo:.3f},{hi:.3f}]':>18}{pgt:>9.3f}{bf:>9.2f}")
    return nav, p_ar3


# ── Part B: Walk-forward ──

def walkforward(panel, cols, bond, ret, cash_ret, label,
                windows=(60, 120), his=(0.60, 0.70, 0.80),
                los=(0.30, 0.50), ks=(0.3, 0.5, 0.7)):
    base = R.run_pool(panel, cols, bond, cash_ret, 0.20, 3.0)
    nav0 = R.simulate(base.weights, ret, cash_ret)
    # 预计算所有参数的门控 NAV (PIT, 可切片)
    navs = {'no_gate': nav0}
    for w in windows:
        p = R.pit_pct(R.compute_ar3(ret, window=w)).shift(1)
        for hi in his:
            for lo in los:
                for k in ks:
                    g = hyst(p, hi, lo, k).reindex(nav0.index).fillna(1.0)
                    navs[(w, hi, lo, k)] = R.simulate(
                        base.weights, ret, cash_ret, gate=g)
    oos_years = ['2022', '2023', '2024', '2025', '2026']
    print(f"\n=== Part B Walk-forward: {label} ===")
    print(f"{'OOS年':>7}{'IS选出参数':>26}{'OOS门控 CAGR/Sh/DD':>30}"
          f"{'OOS基线 CAGR/Sh/DD':>30}")
    sel_params = []
    for y in oos_years:
        is_end = f'{int(y)-1}-12-31'
        oos_a, oos_b = f'{y}-01-01', f'{y}-06-30' if y == '2026' else f'{y}-12-31'
        best, best_sh = None, -99
        for key, nav in navs.items():
            if key == 'no_gate':
                continue
            c, s, d = metr(nav.loc[:is_end] if False else nav[nav.index <= is_end])
            if np.isfinite(s) and s > best_sh:
                best_sh, best = s, key
        if best is None:
            continue
        g_nav = navs[best].loc[oos_a:oos_b]
        b_nav = nav0.loc[oos_a:oos_b]
        gc, gs, gd = metr(g_nav)
        bc, bs, bd = metr(b_nav)
        sel_params.append((y, best, gs, bs, gd, bd))
        print(f"{y:>7}{str(best):>26}"
              f"{f'{gc*100:5.1f}%/{gs:4.2f}/{gd*100:6.1f}%':>30}"
              f"{f'{bc*100:5.1f}%/{bs:4.2f}/{bd*100:6.1f}%':>30}")
    # 汇总
    if sel_params:
        gs = np.mean([p[2] for p in sel_params])
        bs = np.mean([p[3] for p in sel_params])
        gd = np.mean([p[4] for p in sel_params])
        bd = np.mean([p[5] for p in sel_params])
        print(f"{'均值':>7}{'':>26}{f'Sh {gs:.2f} (基线 {bs:.2f})':>30}"
              f"{f'DD {gd*100:.1f}% (基线 {bd*100:.1f}%)':>30}")
    return sel_params


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else 'both'
    for name in ('REF41', 'P2b'):
        panel, cols, bond, ret, cash_ret = load(name)
        if which in ('a', 'both'):
            bayes_analysis(panel, cols, bond, ret, cash_ret, name)
        if which in ('b', 'both'):
            walkforward(panel, cols, bond, ret, cash_ret, name)


if __name__ == '__main__':
    main()
