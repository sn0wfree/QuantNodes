#!/usr/bin/env python3
# coding=utf-8
"""贝叶斯后验概率驱动的连续仓位 (替代二值门控).

核心设计 (因果/PIT):
  state_t = (AR3 分位桶, 组合波动分位桶)   5×5 网格
  outcome = 未来20日收益处于扩张P20以下 (坏窗), 于 t+20 可观测
  随 t 前进, 把已实现 (state_j, bad_j), j<=t-20 计入对应桶
  后验: Beta(p0·s + k_bad, (1-p0)·s + n - k_bad), p0=扩张基准率
  仓位乘数 m_t = 1 - k·driver_t, 两驱动量:
    A 后验均值超额: max(0, (post_mean - p0)/(1-p0))
    B 贝叶斯置信度: max(0, 2·(P(p>p0) - 0.5))
  driver B 自动压低"证据薄弱单元"的动作 — 直接把噪声单元归零
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy.stats import beta as Beta

import resonance_factor_test as R

ANN = 252
HORIZON = 20


def prep(nav, ret, ar3_window=60, n_buckets=5, thr_win=250):
    """构造 state 桶 + 坏窗标签 (全部 PIT)."""
    ar3 = R.compute_ar3(ret, window=ar3_window)
    p_ar3 = R.pit_pct(ar3).shift(1).reindex(nav.index)
    vol20 = nav.pct_change().rolling(20).std()
    p_vol = vol20.expanding(120).rank(pct=True).shift(1).reindex(nav.index)
    b_ar = np.minimum((p_ar3 * n_buckets).fillna(-1).astype(int), n_buckets - 1)
    b_vo = np.minimum((p_vol * n_buckets).fillna(-1).astype(int), n_buckets - 1)
    fwd = nav.shift(-HORIZON) / nav - 1
    thr = fwd.expanding(min_periods=thr_win).quantile(0.20).shift(1)
    bad = (fwd < thr)
    p_base = bad.expanding(min_periods=thr_win).mean().shift(1)
    return b_ar, b_vo, bad, p_base


def bayes_gate(nav, b_ar, b_vo, bad, p_base, n_buckets=5, strength=30,
               k=0.7, driver='B', horizon=HORIZON):
    """在线贝叶斯后验 → 连续仓位乘数 (因果)."""
    idx = nav.index
    n = len(idx)
    cells = n_buckets * n_buckets
    cnt = np.zeros(cells)      # 观测数
    cnt_bad = np.zeros(cells)  # 坏窗数
    m = np.ones(n)
    for i in range(n):
        # 1) 结算 j = i - horizon 的结果 (已实现)
        j = i - horizon
        if j >= 0:
            a, v = b_ar.iloc[j], b_vo.iloc[j]
            if a >= 0 and v >= 0 and np.isfinite(bad.iloc[j]):
                c = int(a) * n_buckets + int(v)
                cnt[c] += 1
                if bad.iloc[j]:
                    cnt_bad[c] += 1
        # 2) 当前状态桶
        a, v = b_ar.iloc[i], b_vo.iloc[i]
        if a < 0 or v < 0:
            continue
        c = int(a) * n_buckets + int(v)
        nb, ntot = cnt_bad[c], cnt[c]
        if ntot < 10:
            continue
        p0 = p_base.iloc[i]
        if not np.isfinite(p0) or p0 <= 0 or p0 >= 1:
            continue
        a_post = p0 * strength + nb
        b_post = (1 - p0) * strength + (ntot - nb)
        if driver == 'A':
            post_mean = a_post / (a_post + b_post)
            d = max(0.0, (post_mean - p0) / (1 - p0))
        else:  # 'B' 置信度
            p_gt = 1 - Beta.cdf(p0, a_post, b_post)
            d = max(0.0, 2.0 * (p_gt - 0.5))
        m[i] = 1.0 - k * min(d, 1.0)
    return pd.Series(m, index=idx)


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


def main():
    panel, cash_ret = R.load_data()
    ref41 = [c for c in panel.columns if c != 'csi_1000' and c not in R.NEW_CLS]
    ret = panel[ref41].pct_change(fill_method=None)
    base = R.run_pool(panel, ref41, None, cash_ret, 0.20, 3.0)
    nav0 = R.simulate(base.weights, ret, cash_ret)
    c0, s0, d0 = metr(nav0)
    print(f"REF41 基线: CAGR {c0*100:.2f}% Sh {s0:.2f} MaxDD {d0*100:.2f}%")

    for nb in (3, 5):
        b_ar, b_vo, bad, p_base = prep(nav0, ret, n_buckets=nb)
        for strength in (10, 30, 60):
            for driver in ('A', 'B'):
                for k in (0.5, 0.7):
                    m = bayes_gate(nav0, b_ar, b_vo, bad, p_base,
                                   n_buckets=nb, strength=strength,
                                   k=k, driver=driver)
                    nav = R.simulate(base.weights, ret, cash_ret, gate=m)
                    c, s, d = metr(nav)
                    turn = m.diff().abs().mean()
                    print(f"桶{nb}×{nb} s={strength:<3} 驱动{driver} k={k}: "
                          f"CAGR {c*100:>6.2f}% Sh {s:>5.2f} MaxDD {d*100:>7.2f}% "
                          f"仓位换手 {turn:.4f}")


if __name__ == '__main__':
    main()
