#!/usr/bin/env python3
# coding=utf-8
"""生产候选验证: 前视测试 (look-ahead) + 过拟合测试 (overfitting).

前视:
  A1 信号级因果重放 — 用 ≤t 数据重算 AR3/vol/dd/门控, 与全run 比对
  A2 管道级因果重放 — panel 截断到 t 重跑, 权重与全run 比对
  A3 未来信息对照 (sanity) — "偷看"版门控应远优于 PIT 版
过拟合:
  B1 参数高原 — 候选点邻域网格 (尖峰 vs 高原)
  B2 门控时序置换 (circular shift placebo) — 分布 + p 值
  B3 块 bootstrap — Sharpe 改善的 95% CI
  B4 子样本稳定性 — 前后半段
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.d_pipeline import (
    run_d_sleeve, _absorption_ratio, _pit_pct, apply_resonance_gate)
from QuantNodes.strategy.allweather.volume_loader import load_volume_oi_panel
from QuantNodes.strategy.allweather.class_erc import ASSET_CLASSES

ANN = 252
SEED = 20260906


def load():
    panel = pd.read_parquet('data/allweather/panel_v5.parquet').sort_index()
    panel.index = pd.to_datetime(panel.index)
    v7 = pd.read_parquet('data/high_freq_macro/v7_index_v2.parquet')
    cash = v7['中债1-3年国债财富指数'].dropna()
    cash.index = pd.to_datetime(cash.index)
    cash_ret = cash.pct_change(fill_method=None)
    new = ('etf_510500', 'etf_588000', 'etf_159920', 'etf_513520',
           'etf_511260', 'etf_159985', 'csi_1000')
    ref41 = [c for c in panel.columns if c not in new]
    return panel.loc[:'2026-06-30'][ref41], cash_ret


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


def run(panel, vol, oi, cash_ret, **kw):
    base = dict(panel=panel, bond_cap=0.40, window=252,
                rebalance_freq='W-FRI', cov_estimator='lw',
                neg_trend_mode='trend', neg_trend_preset='balanced',
                include_rapid=True, momentum_beta=0.20, volume=vol, oi=oi,
                asset_classes=dict(ASSET_CLASSES), cash_ret=cash_ret,
                target_vol=0.15, max_leverage=2.5, financing_rate=0.02)
    base.update(kw)
    return run_d_sleeve(**base)


# ── A1 信号级因果重放 ──
def test_a1_signal_replay(panel, cash_ret, n_check=40):
    print("\n=== A1 信号级因果重放 (AR3/vol/dd/门控) ===")
    ret = panel.pct_change(fill_method=None)
    ar3_full = _absorption_ratio(ret, 60)
    p_full = _pit_pct(ar3_full).shift(1)
    rng = np.random.default_rng(SEED)
    idx = p_full.dropna().index
    picks = rng.choice(np.arange(60, len(idx) - 1), size=n_check, replace=False)
    diffs = []
    for i in sorted(picks):
        t = idx[i]
        r_trunc = ret.loc[:t]
        a_trunc = _absorption_ratio(r_trunc, 60)
        p_trunc = _pit_pct(a_trunc).shift(1)
        if np.isfinite(p_full.loc[t]) and np.isfinite(p_trunc.loc[t]):
            diffs.append(abs(p_full.loc[t] - p_trunc.loc[t]))
    diffs = np.array(diffs)
    print(f"  样本 {len(diffs)} 个: AR3 分位 |Δ| max={diffs.max():.2e} "
          f"mean={diffs.mean():.2e}  → {'✅ 无前视' if diffs.max()<1e-9 else '❌ 存在差异'}")
    return diffs.max() < 1e-9


# ── A2 管道级因果重放 ──
def test_a2_pipeline_replay(panel, vol, oi, cash_ret, n_check=6):
    print("\n=== A2 管道级因果重放 (截断重跑 vs 全run 权重) ===")
    full = run(panel, vol, oi, cash_ret, correlation_k=1.0,
               correlation_threshold=0.60, resonance_k=0.7)
    w_full = full.weights
    # 选调仓日 (权重发生变化的日期) 作为检查点
    chg = w_full.diff().abs().sum(axis=1)
    rb_dates = chg[chg > 1e-9].index
    rng = np.random.default_rng(SEED)
    picks = [rb_dates[i] for i in
             np.sort(rng.choice(np.arange(30, len(rb_dates) - 1),
                                size=n_check, replace=False))]
    print(f"  {'日期':>12}{'权重 |Δ| max':>16}{'判定':>8}")
    ok = True
    for t in picks:
        sub = panel.loc[:t]
        v2, o2 = load_volume_oi_panel(sub, cache_path=None)
        res = run(sub, v2, o2, cash_ret, correlation_k=1.0,
                  correlation_threshold=0.60, resonance_k=0.7)
        w_t = res.weights.loc[t] if t in res.weights.index else None
        if w_t is None:
            continue
        d = (w_t - w_full.loc[t]).abs().max()
        ok = ok and (d < 1e-9)
        print(f"  {str(t.date()):>12}{d:>16.2e}{'✅' if d<1e-9 else '❌':>8}")
    print(f"  结论: {'✅ 无前视 (截断重跑权重一致)' if ok else '❌ 存在前视'}")
    return ok


# ── A3 未来信息对照 ──
def test_a3_peeking(panel, vol, oi, cash_ret):
    print("\n=== A3 未来信息对照 (sanity) ===")
    pre = run(panel, vol, oi, cash_ret, correlation_k=1.0,
              correlation_threshold=0.60, resonance_k=0.0)
    w = pre.weights
    ret = panel.pct_change(fill_method=None).reindex(w.index)
    # "偷看"版: 用未来 20 日收益的正负决定门控 (故意前视)
    fwd = ret.rolling(20).sum().shift(-20).mean(axis=1)
    peek = (1.0 - 0.7 * (fwd < 0).astype(float)).shift(1).fillna(1.0)
    nav_peek = _sim(w, ret, cash_ret, peek)
    # PIT 版
    res = run(panel, vol, oi, cash_ret, correlation_k=1.0,
              correlation_threshold=0.60, resonance_k=0.7)
    nav_pit = res.nav
    for lbl, nav in [('PIT 门控 (真实)', nav_pit), ('偷看未来 (sanity)', nav_peek)]:
        c, s, d = metr(nav[nav.index >= '2019-01-01'])
        print(f"  {lbl:<20} CAGR {c*100:>6.2f}% Sh {s:>5.2f} MaxDD {d*100:>7.2f}%")
    print("  说明: 偷看版显著更优 → 测试可检测前视; PIT 版与其差距 = 信号的真实价值")


def _sim(w, ret, cash_ret, gate=None):
    ww = w.copy()
    if gate is not None:
        ww = ww.mul(gate.reindex(ww.index).ffill().fillna(1.0).clip(0, 1), axis=0)
    rate = pd.Series({c: (0.0001 if c.startswith('fut_') else 0.0005)
                      for c in ww.columns})
    port = (ww.shift(1) * ret).sum(axis=1)
    port -= (ww.diff().abs() * rate).sum(axis=1).fillna(0)
    port -= (ww.sum(axis=1).shift(1) - 1).clip(lower=0) * 0.02 / ANN
    port += (1 - ww.sum(axis=1).shift(1)).clip(lower=0) \
        * cash_ret.reindex(ww.index).ffill().fillna(0)
    return (1 + port.fillna(0)).cumprod()


# ── B 过拟合 ──
def test_b(panel, vol, oi, cash_ret):
    print("\n=== B1 参数高原 (候选点邻域) ===")
    grid = []
    for ck in (0.5, 1.0):
        for rk in (0.5, 0.7):
            for fl in (-0.03, -0.05):
                res = run(panel, vol, oi, cash_ret, correlation_k=ck,
                          correlation_threshold=0.60, resonance_k=rk,
                          resonance_dd_floor=fl)
                c, s, d = metr(res.nav[res.nav.index >= '2019-01-01'])
                grid.append((ck, rk, fl, c, s, d))
                print(f"  corr_k={ck} res_k={rk} dd_floor={fl}: "
                      f"CAGR {c*100:>6.2f}% Sh {s:>5.2f} MaxDD {d*100:>7.2f}%")
    shs = [g[4] for g in grid]
    print(f"  邻域 Sharpe: min {min(shs):.2f} max {max(shs):.2f} "
          f"→ {'✅ 高原' if min(shs) > 1.0 else '⚠️ 部分邻域弱'}")

    print("\n=== B2 门控时序置换 (circular shift placebo, N=30) ===")
    pre = run(panel, vol, oi, cash_ret, correlation_k=1.0,
              correlation_threshold=0.60, resonance_k=0.0)
    w = pre.weights
    ret = panel.pct_change(fill_method=None).reindex(w.index)
    g_real = apply_resonance_gate(w, ret, cash_ret, window=60, enter=0.70,
                                  exit_th=0.40, vol_exit=0.50,
                                  dd_floor=-0.03, k=0.7)
    m_real = (g_real.iloc[:, 0] / w.iloc[:, 0].replace(0, np.nan)).fillna(1.0)
    m_real = m_real.clip(0, 1)
    c, s_real, d = metr(_sim(w, ret, cash_ret, m_real)[
        _sim(w, ret, cash_ret, m_real).index >= '2019-01-01'])
    rng = np.random.default_rng(SEED)
    dist = []
    for _ in range(30):
        lag = int(rng.integers(60, len(m_real) - 61))
        m_sh = m_real.shift(lag).fillna(1.0)
        nav = _sim(w, ret, cash_ret, m_sh)
        _, s2, _ = metr(nav[nav.index >= '2019-01-01'])
        dist.append(s2)
    dist = np.array(dist)
    p = (dist >= s_real).mean()
    print(f"  真实 Sharpe {s_real:.2f} | 置换分布 mean {dist.mean():.2f} "
          f"P95 {np.percentile(dist,95):.2f}")
    print(f"  p 值 (置换 ≥ 真实) = {p:.3f} → "
          f"{'✅ 显著 (p<0.05)' if p<0.05 else '⚠️ 不显著'}")

    print("\n=== B3 块 bootstrap (Sharpe 改善 95% CI, N=200) ===")
    nav_full = run(panel, vol, oi, cash_ret, correlation_k=1.0,
                   correlation_threshold=0.60, resonance_k=0.7).nav
    nav_base = run(panel, vol, oi, cash_ret).nav
    rf = nav_full.pct_change().dropna()
    rb = nav_base.pct_change().dropna()
    common = rf.index.intersection(rb.index)
    rf, rb = rf.loc[common], rb.loc[common]
    rng = np.random.default_rng(SEED)
    n, blk = len(rf), 20
    diffs = []
    for _ in range(200):
        starts = rng.integers(0, n - blk, size=n // blk + 1)
        idx = np.concatenate([np.arange(s, s + blk) for s in starts])[:n]
        a, b = rf.iloc[idx], rb.iloc[idx]
        sa = a.mean() * ANN / (a.std() * np.sqrt(ANN))
        sb = b.mean() * ANN / (b.std() * np.sqrt(ANN))
        diffs.append(sa - sb)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  ΔSharpe 均值 {diffs.mean():.3f}, 95% CI [{lo:.3f}, {hi:.3f}] → "
          f"{'✅ CI 不含 0' if lo > 0 else '⚠️ CI 含 0'}")

    print("\n=== B4 子样本稳定性 ===")
    for lbl, a, b in [('前半 2019-2022', '2019-01-01', '2022-12-31'),
                      ('后半 2023-2026', '2023-01-01', '2026-06-30')]:
        cf, sf, df = metr(nav_full.loc[a:b])
        cb, sb, db = metr(nav_base.loc[a:b])
        print(f"  {lbl}: 组合 Sh {sf:.2f}/DD {df*100:.1f}%  "
              f"vs 基线 Sh {sb:.2f}/DD {db*100:.1f}%")


def main():
    panel, cash_ret = load()
    vol, oi = load_volume_oi_panel(panel, cache_path=None)
    test_a1_signal_replay(panel, cash_ret)
    test_a2_pipeline_replay(panel, vol, oi, cash_ret)
    test_a3_peeking(panel, vol, oi, cash_ret)
    test_b(panel, vol, oi, cash_ret)


if __name__ == '__main__':
    main()
