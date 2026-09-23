#!/usr/bin/env python3
# coding=utf-8
"""完整版 Boons-Prado basis momentum 验证 (§5.49b).

原始定义 (Journal of Finance 2019):
  BM_t = r_near(12m) − r_far(12m)
  其中 near/far = t 时刻按到期日排序的第 1/第 2 近月合约,
  r = 该特定合约过去 12 个月的累计收益.

数学等价: BM_t = log(P1_t/P1_{t-12}) − log(P2_t/P2_{t-12})
               = Δ log(P_near/P_far) over 12m  (对固定合约对)

结论 (2026-09-23): 中国商品期货合约上市提前期仅 ~12 个月, t 时刻近月合约
不存在 12 个月前价格 → 逐合约 BM **无法构造** (本脚本在真实数据上仅得
个位数交易日, 合成数据已复现该约束). 经济等价的可行实现为 Δcarry(12m),
已测有效但**与 carry 冗余** (相关 0.677, 组合不加分). 详见报告 §5.49b.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, '.')
import numpy as np                                          # noqa: E402
import pandas as pd                                         # noqa: E402

from scripts.extract_carry import PANEL_FUTS, _expiry, _query  # noqa: E402

REPO = Path(__file__).resolve().parent
CACHE = REPO / "data/allweather/bm_cache"
HORIZON = 365          # 日历日 ≈ 12 个月


def proper_bm(df: pd.DataFrame, horizon: int = HORIZON) -> pd.Series:
    """单产品完整版 basis momentum (逐合约 12 月动量差)."""
    df = df.copy()
    df["exp"] = df["Symbol"].map(_expiry)
    df = df.dropna(subset=["exp"])
    df["TradeDate"] = pd.to_datetime(df["TradeDate"])
    px = df.pivot_table(index="TradeDate", columns="Symbol",
                        values="Close", aggfunc="last").sort_index()
    exp = df.groupby("Symbol")["exp"].first()
    idx = px.index
    out = {}
    for t in idx:
        row = px.loc[t].dropna()
        if len(row) < 2:
            continue
        e = exp.reindex(row.index).dropna().sort_values()
        if len(e) < 2:
            continue
        near, far = e.index[0], e.index[1]
        prior = idx[idx <= t - pd.Timedelta(days=horizon)]
        if len(prior) == 0:
            continue
        t0 = prior[-1]
        p1t, p10 = px.at[t, near], px.at[t0, near]
        p2t, p20 = px.at[t, far], px.at[t0, far]
        if pd.notna(p10) and pd.notna(p20) and p10 > 0 and p20 > 0 \
                and pd.notna(p1t) and pd.notna(p2t):
            out[t] = (p1t / p10 - 1.0) - (p2t / p20 - 1.0)
    s = pd.Series(out).sort_index()
    return s


def one(item):
    col, view = item
    cp = CACHE / f"{col}.parquet"
    if cp.exists():
        s = pd.read_parquet(cp).iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return col, s
    df = _query(f"SELECT TradeDate, Symbol, Close, OpenInterest "
                f"FROM daily_quote.v_{view}_daily "
                f"WHERE Close > 0 AND OpenInterest > 0")
    s = proper_bm(df)
    pd.DataFrame({col: s}).to_parquet(cp)
    return col, s


def metr(nav):
    nav = nav.dropna()
    r = nav.pct_change(fill_method=None).dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    v = r.std() * np.sqrt(252)
    return ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1,
            r.mean() * 252 / v, v, (nav / nav.cummax() - 1).min())


def main():
    import alpha_factors_test as A
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(one, kv): kv[0] for kv in PANEL_FUTS.items()}
        for f in as_completed(futs):
            col = futs[f]
            try:
                c, s = f.result()
                if len(s):
                    frames[c] = s
                    print(f"  {c}: {len(s)} 天, BM 均值 {s.mean()*100:+.2f}%",
                          flush=True)
            except Exception as e:
                print(f"  {col}: ✗ {type(e).__name__}: {str(e)[:60]}",
                      flush=True)
    bm = pd.DataFrame(frames).sort_index()
    bm.to_parquet(REPO / "data/allweather/basis_mom_daily.parquet")
    print(f"\n完整版 BM: {bm.shape}")

    panel = pd.read_parquet('data/allweather/panel_v5.parquet').sort_index()
    panel.index = pd.to_datetime(panel.index)
    carry = pd.read_parquet('data/allweather/carry_daily.parquet')
    carry.index = pd.to_datetime(carry.index)
    P = panel.loc[:'2026-06-30'][[c for c in panel.columns if c not in A.NEW]]

    # 代理 (Δcarry 252)
    proxy = carry - carry.shift(252)
    s_bm = A.sleeve_from_signal(P, bm)
    s_proxy = A.sleeve_from_signal(P, proxy)
    s_carry = A.sleeve_from_signal(P, carry)
    j = s_bm.index.intersection(s_proxy.index).intersection(s_carry.index)
    print(f"\n完整版 BM  vs Δcarry(252) 代理: {s_bm.reindex(j).corr(s_proxy.reindex(j)):.3f}")
    print(f"完整版 BM  vs carry(level):     {s_bm.reindex(j).corr(s_carry.reindex(j)):.3f}")
    print(f"Δcarry(252) vs carry(level):    {s_proxy.reindex(j).corr(s_carry.reindex(j)):.3f}")
    for nm, s in (('完整版 BM', s_bm), ('Δcarry(252)', s_proxy),
                  ('carry', s_carry)):
        c, sh, v, d = metr((1 + s).cumprod())
        print(f"  {nm:<14} CAGR {c*100:>6.2f}% Sh {sh:>5.2f} "
              f"Vol {v*100:>5.2f}% MaxDD {d*100:>6.2f}%")


if __name__ == '__main__':
    main()
