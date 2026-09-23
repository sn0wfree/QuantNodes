#!/usr/bin/env python3
# coding=utf-8
"""提取期货期权 IV 曲面特征 — §5.53 (期限结构 / 偏度).

每个产品每日输出 (Black-76 反解):
  iv_near : 最近到期 (T>7d) 的 ATM IV
  iv_far  : 次近到期的 ATM IV
  iv_put  : 近月 0.95×F 看跌 IV
  iv_call : 近月 1.05×F 看涨 IV
派生: IVTS = iv_far − iv_near (波动率期限结构斜率)
      skew = iv_put − iv_call (偏度)
输出: data/allweather/iv_surface/<col>.parquet
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.extract_carry import _expiry  # noqa: E402
from scripts.extract_iv import (  # noqa: E402
    CACHE, IV_PRODS, _query, black76_iv, parse_option)

SURF = REPO / "data/allweather/iv_surface"
FUT_RE = r"^[A-Za-z]{1,2}[0-9]{3,4}$"
MIN_T = 7 / 365


def compute_surface(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TradeDate"] = pd.to_datetime(df["TradeDate"])
    is_fut = df["Symbol"].str.match(FUT_RE)
    fut = df[is_fut].copy()
    fut["exp"] = fut["Symbol"].map(_expiry)
    fut = fut.dropna(subset=["exp"])
    fmap = fut.set_index(["TradeDate", "exp"])["Close"]

    opt = df[~is_fut].copy()
    parsed = [parse_option(s) for s in opt["Symbol"]]
    keep = [x is not None for x in parsed]
    opt = opt[keep]
    if opt.empty:
        return pd.DataFrame()
    opt = pd.concat([opt, pd.DataFrame(
        [x for x in parsed if x is not None],
        columns=["prod", "exp", "typ", "strike"], index=opt.index)], axis=1)

    # 数据驱动到期日 (§5.54): 每个 (品种, 到期月) 的实际最后交易日
    # 已到期 → 用数据里的 max TradeDate; 未到期 → day-15 近似
    data_end = df["TradeDate"].max()
    grp_last = opt.groupby(["prod", "exp"])["TradeDate"].max()
    exp_actual = {k: (ltd if ltd < data_end - pd.Timedelta(days=3) else k[1])
                  for k, ltd in grp_last.items()}
    opt["exp_act"] = [exp_actual[(p, e)]
                      for p, e in zip(opt["prod"], opt["exp"])]

    rows = {}
    for d, dag in opt.groupby("TradeDate", sort=True):
        exp_iv = {}
        for e, g in dag.groupby("exp"):
            F = fmap.get((d, e), np.nan)
            T = (g["exp_act"].iloc[0] - d).days / 365.0
            if not np.isfinite(F) or F <= 0 or T <= MIN_T or T > 0.5:
                continue
            pick = g.assign(dist=(g["strike"] - F).abs()).nsmallest(4, "dist")
            ivs = [black76_iv(x["Close"], F, x["strike"], T, x["typ"])
                   for _, x in pick.iterrows()]
            ivs = [v for v in ivs if np.isfinite(v) and v < 3]
            if ivs:
                exp_iv[e] = float(np.mean(ivs))
        if not exp_iv:
            continue
        es = sorted(exp_iv)
        near = es[0]
        far = es[1] if len(es) > 1 else None
        put_iv = call_iv = np.nan
        F = fmap.get((d, near), np.nan)
        T = (exp_actual[(dag["prod"].iloc[0], near)] - d).days / 365.0
        if np.isfinite(F) and F > 0 and T > MIN_T:
            g = dag[dag["exp"] == near]
            puts, calls = g[g["typ"] == "P"], g[g["typ"] == "C"]
            if len(puts):
                x = puts.loc[(puts["strike"] - 0.95 * F).abs().idxmin()]
                put_iv = black76_iv(x["Close"], F, x["strike"], T, "P")
            if len(calls):
                x = calls.loc[(calls["strike"] - 1.05 * F).abs().idxmin()]
                call_iv = black76_iv(x["Close"], F, x["strike"], T, "C")
        rows[d] = {"iv_near": exp_iv[near],
                   "iv_far": exp_iv[far] if far else np.nan,
                   "iv_put": put_iv, "iv_call": call_iv}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def one(item):
    col, view = item
    cp = SURF / f"{col}.parquet"
    if cp.exists():
        s = pd.read_parquet(cp)
        s.index = pd.to_datetime(s.index)
        return col, s
    # 复用 ATM IV 的原始抓取缓存 (避免重复网络)
    raw = CACHE / f"{col}_raw.parquet"
    if raw.exists():
        df = pd.read_parquet(raw)
    else:
        df = _query(
            f"SELECT TradeDate, Symbol, Close, Volume "
            f"FROM daily_quote.v_{view}_daily "
            f"WHERE Close > 0 AND (match(Symbol,'{FUT_RE}') "
            f"OR (match(Symbol,'[CP]') AND Volume > 0))")
        df.to_parquet(raw)
    s = compute_surface(df)
    s.to_parquet(cp)
    return col, s


def main():
    SURF.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    out = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        fs = {ex.submit(one, kv): kv[0] for kv in IV_PRODS.items()}
        for f in as_completed(fs):
            col = fs[f]
            try:
                c, s = f.result()
                if len(s):
                    out[c] = s
                    ts = (s["iv_far"] - s["iv_near"]).mean()
                    sk = (s["iv_put"] - s["iv_call"]).mean()
                    print(f"  {c}: {len(s)} 天, IVTS {ts*100:+.2f}% "
                          f"skew {sk*100:+.2f}%", flush=True)
            except Exception as e:
                print(f"  {col}: ✗ {type(e).__name__}: {str(e)[:70]}",
                      flush=True)
    print(f"\n完成 {len(out)} 产品 → {SURF}")


if __name__ == "__main__":
    main()
