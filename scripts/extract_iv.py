#!/usr/bin/env python3
# coding=utf-8
"""提取期货期权 ATM 隐含波动率 (IV) — §5.52 新 alpha 数据源.

数据源: daily_quote.v_<PRODUCT>_daily 中的期权合约
  (Symbol 如 cu2007C35500 / i2612-P-610 → 品种/到期/类型/行权价)
方法: Black-76 (期货期权) 数值反解 IV
  - 标的 F = 同品种同到期月期货合约当日 Close
  - T = (到期月 15 日 − 交易日)/365 (合约级精确到期仅 2025-12+ 有, 近似)
  - ATM: 取离 F 最近的 ≤4 个行权价, 平均其 IV
输出: data/allweather/iv_daily.parquet (产品 × 日期的 ATM IV)
"""
from __future__ import annotations

import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.extract_carry import _expiry  # noqa: E402

OUT = REPO / "data/allweather/iv_daily.parquet"
CACHE = REPO / "data/allweather/iv_cache"
R = 0.02

IV_PRODS = {"fut_cu": "CU", "fut_au": "AU", "fut_ag": "AG",
            "fut_rb": "RB", "fut_m": "M", "fut_sr": "SR", "fut_i": "I",
            "fut_al": "AL", "fut_zn": "ZN", "fut_ni": "NI", "fut_hc": "HC",
            "fut_j": "J", "fut_jm": "JM", "fut_y": "Y", "fut_p": "P",
            "fut_c": "C", "fut_cf": "CF", "fut_ta": "TA", "fut_ma": "MA",
            "fut_pp": "PP", "fut_l": "L", "fut_bu": "BU", "fut_sc": "SC"}
FUT_RE = r"^[A-Za-z]{1,2}[0-9]{3,4}$"
OPT_RE = re.compile(r"^([A-Za-z]{1,2})(\d{3,4})([CP])(\d+(?:\.\d+)?)$")


def _query(sql: str, timeout: int = 1200) -> pd.DataFrame:
    import subprocess
    r = subprocess.run(
        ["curl", "-s", "-G", "http://default:Imsn0wfree@0.0.0.0:8123/",
         "--data-urlencode", f"query={sql} FORMAT TSVWithNames"],
        capture_output=True, timeout=timeout)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(r.stderr.decode()[:200] or "empty response")
    return pd.read_csv(StringIO(r.stdout.decode()), sep="\t")


def parse_option(sym: str):
    m = OPT_RE.match(sym.replace("-", ""))
    if not m:
        return None
    prod, digits, typ, strike = m.groups()
    exp = _expiry(prod + digits)
    if exp is None:
        return None
    return prod.upper(), exp, typ, float(strike)


def black76_iv(price, F, K, T, typ, r=R):
    if T <= 1 / 365 or price <= 0 or F <= 0 or K <= 0:
        return np.nan
    sq = np.sqrt(T)

    def f(sig):
        d1 = (np.log(F / K) + 0.5 * sig * sig * T) / (sig * sq)
        d2 = d1 - sig * sq
        if typ == "C":
            return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2)) - price
        return np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1)) - price

    try:
        if f(1e-3) * f(5.0) > 0:
            return np.nan
        return brentq(f, 1e-3, 5.0)
    except Exception:
        return np.nan


def compute_iv(df: pd.DataFrame, max_iv=3.0) -> pd.Series:
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
        return pd.Series(dtype=float)
    opt = pd.concat([opt, pd.DataFrame(
        [x for x in parsed if x is not None],
        columns=["prod", "exp", "typ", "strike"], index=opt.index)], axis=1)

    out = {}
    for (d, e), g in opt.groupby(["TradeDate", "exp"], sort=False):
        F = fmap.get((d, e), np.nan)
        if not np.isfinite(F) or F <= 0:
            continue
        T = (e - d).days / 365.0
        if T <= 1 / 365 or T > 0.5:
            continue
        g = g.assign(dist=(g["strike"] - F).abs()).nsmallest(4, "dist")
        ivs = [black76_iv(row["Close"], F, row["strike"], T, row["typ"])
               for _, row in g.iterrows()]
        ivs = [v for v in ivs if np.isfinite(v) and v < max_iv]
        if ivs:
            out[d] = float(np.mean(ivs))
    return pd.Series(out).sort_index()


def one(item):
    col, view = item
    cp = CACHE / f"{col}.parquet"
    if cp.exists():
        s = pd.read_parquet(cp).iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return col, s
    df = _query(
        f"SELECT TradeDate, Symbol, Close, Volume "
        f"FROM daily_quote.v_{view}_daily "
        f"WHERE Close > 0 AND (match(Symbol,'{FUT_RE}') "
        f"OR (match(Symbol,'[CP]') AND Volume > 0))")
    s = compute_iv(df)
    pd.DataFrame({col: s}).to_parquet(cp)
    return col, s


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(one, kv): kv[0] for kv in IV_PRODS.items()}
        for f in as_completed(futs):
            col = futs[f]
            try:
                c, s = f.result()
                if len(s):
                    frames[c] = s
                    print(f"  {c}: {len(s)} 天 "
                          f"({s.index[0].date()} → {s.index[-1].date()}), "
                          f"IV 均值 {s.mean()*100:.1f}%", flush=True)
            except Exception as e:
                print(f"  {col}: ✗ {type(e).__name__}: {str(e)[:70]}",
                      flush=True)
    iv = pd.DataFrame(frames).sort_index()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    iv.to_parquet(OUT)
    print(f"\n已保存: {OUT} ({iv.shape})")


if __name__ == "__main__":
    main()
