#!/usr/bin/env python3
# coding=utf-8
"""提取期货期限结构曲率 (curvature / butterfly) — §5.55.

定义 (近/中/远 三合约):
  carry_near = (P1/P2 − 1) × 365/(T2−T1)
  carry_far  = (P2/P3 − 1) × 365/(T3−T2)
  curvature  = carry_far − carry_near   (期限结构二阶差分)

正 = 远端斜率更陡 (凸); 负 = 近端更陡 (凹).
输出: data/allweather/curve_daily.parquet (产品 × 日期)
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.extract_carry import (  # noqa: E402
    PANEL_FUTS, _expiry, fetch_product)

OUT = REPO / "data/allweather/curve_daily.parquet"
CACHE = REPO / "data/allweather/curve_cache"


def compute_curve(df: pd.DataFrame) -> pd.Series:
    """单产品期限结构曲率时序 (近/中/远 三合约二阶差分)."""
    df = df.copy()
    df["exp"] = df["Symbol"].map(_expiry)
    df = df.dropna(subset=["exp"]).sort_values(["TradeDate", "exp"])
    df["rnk"] = df.groupby("TradeDate")["exp"].rank(method="first")
    c1 = df[df["rnk"] == 1].set_index("TradeDate")
    c2 = df[df["rnk"] == 2].set_index("TradeDate")
    c3 = df[df["rnk"] == 3].set_index("TradeDate")
    j = c1.index.intersection(c2.index).intersection(c3.index)
    d12 = (c2.loc[j, "exp"] - c1.loc[j, "exp"]).dt.days
    d23 = (c3.loc[j, "exp"] - c2.loc[j, "exp"]).dt.days
    ok = ((d12 > 0) & (d23 > 0) & (c1.loc[j, "Close"] > 0)
          & (c2.loc[j, "Close"] > 0) & (c3.loc[j, "Close"] > 0))
    near = (c1.loc[j, "Close"] / c2.loc[j, "Close"] - 1.0) * 365.0 / d12
    far = (c2.loc[j, "Close"] / c3.loc[j, "Close"] - 1.0) * 365.0 / d23
    s = (far - near)[ok].sort_index()
    s.index = pd.to_datetime(s.index)
    return s


def one(item):
    col, view = item
    cp = CACHE / f"{col}.parquet"
    if cp.exists():
        s = pd.read_parquet(cp).iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return col, s
    df = fetch_product(view)
    s = compute_curve(df)
    pd.DataFrame({col: s}).to_parquet(cp)
    return col, s


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        fs = {ex.submit(one, kv): kv[0] for kv in PANEL_FUTS.items()}
        for f in as_completed(fs):
            col = fs[f]
            try:
                c, s = f.result()
                if len(s):
                    frames[c] = s
                    print(f"  {c}: {len(s)} 天, 曲率均值 {s.mean()*100:+.2f}%",
                          flush=True)
            except Exception as e:
                print(f"  {col}: ✗ {type(e).__name__}: {str(e)[:60]}",
                      flush=True)
    curv = pd.DataFrame(frames).sort_index()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    curv.to_parquet(OUT)
    print(f"\n已保存: {OUT} ({curv.shape})")


if __name__ == "__main__":
    main()
