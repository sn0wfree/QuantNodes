#!/usr/bin/env python3
# coding=utf-8
"""延长 cash_proxy (中债1-3年国债财富指数) 至最新 (§5.47).

原源停在 2026-05-29 (G5-lite 现金腿核心). akshare 仅有中债-综合指数
(无国债专属); 用 `bond_composite_index_cbond('财富','1-3年')` (2002→今)
**按收益链接**延长: 保留原历史, 从断点起用新指数的日收益累乘续接
(现金腿对两个短久期债券财富指数的差异不敏感, 日均差 ~0.8pp/年).

用法: python scripts/extend_cash_proxy.py [--end 2026-09-03]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
V7 = REPO / "data/high_freq_macro" / "v7_index_v2.parquet"
COL = "中债1-3年国债财富指数"


def fetch_new_index() -> pd.Series:
    """中债-综合指数 1-3年 财富 (财富 = 全收益)."""
    import akshare as ak
    df = ak.bond_composite_index_cbond(indicator="财富", period="1-3年")
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].sort_index()


def extend(v7_path: Path = V7, end: str = "2026-09-03",
           backup: bool = True) -> pd.Series:
    v7 = pd.read_parquet(v7_path)
    v7.index = pd.to_datetime(v7.index)
    old = v7[COL].dropna()
    old_last_date, old_last_val = old.index.max(), old.iloc[-1]

    new = fetch_new_index()
    new = new.loc[:pd.Timestamp(end)]
    add = new.loc[new.index > old_last_date]
    if len(add) == 0:
        print(f"无需延长 (旧已至 {old_last_date.date()})")
        return old

    # 收益链接: 以旧末端为基准, 累乘新指数的日收益
    rets = new.pct_change(fill_method=None).loc[add.index]
    chained = old_last_val * (1.0 + rets.fillna(0.0)).cumprod()
    merged = pd.concat([old, chained])
    merged = merged[~merged.index.duplicated(keep="last")]

    if backup and not (v7_path.with_suffix(".parquet.bak-20260906").exists()):
        shutil.copy2(v7_path, v7_path.with_suffix(".parquet.bak-20260906"))
        print(f"[backup] {v7_path.name}.bak-20260906")

    print(f"延长 {len(add)} 天: {add.index[0].date()} → {add.index[-1].date()} "
          f"(值 {old_last_val:.4f} → {merged.iloc[-1]:.4f})")
    v7[COL] = merged
    v7.to_parquet(v7_path, index=True)
    print(f"[save] {v7_path.name}: {v7.shape}")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end", default="2026-09-03")
    args = ap.parse_args()
    s = extend(end=args.end)
    r = s.pct_change(fill_method=None).dropna()
    j = r.loc["2026-05-20":]
    print(f"校验: 全序列 {s.index[0].date()} → {s.index[-1].date()}, "
          f"末段日收益 max|r| = {j.abs().max()*1e4:.1f}bp (应无跳变)")


if __name__ == "__main__":
    main()
