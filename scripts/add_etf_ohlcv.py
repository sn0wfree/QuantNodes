#!/usr/bin/env python3
# coding=utf-8
"""补全 OHLCV 缺失资产 (512890 红利低波) + 重新前复权.

源约定: raw 文件 = fund_etf_hist_sina (收盘/成交量逐值一致; em 成交量单位为手×100).
流程: 备份 → 拉取 512890 → 对齐现有 index → 合并 raw → 重跑前复权 → 校验.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from fix_ohlcv_adjust import detect_corp_actions, forward_adjust  # noqa: E402

SRC = REPO / "data/real" / "etf_ohlcv_2018-01-01_2026-06-30.parquet"
DST = REPO / "data/real" / "etf_ohlcv_2018-01-01_2026-06-30_adjusted.parquet"
FIELDS = ["open", "high", "low", "close", "volume"]


def fetch_sina(code: str) -> pd.DataFrame:
    """经 sina 接口拉取 ETF OHLCV (与 raw 文件同源同单位)."""
    import akshare as ak
    sym = f"sh{code}" if code.startswith(("5", "6")) else f"sz{code}"
    df = ak.fund_etf_hist_sina(symbol=sym)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df[FIELDS]


def main():
    if not SRC.exists():
        raise SystemExit(f"缺 raw 文件: {SRC}")
    raw = pd.read_parquet(SRC)
    raw.index = pd.to_datetime(raw.index)
    have = {str(c) for c in raw.columns.get_level_values(0)}
    targets = [c for c in ("512890", "512100") if c not in have]
    if not targets:
        print("512890 已存在, 无需补全")
        return

    for p in (SRC, DST):
        bak = p.with_suffix(".parquet.bak-20260906")
        if not bak.exists():
            shutil.copy2(p, bak)
            print(f"[backup] {bak.name}")

    for code in targets:
        df = fetch_sina(code)
        df = df.reindex(raw.index)           # 对齐现有 index (范围/日历)
        block = pd.concat({code: df}, axis=1)[code][FIELDS]
        block.columns = pd.MultiIndex.from_product([[code], FIELDS])
        raw = pd.concat([raw, block], axis=1)
        n = int(block[(code, "close")].notna().sum())
        first = block[(code, "close")].first_valid_index()
        last = block[(code, "close")].last_valid_index()
        print(f"[add] {code}: {n} 行 ({first.date()} → {last.date()})")

    raw = raw.sort_index(axis=1)
    raw.to_parquet(SRC, index=True)
    print(f"[save] {SRC.name}: {raw.shape}, 资产 {len(set(raw.columns.get_level_values(0)))}")

    acts = detect_corp_actions(raw)
    print(f"[detect] 公司行为事件: {len(acts)}")
    for a in acts:
        print(f"  {a['code']} @ {a['date']:%Y-%m-%d}: {a['ratio']-1:+.1%}")
    adj = forward_adjust(raw, acts)
    adj.to_parquet(DST, index=True)
    print(f"[save] {DST.name}: {adj.shape}")

    # 校验: 复权后无 >50% 跳变
    bad = []
    for code in adj.columns.levels[0]:
        c = adj[code]["close"].dropna()
        r = c / c.shift(1) - 1
        for d in r[r.abs() > 0.50].index:
            bad.append((code, d.date(), round(float(r.loc[d]), 3)))
    print(f"[verify] 复权后 |跳变|>50%: {bad if bad else '无 ✓'}")


if __name__ == "__main__":
    main()
