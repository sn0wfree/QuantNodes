#!/usr/bin/env python3
"""scripts/cmf/fetch_fred_oil.py — FRED原油数据拉取 (WTI/Brent).

金油比值研究专用: 拉取DCOILWTICO + DCOILBRENTEU长历史数据.
复用现有fredapi基础设施 (data/cmf/.keys.json).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import duckdb
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data/cmf/fundamental.duckdb"
KEYS_PATH = REPO / "data/cmf" / ".keys.json"

SERIES = [
    ("DCOILWTICO", "wti"),
    ("DCOILBRENTEU", "brent"),
]


def fetch_fred_series(series_id: str, fred_key: str) -> pd.DataFrame:
    """拉取FRED时间序列."""
    from fredapi import Fred
    fred = Fred(api_key=fred_key)
    s = fred.get_series(series_id, observation_start="1985-01-01")
    df = pd.DataFrame({"value": s.values, "trade_date": s.index})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["source"] = "fred_dcoil"
    df["value_col"] = None
    return df


def main():
    print("=" * 60)
    print("FRED 原油数据拉取 (WTI + Brent)")
    print("=" * 60)

    if not KEYS_PATH.exists():
        print(f"✗ 缺少 {KEYS_PATH}")
        return
    keys = json.loads(KEYS_PATH.read_text(encoding="utf-8"))
    fred_key = keys.get("fred_api_key", "")
    if not fred_key:
        print("✗ 缺少 fred_api_key")
        return

    con = duckdb.connect(DB_PATH)

    for series_id, value_col in SERIES:
        print(f"\n[拉取] {series_id} -> {value_col}")
        try:
            df = fetch_fred_series(series_id, fred_key)
        except Exception as e:
            print(f"  ✗ FAIL: {type(e).__name__}: {str(e)[:80]}")
            continue

        if df.empty:
            print("  ⚠️ empty")
            continue

        df["value_col"] = value_col
        # 幂等写入
        con.execute(
            "DELETE FROM fundamentals_daily WHERE source=? AND value_col=?",
            ["fred_dcoil", value_col],
        )
        out = df[["trade_date", "source", "value_col", "value"]].copy()
        out["trade_date"] = out["trade_date"].dt.date
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["value"])
        con.append("fundamentals_daily", out, by_name=True)
        n = con.execute(
            "SELECT COUNT(*) FROM fundamentals_daily WHERE source='fred_dcoil' AND value_col=?",
            [value_col],
        ).fetchone()[0]
        dr = con.execute(
            "SELECT MIN(trade_date), MAX(trade_date) FROM fundamentals_daily "
            "WHERE source='fred_dcoil' AND value_col=?",
            [value_col],
        ).fetchone()
        print(f"  ✓ 已写入 {n}行, {dr[0]} ~ {dr[1]}")

    con.close()
    print("\n✓ 拉取完成")


if __name__ == "__main__":
    main()