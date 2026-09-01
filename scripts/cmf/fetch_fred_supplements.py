#!/usr/bin/env python3
"""fetch_fred_supplements.py — 拉取FRED日频数据补全黄金因子缺口.

补全目标 (4个可自动拉取):
  dxy (DTWEXBGS)       → gold_macro/dxy (宽基美元指数)
  tips10y (DFII10)     → gold_fred/tips10y (10年TIPS, 2003起, 5919行)
  tips (DFII5)         → gold_fred/tips (5年TIPS, 2003起)
  fed_rate (DFF)       → gold_fred/fed_rate_daily (联邦基金有效日利率)

幂等: 全量DELETE+INSERT (idempotent, safe to re-run)
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import json
import urllib.request
import urllib.parse
import pandas as pd
import duckdb

DB_PATH = "data/cmf/fundamental.duckdb"
KEYS_PATH = "data/cmf/.keys.json"

FRED_SERIES = [
    ("DTWEXBGS", "gold_macro", "dxy", "宽基美元指数(DXY代理)", "1990-01-01"),
    ("DFII10",   "gold_fred",  "tips10y", "10年期TIPS收益率", "2003-01-01"),
    ("DFII5",    "gold_fred",  "tips", "5年期TIPS收益率", "2003-01-01"),
    ("DFF",      "gold_fred",  "fed_rate_daily", "联邦基金有效日利率", "1990-01-01"),
]


def load_api_key() -> str:
    with open(KEYS_PATH) as f:
        return json.load(f)["fred_api_key"]


def fetch_fred_series(series_id: str, api_key: str, start: str) -> pd.DataFrame:
    params = urllib.parse.urlencode({
        "series_id": series_id, "api_key": api_key,
        "file_type": "json", "observation_start": start,
    })
    url = f"https://api.stlouisfed.org/fred/series/observations?{params}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    rows = []
    for obs in data.get("observations", []):
        if obs["value"] == ".":
            continue
        rows.append({"trade_date": pd.Timestamp(obs["date"]), "value": float(obs["value"])})
    df = pd.DataFrame(rows).set_index("trade_date").sort_index()
    return df


def main():
    api_key = load_api_key()
    con = duckdb.connect(DB_PATH)

    for series_id, source, value_col, desc, start in FRED_SERIES:
        print(f"\n[拉取] {desc} ({series_id}) → {source}.{value_col}")
        df = fetch_fred_series(series_id, api_key, start)
        if len(df) == 0:
            print(f"    ✗ 无数据")
            continue
        print(f"    获取 {len(df)} 行, {df.index.min().date()} ~ {df.index[-1].date()}")

        # 全量覆盖 (幂等)
        con.execute("DELETE FROM fundamentals_daily WHERE source=? AND value_col=?", [source, value_col])
        df_out = df.reset_index().rename(columns={"index": "trade_date"})
        df_out["trade_date"] = pd.to_datetime(df_out["trade_date"])
        df_out["source"] = source
        df_out["value_col"] = value_col
        df_out["value"] = df_out["value"].astype(float)
        con.execute("""
            INSERT INTO fundamentals_daily (trade_date, source, value_col, value)
            SELECT trade_date, source, value_col, value FROM df_out
        """)
        print(f"    ✓ 全量写入 {len(df_out)} 行")

    # 验证
    print("\n" + "="*60)
    print("验证补全结果:")
    for _, source, value_col, desc, _ in FRED_SERIES:
        r = con.execute(
            "SELECT COUNT(*), MIN(trade_date), MAX(trade_date) FROM fundamentals_daily WHERE source=? AND value_col=?",
            [source, value_col]
        ).fetchone()
        print(f"  {desc:<20} {r[0]:>6} 行, {str(r[1])[:10]:>12} ~ {str(r[2])[:10]:>12}")

    con.close()
    print("\n✓ 补全完成")


if __name__ == "__main__":
    main()
