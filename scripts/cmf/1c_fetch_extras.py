#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/1c_fetch_extras.py — EIA/FRED/DXY 数据补齐 fetcher.

数据源:
  - EIA API v2 (商业原油库存 WCRSTUS1, 周度)
  - FRED API (TIPS 10Y DFII10, 日度)
  - akshare currency_boc_safe 合成 ICE DXY (日度)

输出:
  - data/cmf/fundamental.duckdb 新增:
    eia_weekly_inventory (eia_crude_stocks 千桶, 1983+)
    fundamentals_daily 新增 value_col: 'dxy', 'tips10y'
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Callable

warnings.filterwarnings("ignore")

import duckdb
import pandas as pd
import requests
from CodersWheel.QuickTool.file_cache import file_cache

REPO = Path(__file__).resolve().parents[2]
KEYS_PATH = REPO / "data" / "cmf" / ".keys.json"
DB_PATH = REPO / "data" / "cmf" / "fundamental.duckdb"


@file_cache(enable_cache=True, granularity='H')
def _eia_weekly_cached(api_key: str) -> pd.DataFrame:
    """EIA WCRSTUS1 — 核心 API 调用 (带缓存)."""
    url = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
    params = {
        "api_key": api_key,
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": "WCRSTUS1",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": 10000,
    }
    try:
        r = requests.get(url, params=params, timeout=90)
        if r.status_code == 200:
            rows = r.json().get("response", {}).get("data", [])
            if rows:
                df = pd.DataFrame(rows)
                df["trade_date"] = pd.to_datetime(df["period"])
                df["eia_crude_stocks"] = pd.to_numeric(df["value"], errors="coerce")
                return df[["trade_date", "eia_crude_stocks"]].dropna().sort_values("trade_date")
    except Exception:
        pass
    return pd.DataFrame()


@file_cache(enable_cache=True, granularity='H')
def _fred_tips10y_cached(api_key: str) -> pd.DataFrame:
    """FRED DFII10 — 核心 API 调用 (带缓存)."""
    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        s = fred.get_series("DFII10", observation_start="1990-01-01")
        df = pd.DataFrame({"trade_date": s.index, "tips10y": s.values})
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.dropna()
    except Exception:
        pass
    return pd.DataFrame()


@file_cache(enable_cache=True, granularity='H')
def _dxy_synthetic_cached() -> pd.DataFrame:
    """DXY合成 — 核心调用 (带缓存)."""
    try:
        import akshare as ak
        df = ak.currency_boc_safe()
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.set_index("日期").sort_index()
        df["EURUSD"] = df["欧元"] / df["美元"]
        df["USDJPY"] = df["美元"] / df["日元"]
        df["GBPUSD"] = df["英镑"] / df["美元"]
        df["USDCAD"] = df["美元"] / df["加元"]
        df["USDSEK"] = df["美元"] / df["瑞典克朗"]
        df["USDCHF"] = df["美元"] / df["瑞士法郎"]
        dxy = 50.14348112 * (
            df["EURUSD"] ** (-0.576)
            * df["USDJPY"] ** (0.136)
            * df["GBPUSD"] ** (-0.119)
            * df["USDCAD"] ** (0.091)
            * df["USDSEK"] ** (0.042)
            * df["USDCHF"] ** (0.036)
        )
        return pd.DataFrame({"trade_date": dxy.index, "dxy": dxy.values}).dropna()
    except Exception:
        pass
    return pd.DataFrame()


def load_keys() -> dict[str, str]:
    """从 .keys.json 读 key, 缺失返回空 (fetcher 调用方容错)."""
    if not KEYS_PATH.exists():
        return {}
    return json.loads(KEYS_PATH.read_text(encoding="utf-8"))


def _mask(k: str | None) -> str:
    if not k:
        return "(missing)"
    return k[:4] + "***" + k[-4:]


def fetch_eia_weekly_crude(api_key: str, retries: int = 5, timeout: int = 90) -> pd.DataFrame:
    """EIA API v2 — 拉美国商业原油库存 (内部调用缓存版本)."""
    if not api_key or api_key == "PLACEHOLDER_FILL_REAL_KEY":
        raise ValueError("EIA API key 未配置或仍为占位符")
    for attempt in range(retries):
        df = _eia_weekly_cached(api_key)
        if not df.empty:
            return df
        if attempt < retries - 1:
            time.sleep(2)
    return pd.DataFrame()


def fetch_fred_tips10y(api_key: str) -> pd.DataFrame:
    """FRED DFII10 — 内部调用缓存版本."""
    if not api_key or api_key == "PLACEHOLDER_FILL_REAL_KEY":
        raise ValueError("FRED API key 未配置或仍为占位符")
    df = _fred_tips10y_cached(api_key)
    if df.empty:
        raise RuntimeError("FRED DFII10 拉取失败")
    return df


def fetch_dxy_synthetic() -> pd.DataFrame:
    """DXY合成 — 内部调用缓存版本."""
    df = _dxy_synthetic_cached()
    if df.empty:
        raise RuntimeError("DXY 合成拉取失败")
    return df


def upsert_fundamentals_daily(con, source: str, value_col: str, df: pd.DataFrame) -> int:
    """将 DataFrame (trade_date, value) 写入 fundamentals_daily 长表."""
    con.execute(
        "DELETE FROM fundamentals_daily WHERE source = ? AND value_col = ?",
        [source, value_col],
    )
    value_series = df["value"] if "value" in df.columns else df.iloc[:, 1]
    rows = pd.DataFrame({
        "source": source,
        "trade_date": df["trade_date"].dt.date,
        "value_col": value_col,
        "value": value_series,
    })
    rows = rows.dropna(subset=["value"])
    if len(rows) == 0:
        return 0
    con.append("fundamentals_daily", rows, by_name=True)
    return len(rows)


def upsert_eia_table(con, df: pd.DataFrame) -> int:
    """eia_weekly_inventory 表 (trade_date, eia_crude_stocks)."""
    con.execute("""
        CREATE OR REPLACE TABLE eia_weekly_inventory (
            trade_date DATE,
            eia_crude_stocks DOUBLE
        );
    """)
    con.append("eia_weekly_inventory", df, by_name=True)
    return len(df)


def main():
    print("=" * 70)
    print("CMF Phase 6 — EIA/FRED/DXY 数据补齐")
    print("=" * 70)

    keys = load_keys()
    eia_key = keys.get("eia_api_key", "")
    fred_key = keys.get("fred_api_key", "")
    print(f"EIA key: {_mask(eia_key)}")
    print(f"FRED key: {_mask(fred_key)}")
    print()

    con = duckdb.connect(str(DB_PATH))

    fetches: list[tuple[str, str, str, Callable]] = [
        ("fundamentals_daily", "akshare_currency_boc", "dxy",    lambda: fetch_dxy_synthetic()),
        ("fundamentals_daily", "fred_dfii10",           "tips10y", lambda: fetch_fred_tips10y(fred_key)),
    ]

    for tbl, source, value_col, fn in fetches:
        label = f"{value_col} ({source})"
        print(f"\n[抓取] {label} → {tbl}")
        try:
            df = fn()
            df = df.rename(columns={df.columns[1]: "value"})
            n = upsert_fundamentals_daily(con, source=source, value_col=value_col, df=df)
            print(f"  ✓ {n} 行")
            print(f"  范围: {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}")
        except Exception as e:
            print(f"  ✗ FAIL: {type(e).__name__}: {str(e)[:120]}")

    print(f"\n[抓取] EIA WCRSTUS1 (美国商业原油库存, 千桶) → eia_weekly_inventory")
    try:
        df = fetch_eia_weekly_crude(eia_key)
        n = upsert_eia_table(con, df)
        print(f"  ✓ {n} 行")
        print(f"  范围: {df['trade_date'].min().date()} ~ {df['trade_date'].max().date()}")
        print(f"  最新值: {df['eia_crude_stocks'].iloc[-1]:,.0f} 千桶 @ {df['trade_date'].iloc[-1].date()}")
    except Exception as e:
        print(f"  ✗ FAIL: {type(e).__name__}: {str(e)[:200]}")

    print("\n" + "=" * 70)
    print("DuckDB 摘要:")
    for tbl in con.execute("SHOW TABLES").fetchall():
        tname = tbl[0]
        n = con.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
        date_col = "report_date" if tname == "tfapi_cftc" else "trade_date"
        try:
            s = con.execute(f"SELECT MIN({date_col}), MAX({date_col}) FROM {tname}").fetchone()
            print(f"  {tname:35s}  {n:>7}行  [{s[0]}~{s[1]}]")
        except:
            print(f"  {tname:35s}  {n:>7}行")

    print("\nfundamentals_daily 新增 value_col 状态:")
    for r in con.execute("""
        SELECT value_col, COUNT(*), MIN(trade_date), MAX(trade_date)
        FROM fundamentals_daily
        WHERE value_col IN ('dxy', 'tips10y')
        GROUP BY value_col ORDER BY value_col
    """).fetchall():
        print(f"  {r[0]:10s}  {r[1]:>5}行  [{r[2]}~{r[3]}]")

    con.close()


if __name__ == "__main__":
    main()