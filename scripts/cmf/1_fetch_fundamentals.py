#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/1_fetch_fundamentals.py — 抓取基本面数据并落盘 DuckDB.

来源: akshare 免费接口 (Phase 0 确认可达).

输出: data/cmf/fundamental.duckdb, 三表:
  - inventory_daily  (铜LME库存/原油EIA库存/豆粕DCE库存)
  - fundamentals_daily (钻机数/工业产出/猪价/CPI/PMI)
  - futures_main_close (CU/SC/M主力合约日度收盘价)
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import duckdb
import akshare as ak

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "cmf" / "fundamental.duckdb"


def fetch_lme_stock() -> pd.DataFrame:
    """LME库存 (含铜铝锌镍, 2014-07起, 日频)."""
    df = ak.macro_euro_lme_stock()
    if df is None or df.empty:
        return pd.DataFrame()
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.rename(columns={"日期": "trade_date"}).sort_values("trade_date")
    keep = ["trade_date", "铜-库存", "铜-注册仓单", "铝-库存", "铝-注册仓单", "锌-库存", "镍-库存"]
    return df[[c for c in keep if c in df.columns]]


def fetch_eia_crude() -> pd.DataFrame:
    """EIA原油库存变动 (1982-09起, 月频!)."""
    df = ak.macro_usa_eia_crude_rate()
    if df is None or df.empty:
        return pd.DataFrame()
    df["日期"] = pd.to_datetime(df["日期"])
    df = df[df["商品"].astype(str).str.contains("EIA原油库存")]
    return df.rename(columns={"日期": "trade_date", "今值": "eia_chg_kbd"})[["trade_date", "eia_chg_kbd"]].sort_values("trade_date")


def fetch_rig_count() -> pd.DataFrame:
    """美国钻机数 (1987-07起, 周频)."""
    df = ak.macro_usa_rig_count()
    if df is None or df.empty:
        return pd.DataFrame()
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.rename(columns={"日期": "trade_date"})
    keep = ["trade_date", "美国石油钻井_钻井数"]
    df = df[[c for c in keep if c in df.columns]]
    df.columns = ["trade_date", "us_oil_rig"]
    return df.sort_values("trade_date")


def fetch_futures_main(symbol: str) -> pd.DataFrame:
    """期货主力日度收盘."""
    df = ak.futures_main_sina(symbol=symbol)
    if df is None or df.empty:
        return pd.DataFrame()
    rename_map = {"日期": "trade_date", "开盘价": "open", "最高价": "high",
                  "最低价": "low", "收盘价": "close", "成交量": "volume",
                  "持仓量": "open_interest"}
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["symbol"] = symbol
    return df.sort_values("trade_date")


def fetch_hog_spot_price() -> pd.DataFrame:
    """生猪价格指数 (2015-01起, 周频)."""
    df = ak.index_hog_spot_price()
    if df is None or df.empty:
        return pd.DataFrame()
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.rename(columns={"日期": "trade_date", "指数": "hog_index"})
    return df[["trade_date", "hog_index"]].sort_values("trade_date")


def fetch_china_pmi() -> pd.DataFrame:
    """中国制造业PMI (列名是 '2026年07月份' 这种)."""
    df = ak.macro_china_pmi()
    if df is None or df.empty:
        return pd.DataFrame()
    month_col = "月份"
    df[month_col] = df[month_col].str.replace("月份", "", regex=False)
    df[month_col] = pd.to_datetime(df[month_col], format="%Y年%m", errors="coerce")
    df = df.dropna(subset=[month_col])
    df = df.rename(columns={month_col: "trade_date", "制造业-指数": "pmi_mfg"})
    return df[["trade_date", "pmi_mfg"]].sort_values("trade_date")


def fetch_usa_industrial() -> pd.DataFrame:
    """美国工业产出指数."""
    df = ak.macro_usa_industrial_production()
    if df is None or df.empty:
        return pd.DataFrame()
    df["日期"] = pd.to_datetime(df["日期"])
    df = df.rename(columns={"日期": "trade_date", "今值": "us_ip"})
    return df[["trade_date", "us_ip"]].sort_values("trade_date")


def fetch_usa_cpi() -> pd.DataFrame:
    """美国CPI."""
    df = ak.macro_usa_cpi_monthly()
    if df is None or df.empty:
        return pd.DataFrame()
    date_col = next((c for c in df.columns if "日期" in c), None)
    val_col = next((c for c in df.columns if "今值" in c), None)
    if not date_col or not val_col:
        return pd.DataFrame()
    df[date_col] = pd.to_datetime(df[date_col])
    return df.rename(columns={date_col: "trade_date", val_col: "us_cpi_yoy"})[["trade_date", "us_cpi_yoy"]].sort_values("trade_date")


def main():
    print("=" * 70)
    print("CMF Phase 1 — 基本面数据抓取 → DuckDB")
    print("=" * 70)

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))

    fetches = [
        ("inventory_daily", "LME铜库存", fetch_lme_stock),
        ("inventory_daily", "EIA原油库存变动", fetch_eia_crude),
        ("fundamentals_daily", "美国钻机数", fetch_rig_count),
        ("fundamentals_daily", "猪价指数", fetch_hog_spot_price),
        ("fundamentals_daily", "中国PMI", fetch_china_pmi),
        ("fundamentals_daily", "美国工业产出", fetch_usa_industrial),
        ("fundamentals_daily", "美国CPI", fetch_usa_cpi),
    ]

    con.execute("""
        CREATE OR REPLACE TABLE inventory_daily (
            source VARCHAR, trade_date DATE, value_col VARCHAR, value DOUBLE
        );
        CREATE OR REPLACE TABLE fundamentals_daily (
            source VARCHAR, trade_date DATE, value_col VARCHAR, value DOUBLE
        );
    """)

    for table, source_name, fn in fetches:
        print(f"\n[抓取] {source_name} → {table}")
        try:
            df = fn()
        except Exception as e:
            print(f"  ✗ FAIL: {type(e).__name__}: {str(e)[:80]}")
            continue
        if df.empty:
            print(f"  ⚠️ empty")
            continue
        value_cols = [c for c in df.columns if c != "trade_date"]
        rows = []
        for col in value_cols:
            sub = df[["trade_date", col]].dropna(subset=[col]).copy()
            sub["source"] = source_name
            sub["value_col"] = col
            sub = sub.rename(columns={col: "value"})
            rows.append(sub)
        if not rows:
            print(f"  ⚠️ 无非空值")
            continue
        stacked = pd.concat(rows, ignore_index=True)
        con.execute(f"DELETE FROM {table} WHERE source = ?", [source_name])
        con.append(table, stacked, by_name=True)
        print(f"  ✓ {len(stacked)} 行 (cols={value_cols})")

    fut_fetches = [
        ("CU0", "futures_main_close_cu"),
        ("SC0", "futures_main_close_sc"),
        ("M0", "futures_main_close_m"),
    ]
    for sym, table in fut_fetches:
        print(f"\n[抓取] 期货主力 {sym} → {table}")
        try:
            df = fetch_futures_main(sym)
        except Exception as e:
            print(f"  ✗ FAIL: {type(e).__name__}: {str(e)[:80]}")
            continue
        if df.empty:
            print(f"  ⚠️ empty")
            continue
        con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df")
        print(f"  ✓ {len(df)} 行, cols={list(df.columns)}")

    print("\n" + "=" * 70)
    print("DuckDB 内容摘要:")
    for tbl in con.execute("SHOW TABLES").fetchall():
        tname = tbl[0]
        n = con.execute(f"SELECT COUNT(*) FROM {tname}").fetchone()[0]
        s = con.execute(f"SELECT MIN(trade_date), MAX(trade_date) FROM {tname}").fetchone()
        print(f"  {tname:30s}  {n:>7}行  [{s[0]}~{s[1]}]")

    con.close()
    print(f"\n✓ 已写入: {DB_PATH}")


if __name__ == "__main__":
    main()