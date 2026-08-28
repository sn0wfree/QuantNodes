#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/1e_fetch_gold.py — 黄金数据迁移 SQLite→DuckDB.

数据源:
    1. ~/Public/gold/database/gold_analysis.db (SQLite, 14张表)
    2. akshare AU0 (SHFE黄金主力连续合约)

输出: data/cmf/fundamental.duckdb
    - futures_main_close_au  (SHFE黄金主力日度收盘)
    - fundamentals_daily (source='gold_fred'/'gold_macro'/'gold_position'/'gold_risk')
"""
from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import duckdb
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "cmf" / "fundamental.duckdb"
SQLITE_PATH = Path.home() / "Public" / "gold" / "database" / "gold_analysis.db"


def _read_sqlite(sql: str) -> pd.DataFrame:
    """读取SQLite数据, 日期转datetime."""
    con = sqlite3.connect(str(SQLITE_PATH))
    df = pd.read_sql(sql, con)
    con.close()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _to_eav(df: pd.DataFrame, source: str, value_cols: list[str]) -> pd.DataFrame:
    """将宽表转EAV格式 (source, trade_date, value_col, value)."""
    rows = []
    for col in value_cols:
        sub = df[["date", col]].dropna(subset=[col]).copy()
        sub["source"] = source
        sub["value_col"] = col
        sub = sub.rename(columns={"date": "trade_date", col: "value"})
        rows.append(sub[["trade_date", "source", "value_col", "value"]])
    if not rows:
        return pd.DataFrame(columns=["trade_date", "source", "value_col", "value"])
    return pd.concat(rows, ignore_index=True)


def fetch_fred_indicators() -> pd.DataFrame:
    """FRED指标 (TIPS, RIR, VIX, EPU, GVZ) - 月频/日频混合."""
    df = _read_sqlite(
        "SELECT date, indicator_name, value FROM fred_indicators ORDER BY date"
    )
    return df.rename(columns={"indicator_name": "value_col"}).copy()


def fetch_usd_index() -> pd.DataFrame:
    """美元指数 (日度, 2020+)."""
    df = _read_sqlite("SELECT date, value AS dxy FROM usd_index ORDER BY date")
    return _to_eav(df, "gold_macro", ["dxy"])


def fetch_treasury() -> pd.DataFrame:
    """美国国债收益率 (日度, 1990+)."""
    df = _read_sqlite(
        "SELECT date, us_2y, us_10y, us_10y_2y_spread "
        "FROM us_treasury_yields ORDER BY date"
    )
    return _to_eav(df, "gold_macro", ["us_2y", "us_10y", "us_10y_2y_spread"])


def fetch_cftc() -> pd.DataFrame:
    """CFTC黄金持仓 (周度, 1986+)."""
    df = _read_sqlite(
        "SELECT date, net_contracts AS cftc_net, open_interest AS cftc_oi "
        "FROM cftc_gold_positions ORDER BY date"
    )
    return _to_eav(df, "gold_position", ["cftc_net", "cftc_oi"])


def fetch_comex_inventory() -> pd.DataFrame:
    """COMEX黄金库存 (日度, 2004+)."""
    df = _read_sqlite(
        "SELECT date, inventory_tonnes AS comex_inv "
        "FROM comex_inventory ORDER BY date"
    )
    return _to_eav(df, "gold_position", ["comex_inv"])


def fetch_etf() -> pd.DataFrame:
    """SPDR黄金ETF持仓 (日度, 2020+)."""
    df = _read_sqlite("SELECT date, tonnes AS etf_tonnes FROM etf_holdings ORDER BY date")
    return _to_eav(df, "gold_position", ["etf_tonnes"])


def fetch_gprd() -> pd.DataFrame:
    """GPRD地缘政治风险 (周度, 1985+)."""
    df = _read_sqlite("SELECT date, gprd_value AS gprd FROM gprd_index ORDER BY date")
    return _to_eav(df, "gold_risk", ["gprd"])


def fetch_fed_rate() -> pd.DataFrame:
    """美联储利率 (事件驱动, 1982+)."""
    df = _read_sqlite("SELECT date, rate AS fed_rate FROM fed_interest_rate ORDER BY date")
    return _to_eav(df, "gold_macro", ["fed_rate"])


def fetch_central_bank() -> pd.DataFrame:
    """中国央行黄金储备 (月度, 2008+)."""
    df = _read_sqlite(
        "SELECT date, reserves_tonnes AS cb_reserves "
        "FROM central_bank_reserves ORDER BY date"
    )
    return _to_eav(df, "gold_position", ["cb_reserves"])


def fetch_gold_prices_spot() -> pd.DataFrame:
    """伦敦金现价格 (日度, 2006+)."""
    df = _read_sqlite("SELECT date, price AS gold_spot FROM gold_prices ORDER BY date")
    return _to_eav(df, "gold_macro", ["gold_spot"])


def fetch_futures_au() -> pd.DataFrame:
    """SHFE黄金主力连续合约 (日度, 2008+)."""
    import akshare as ak

    df = ak.futures_main_sina(symbol="AU0")
    if df is None or df.empty:
        return pd.DataFrame()
    rename = {
        "日期": "trade_date", "开盘价": "open", "最高价": "high",
        "最低价": "low", "收盘价": "close", "成交量": "volume",
        "持仓量": "open_interest",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["symbol"] = "AU0"
    return df.sort_values("trade_date")


def main():
    print("=" * 70)
    print("CMF Phase 1e — 黄金数据迁移 → DuckDB")
    print("=" * 70)

    if not SQLITE_PATH.exists():
        print(f"✗ SQLite数据库不存在: {SQLITE_PATH}")
        return

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute("""
        CREATE TABLE IF NOT EXISTS fundamentals_daily (
            source VARCHAR, trade_date DATE, value_col VARCHAR, value DOUBLE
        )
    """)

    macro_fetches = [
        ("FRED指标 (TIPS/RIR/VIX/EPU/GVZ)", "gold_fred", fetch_fred_indicators),
        ("美元指数", "gold_macro", fetch_usd_index),
        ("美债收益率", "gold_macro", fetch_treasury),
        ("CFTC黄金持仓", "gold_position", fetch_cftc),
        ("COMEX黄金库存", "gold_position", fetch_comex_inventory),
        ("SPDR黄金ETF持仓", "gold_position", fetch_etf),
        ("GPRD地缘政治风险", "gold_risk", fetch_gprd),
        ("美联储利率", "gold_macro", fetch_fed_rate),
        ("央行黄金储备", "gold_position", fetch_central_bank),
        ("伦敦金现价格", "gold_macro", fetch_gold_prices_spot),
    ]

    for desc, source, fn in macro_fetches:
        print(f"\n[迁移] {desc} → fundamentals_daily (source={source})")
        try:
            df = fn()
        except Exception as e:
            print(f"  ✗ FAIL: {type(e).__name__}: {str(e)[:80]}")
            continue
        if df.empty:
            print(f"  ⚠️ empty")
            continue
        df["source"] = source
        if "value_col" not in df.columns:
            df["value_col"] = "value"
        if "trade_date" not in df.columns and "date" in df.columns:
            df = df.rename(columns={"date": "trade_date"})
        out = df[["trade_date", "source", "value_col", "value"]].copy()
        out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.date
        for vc in out["value_col"].unique():
            con.execute(
                "DELETE FROM fundamentals_daily WHERE source = ? AND value_col = ?",
                [source, vc],
            )
        con.append("fundamentals_daily", out, by_name=True)
        print(f"  ✓ {len(out)} 行")

    print(f"\n[抓取] SHFE黄金主力 → futures_main_close_au")
    try:
        df_au = fetch_futures_au()
        if df_au.empty:
            print("  ⚠️ empty")
        else:
            con.execute("CREATE OR REPLACE TABLE futures_main_close_au AS SELECT * FROM df_au")
            print(f"  ✓ {len(df_au)} 行")
    except Exception as e:
        print(f"  ✗ FAIL: {type(e).__name__}: {str(e)[:80]}")

    print("\n" + "=" * 70)
    print("DuckDB 内容摘要 (gold相关):")
    for tbl in ["futures_main_close_au", "fundamentals_daily"]:
        n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        if tbl == "fundamentals_daily":
            n_gold = con.execute(
                "SELECT COUNT(*) FROM fundamentals_daily "
                "WHERE source LIKE 'gold_%'"
            ).fetchone()[0]
            print(f"  {tbl:30s}  {n:>7}行 (gold: {n_gold})")
        else:
            s = con.execute(
                f"SELECT MIN(trade_date), MAX(trade_date) FROM {tbl}"
            ).fetchone()
            print(f"  {tbl:30s}  {n:>7}行  [{s[0]}~{s[1]}]")

    sources = con.execute(
        "SELECT source, value_col, COUNT(*) AS n, MIN(trade_date), MAX(trade_date) "
        "FROM fundamentals_daily WHERE source LIKE 'gold_%' "
        "GROUP BY source, value_col ORDER BY source, value_col"
    ).fetchall()
    print("\n  黄金指标明细:")
    for s, vc, n, dmin, dmax in sources:
        print(f"    {s:15s} {vc:20s} {n:>5}行  [{dmin}~{dmax}]")

    con.close()
    print(f"\n✓ 已写入: {DB_PATH}")


if __name__ == "__main__":
    main()