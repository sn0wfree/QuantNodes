#!/usr/bin/env python3
"""刷新黄金策略缺失/过期数据 (etf_tonnes / gprd / dxy).

数据源:
    - etf_tonnes:  akshare macro_cons_gold (SPDR ETF, corr=1.000 已验证)
    - gprd:        SQLite gprd_index (上游已更新至2026-07-31)
    - dxy:         akshare_currency_boc 延长 gold_macro.dxy 缺失区间

全部幂等: DELETE+APPEND / 只延长不覆盖.
"""
from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")

import akshare as ak
import duckdb
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")

DB_PATH = "data/cmf/fundamental.duckdb"
SQLITE_PATH = "/home/ll/Public/gold/database/gold_analysis.db"


def fetch_etf_tonnes() -> pd.DataFrame:
    """akshare SPDR ETF持仓 (总库存列 = 持仓吨数)."""
    df = ak.macro_cons_gold()
    df = df.rename(columns={"日期": "trade_date", "总库存": "value"})
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["source"] = "gold_position"
    df["value_col"] = "etf_tonnes"
    return df[["trade_date", "source", "value_col", "value"]]


def fetch_gprd_from_sqlite() -> pd.DataFrame:
    """从SQLite读取gprd (上游已用 --download-gprd 更新)."""
    import sqlite3
    con = sqlite3.connect(SQLITE_PATH)
    df = pd.read_sql("SELECT date, gprd_value AS value FROM gprd_index ORDER BY date", con)
    con.close()
    df["trade_date"] = pd.to_datetime(df["date"])
    df["source"] = "gold_risk"
    df["value_col"] = "gprd"
    return df[["trade_date", "source", "value_col", "value"]]


def fetch_dxy_extension() -> pd.DataFrame:
    """从akshare_currency_boc取dxy, 只取gold_macro.dxy缺失日期之后的部分."""
    con = duckdb.connect(DB_PATH, read_only=True)
    boc = con.execute("""
        SELECT trade_date, value FROM fundamentals_daily
        WHERE source='akshare_currency_boc' AND value_col='dxy'
        ORDER BY trade_date
    """).df()
    max_gold = con.execute("""
        SELECT MAX(trade_date) FROM fundamentals_daily
        WHERE source='gold_macro' AND value_col='dxy'
    """).fetchone()[0]
    con.close()

    boc["trade_date"] = pd.to_datetime(boc["trade_date"])
    ext = boc[boc["trade_date"] > pd.Timestamp(max_gold)].copy()
    ext["source"] = "gold_macro"
    ext["value_col"] = "dxy"
    return ext[["trade_date", "source", "value_col", "value"]]


def replace_series(con, df: pd.DataFrame, label: str):
    """DELETE+APPEND 幂等写入."""
    src = df["source"].iloc[0]
    vc = df["value_col"].iloc[0]
    con.execute(
        "DELETE FROM fundamentals_daily WHERE source=? AND value_col=?",
        [src, vc],
    )
    out = df.copy()
    out["trade_date"] = out["trade_date"].dt.date
    con.append("fundamentals_daily", out, by_name=True)
    n = con.execute(
        "SELECT COUNT(*) FROM fundamentals_daily WHERE source=? AND value_col=?",
        [src, vc],
    ).fetchone()[0]
    print(f"  {label}: 已写入 {n} 行")


def append_series(con, df: pd.DataFrame, label: str):
    """只APPEND新日期 (dxy延长用)."""
    out = df.copy()
    out["trade_date"] = out["trade_date"].dt.date
    con.append("fundamentals_daily", out, by_name=True)
    n = con.execute(
        "SELECT COUNT(*) FROM fundamentals_daily WHERE source='gold_macro' AND value_col='dxy'",
    ).fetchone()[0]
    print(f"  {label}: 追加 {len(out)} 行, gold_macro.dxy 现共 {n} 行")


def show_before(con):
    print("=== 更新前状态 ===")
    rows = con.execute("""
        SELECT source, value_col, MIN(trade_date), MAX(trade_date), COUNT(*)
        FROM fundamentals_daily
        WHERE (source='gold_position' AND value_col='etf_tonnes')
           OR (source='gold_risk' AND value_col='gprd')
           OR (source='gold_macro' AND value_col='dxy')
        GROUP BY source, value_col ORDER BY value_col
    """).fetchall()
    for r in rows:
        print(f"  {r[0]:15s} {r[1]:12s} {r[2]} ~ {r[3]}  {r[4]}行")


def show_after(con):
    print("\n=== 更新后状态 ===")
    rows = con.execute("""
        SELECT source, value_col, MIN(trade_date), MAX(trade_date), COUNT(*)
        FROM fundamentals_daily
        WHERE (source='gold_position' AND value_col='etf_tonnes')
           OR (source='gold_risk' AND value_col='gprd')
           OR (source='gold_macro' AND value_col='dxy')
        GROUP BY source, value_col ORDER BY value_col
    """).fetchall()
    for r in rows:
        print(f"  {r[0]:15s} {r[1]:12s} {r[2]} ~ {r[3]}  {r[4]}行")


def main():
    print("=" * 60)
    print("黄金策略数据刷新 (etf_tonnes / gprd / dxy)")
    print("=" * 60)

    # 先拉取全部数据 (网络操作, 失败即中止)
    print("\n[1/4] 拉取 akshare SPDR ETF...")
    etf = fetch_etf_tonnes()
    print(f"  获取 {len(etf)} 行, {etf['trade_date'].min().date()} ~ {etf['trade_date'].max().date()}")

    print("[2/4] 读取 SQLite gprd...")
    gprd = fetch_gprd_from_sqlite()
    print(f"  获取 {len(gprd)} 行, {gprd['trade_date'].min().date()} ~ {gprd['trade_date'].max().date()}")

    print("[3/4] 计算 dxy 延长区间...")
    dxy_ext = fetch_dxy_extension()
    if len(dxy_ext) > 0:
        print(f"  需追加 {len(dxy_ext)} 行, {dxy_ext['trade_date'].min().date()} ~ {dxy_ext['trade_date'].max().date()}")
    else:
        print("  无需延长")

    # 写入
    con = duckdb.connect(DB_PATH)
    print("\n[4/4] 写入 DuckDB...")
    show_before(con)

    print()
    replace_series(con, etf, "etf_tonnes")
    replace_series(con, gprd, "gprd")
    if len(dxy_ext) > 0:
        append_series(con, dxy_ext, "dxy延长")

    show_after(con)
    con.close()
    print("\n✓ 数据刷新完成")


if __name__ == "__main__":
    main()
