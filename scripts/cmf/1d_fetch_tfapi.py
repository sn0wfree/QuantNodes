#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/1d_fetch_tfapi.py — 天风期货 API 数据补齐.

数据源: 天风期货公共 token (内置, 无需用户配置)

覆盖缺口:
  - 洋山铜溢价 / 铜精矿TC (Get_Data)
  - 豆粕库存 / 大豆压榨利润 / 自繁自养利润 (Get_Data)
  - SHFE库存历史 (Get_Data)
  - COMEX 铜 Managed Money / ICE Brent Managed Money (GetCftcReports)
  - 国内 DCE 豆粕净多头 / 上期所铜净多头 (GetFutHolding)

输出: data/cmf/fundamental.duckdb 新增3表:
  - tfapi_indicators (trade_date, index_name, value, source)
  - tfapi_cftc (report_date, market, mm_net_long, oi, ...)
  - tfapi_holding (trade_date, symbol, broker, long, short, net_long)
"""
from __future__ import annotations

import json
import time
import warnings
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

import duckdb
import pandas as pd
import requests
from CodersWheel.QuickTool.file_cache import file_cache

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "data" / "cmf" / "fundamental.duckdb"

# 天风 FYAPI 内置 token (FYAPI.py 第23行公开测试 token)
TOKEN = "rCslVWmoIrlxp3Ud"
BASEURL = "https://fengyun.thanf.com/fypta/ta/"


@file_cache(enable_cache=True, granularity='H')
def _get_data_cached(name: str, begin: str, end: str) -> pd.DataFrame:
    """天风 Get_Data — 核心调用 (带缓存).

    注意: 返回数据需除以3.15还原真实值 (demo.py第49行)
    但"库存因子/基差因子/利润因子"不需要除以3.15 (demo.py第50行)
    这里返回原始值, 由调用方判断是否需要转换.
    """
    params = {"username": TOKEN, "pwd": "pwd", "symbol": name,
              "begin": begin, "end": end, "algo": "b"}
    try:
        r = requests.post(BASEURL + "fyapi.aspx", params=params, timeout=60)
        if r.status_code == 200:
            content = json.loads(r.text, object_pairs_hook=OrderedDict)
            if isinstance(content, list) and len(content) > 0 and "errcode" not in content[0]:
                df = pd.DataFrame(content)
                # 除以3.15还原真实值 (库存/基差/利润因子除外)
                if "指标值" in df.columns:
                    skip_keywords = ["库存因子", "基差因子", "利润因子"]
                    if not any(kw in name for kw in skip_keywords):
                        df["指标值"] = pd.to_numeric(df["指标值"], errors="coerce") / 3.15
                return df
    except Exception:
        pass
    return pd.DataFrame()


@file_cache(enable_cache=True, granularity='H')
def _get_cftc_cached(begin: str, end: str) -> pd.DataFrame:
    """天风 CFTC COT — 核心调用 (带缓存)."""
    try:
        r = requests.post(BASEURL + "cftc_DisaggregatedFuturesOnlyReports.ashx",
                          params={"begin": begin, "end": end, "username": TOKEN},
                          timeout=90)
        if r.status_code == 200:
            content = json.loads(r.text, object_pairs_hook=OrderedDict)
            if isinstance(content, list) and len(content) > 0 and "errcode" not in content[0]:
                return pd.DataFrame(content)
    except Exception:
        pass
    return pd.DataFrame()


@file_cache(enable_cache=True, granularity='H')
def _get_holding_cached(symbol: str, begin: str, end: str) -> pd.DataFrame:
    """天风 GetFutHolding — 核心调用 (带缓存)."""
    try:
        r = requests.post(BASEURL + "fut_holding.ashx",
                          data={"begin": begin, "end": end, "symbol": symbol,
                                "broker": "", "username": TOKEN},
                          timeout=60)
        if r.status_code == 200:
            content = json.loads(r.text, object_pairs_hook=OrderedDict)
            if isinstance(content, list) and len(content) > 0 and "errcode" not in content[0]:
                return pd.DataFrame(content)
    except Exception:
        pass
    return pd.DataFrame()

# ── 核心指标 (按优先级) ──────────────────────────────────────────────────────
INDICATORS = [
    # 铜 — 4 项新增/替代
    ("洋山铜溢价",                              "2012-11-20"),
    ("现货:中国铜冶炼厂:粗炼费(TC)",            "2007-09-30"),
    ("LME库存总计_铜",                          "2010-01-04"),
    ("SHFE库存小计",                            "2010-01-03"),
    ("COMEX:铜:库存",                           "2016-07-08"),
    # 豆粕 — 4 项 (替代 v2 的价格代理)
    ("江苏:压榨利润:大豆(进口)",                 "2005-04-04"),
    ("豆粕库存_中国",                            "2011-07-10"),
    ("大豆库存_中国",                            "2011-07-10"),
    ("养殖利润_自繁自养",                        "2010-01-08"),
]

# ── 期货持仓 (国内) ─────────────────────────────────────────────────────────
HOLDING_SYMBOLS = [
    ("CU", "shfe", "上期所铜"),
    ("M",  "dce",  "大商所豆粕"),
]

# ── CFTC COT 市场 (用于 §3.3 拥挤度 D 项) ──────────────────────────────────
CFTC_MARKETS = [
    "COPPER- #1 - COMMODITY EXCHANGE INC.",
    "BRENT LAST DAY - NEW YORK MERCANTILE EXCHANGE",
    "WTI-BRENT CALENDAR - NEW YORK MERCANTILE EXCHANGE",
    "CRUDE OIL, LIGHT SWEET-WTI - ICE FUTURES EUROPE",
]


def fetch_one_indicator(name: str, begin: str, end: str | None = None,
                         retries: int = 3, timeout: int = 60) -> pd.DataFrame:
    """天风 Get_Data — 单指标 (内部调用缓存版本)."""
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for attempt in range(retries):
        df = _get_data_cached(name, begin, end)
        if not df.empty:
            return df
        if attempt < retries - 1:
            time.sleep(2)
    return pd.DataFrame()


def fetch_cftc_range(begin: str, end: str | None = None,
                      retries: int = 3, timeout: int = 90) -> pd.DataFrame:
    """天风 GetCftcReports — CFTC 周报 (内部调用缓存版本)."""
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for attempt in range(retries):
        df = _get_cftc_cached(begin, end)
        if not df.empty:
            return df
        if attempt < retries - 1:
            time.sleep(2)
    return pd.DataFrame()


def fetch_holding_range(symbol: str, begin: str, end: str,
                         retries: int = 3, timeout: int = 60) -> pd.DataFrame:
    """天风 GetFutHolding — 单品种持仓 (内部调用缓存版本)."""
    for attempt in range(retries):
        df = _get_holding_cached(symbol, begin, end)
        if not df.empty:
            return df
        if attempt < retries - 1:
            time.sleep(2)
    return pd.DataFrame()


def chunk_date_range(begin: str, end: str, chunk_days: int = 365) -> list[tuple[str, str]]:
    """把大日期区间切成多个小子区间, 单次请求避免 timeout."""
    start = datetime.strptime(begin, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    chunks = []
    cur = start
    while cur < end_dt:
        nxt = min(cur + timedelta(days=chunk_days), end_dt)
        chunks.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt
    return chunks


def upsert_tfapi_indicators(con, name: str, df: pd.DataFrame) -> int:
    """写入 tfapi_indicators (trade_date, index_name, value, source)."""
    if df.empty:
        return 0
    con.execute(
        "DELETE FROM tfapi_indicators WHERE index_name = ?",
        [name],
    )
    if "指标时间" in df.columns:
        df = df.rename(columns={"指标时间": "trade_date", "指标值": "value", "指标名称": "index_name"})
    df = df[["trade_date", "index_name", "value"]].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["source"] = "tfapi_get_data"
    if len(df) == 0:
        return 0
    con.append("tfapi_indicators", df, by_name=True)
    return len(df)


def upsert_tfapi_cftc(con, df: pd.DataFrame) -> int:
    """写入 tfapi_cftc (含 Managed Money 长/短/净/OI)."""
    if df.empty:
        return 0
    con.execute("DELETE FROM tfapi_cftc")
    keep_cols = [c for c in [
        "Market_and_Exchange_Names", "Report_Date_as_MM_DD_YYYY",
        "CFTC_Contract_Market_Code", "CFTC_Market_Code", "CFTC_Commodity_Code",
        "Open_Interest_All",
        "M_Money_Positions_Long_ALL", "M_Money_Positions_Short_ALL",
        "M_Money_Positions_Spread_ALL",
        "Prod_Merc_Positions_Long_ALL", "Prod_Merc_Positions_Short_ALL",
        "Swap_Positions_Long_All", "Swap__Positions_Short_All",
    ] if c in df.columns]
    out = df[keep_cols].copy()
    out = out.rename(columns={
        "Market_and_Exchange_Names": "market",
        "Report_Date_as_MM_DD_YYYY": "report_date",
        "Open_Interest_All": "oi",
        "M_Money_Positions_Long_ALL": "mm_long",
        "M_Money_Positions_Short_ALL": "mm_short",
        "M_Money_Positions_Spread_ALL": "mm_spread",
        "Prod_Merc_Positions_Long_ALL": "pm_long",
        "Prod_Merc_Positions_Short_ALL": "pm_short",
        "Swap_Positions_Long_All": "swap_long",
        "Swap__Positions_Short_All": "swap_short",
    })
    out["report_date"] = pd.to_datetime(out["report_date"]).dt.date
    out["mm_net"] = out["mm_long"].fillna(0) - out["mm_short"].fillna(0) - out["mm_spread"].fillna(0)
    out["mm_net_pct"] = out["mm_net"] / out["oi"].replace(0, 1)
    out = out.dropna(subset=["oi", "report_date"])
    con.append("tfapi_cftc", out, by_name=True)
    return len(out)


def upsert_tfapi_holding(con, symbol: str, df: pd.DataFrame) -> int:
    """写入 tfapi_holding (per broker per trade_date per contract)."""
    if df.empty:
        return 0
    con.execute("DELETE FROM tfapi_holding WHERE symbol_prefix = ?", [symbol])
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["symbol_prefix"] = symbol
    con.append("tfapi_holding", df, by_name=True)
    return len(df)


def main():
    print("=" * 70)
    print("CMF Phase 7 — 天风期货 FYAPI 数据补齐")
    print("=" * 70)
    print(f"Token: {TOKEN[:4]}***{TOKEN[-4:]}")
    print()

    con = duckdb.connect(str(DB_PATH))

    con.execute("""
        CREATE TABLE IF NOT EXISTS tfapi_indicators (
            trade_date DATE, index_name VARCHAR, value DOUBLE, source VARCHAR
        );
        CREATE OR REPLACE TABLE tfapi_cftc (
            market VARCHAR,
            report_date DATE,
            CFTC_Contract_Market_Code VARCHAR,
            CFTC_Market_Code VARCHAR,
            CFTC_Commodity_Code VARCHAR,
            oi DOUBLE,
            mm_long DOUBLE, mm_short DOUBLE, mm_spread DOUBLE, mm_net DOUBLE, mm_net_pct DOUBLE,
            pm_long DOUBLE, pm_short DOUBLE, swap_long DOUBLE, swap_short DOUBLE
        );
        CREATE TABLE IF NOT EXISTS tfapi_holding (
            trade_date DATE, symbol VARCHAR, broker VARCHAR,
            volume DOUBLE, long DOUBLE, short DOUBLE, net_long DOUBLE,
            long_change DOUBLE, short_change DOUBLE,
            volume_change DOUBLE, symbol_prefix VARCHAR
        );
    """)

    # ── 1) Get_Data 12 指标 ──────────────────────────────────────────────
    print("[1/3] Get_Data — 12 个核心指标")
    print("-" * 70)
    today = datetime.now().strftime("%Y-%m-%d")
    for name, begin in INDICATORS:
        print(f"  拉取: {name:35s} (from {begin})")
        chunks = chunk_date_range(begin, today, chunk_days=365)
        all_df = []
        for i, (s, e) in enumerate(chunks):
            sub = fetch_one_indicator(name, s, e)
            if not sub.empty:
                all_df.append(sub)
            time.sleep(0.4)
        if all_df:
            combined = pd.concat(all_df, ignore_index=True)
            combined = combined.drop_duplicates(subset=["指标时间"], keep="last")
            n = upsert_tfapi_indicators(con, name, combined)
            print(f"      ✓ {n} 行, 范围 {combined['指标时间'].min()[:10]} ~ {combined['指标时间'].max()[:10]}")
        else:
            print(f"      ⚠️ 无数据")

    print()

    # ── 2) GetCftcReports (历史) ───────────────────────────────────────────
    print("[2/3] GetCftcReports — CFTC COT 历史 (目标 2010+)")
    print("-" * 70)
    chunks = chunk_date_range("2010-01-01", today, chunk_days=180)
    all_cftc = []
    for s, e in chunks:
        sub = fetch_cftc_range(s, e)
        if not sub.empty:
            all_cftc.append(sub)
            print(f"      [{s}~{e}] +{len(sub)} 行")
        time.sleep(0.5)
    if all_cftc:
        cftc_combined = pd.concat(all_cftc, ignore_index=True)
        cftc_combined = cftc_combined.drop_duplicates(
            subset=["Market_and_Exchange_Names", "Report_Date_as_MM_DD_YYYY"], keep="last"
        )
        n = upsert_tfapi_cftc(con, cftc_combined)
        print(f"      ✓ {n} 行, 范围 {cftc_combined['Report_Date_as_MM_DD_YYYY'].min()[:10]} ~ {cftc_combined['Report_Date_as_MM_DD_YYYY'].max()[:10]}")
    print()

    # ── 3) GetFutHolding ───────────────────────────────────────────────────
    print("[3/3] GetFutHolding — 国内持仓")
    print("-" * 70)
    for sym, exch, label in HOLDING_SYMBOLS:
        print(f"  拉取: {sym} ({label})")
        chunks = chunk_date_range("2020-01-01", today, chunk_days=180)
        all_h = []
        for s, e in chunks:
            sub = fetch_holding_range(sym, s, e)
            if not sub.empty:
                all_h.append(sub)
            time.sleep(0.5)
        if all_h:
            comb = pd.concat(all_h, ignore_index=True)
            for col in ['long', 'short', 'volume', 'long_change', 'short_change', 'volume_change']:
                if col in comb.columns:
                    comb[col] = pd.to_numeric(comb[col], errors='coerce')
            comb["net_long"] = comb["long"] - comb["short"]
            n = upsert_tfapi_holding(con, sym, comb)
            print(f"      ✓ {n} 行, 唯一合约 {comb['symbol'].nunique()}个")
        else:
            print(f"      ⚠️ 无数据")
    print()

    # ── 摘要 ────────────────────────────────────────────────────────────────
    print("=" * 70)
    print("DuckDB 摘要:")
    for tbl in ["tfapi_indicators", "tfapi_cftc", "tfapi_holding"]:
        n = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        date_col = "report_date" if tbl == "tfapi_cftc" else "trade_date"
        s = con.execute(f"SELECT MIN({date_col}), MAX({date_col}) FROM {tbl}").fetchone()
        print(f"  {tbl:30s}  {n:>7}行  [{s[0]}~{s[1]}]")

    print()
    print("tfapi_indicators 按 index_name 行数:")
    for r in con.execute("""
        SELECT index_name, COUNT(*), MIN(trade_date), MAX(trade_date)
        FROM tfapi_indicators GROUP BY index_name ORDER BY index_name
    """).fetchall():
        print(f"  {r[0]:35s}  {r[1]:>5}行  [{r[2]}~{r[3]}]")

    print()
    print("tfapi_cftc 关键市场:")
    for r in con.execute("""
        SELECT market, COUNT(*), MIN(report_date), MAX(report_date)
        FROM tfapi_cftc
        WHERE market LIKE '%COPPER%' OR market LIKE '%BRENT%' OR market LIKE '%WTI%'
        GROUP BY market ORDER BY market
    """).fetchall():
        print(f"  {r[0][:50]:50s}  {r[1]:>5}行  [{r[2]}~{r[3]}]")

    con.close()


if __name__ == "__main__":
    main()