#!/usr/bin/env python3
"""获取SHFE黄金(AU)合约的历史数据.

从akshare获取各合约日度数据, 存储到DuckDB.
用于后续渐进式换月价格构建.

模式:
    增量 (默认): 只刷新活跃窗口内的合约 (交割月 >= 上月, <= 未来12个月)
                 + 窗口内尚未入库的新合约, per-contract UPSERT (失败保留旧数据)
    --full:      全量重建 (2010至今全部合约, DROP+CREATE)
"""
import argparse
import time
import warnings

warnings.filterwarnings("ignore")

import akshare as ak
import duckdb
import pandas as pd

DB_PATH = "data/cmf/fundamental.duckdb"
TABLE_NAME = "futures_all_contracts_au"
COLS = ["trade_date", "contract", "open", "high", "low", "close",
        "volume", "open_interest", "settle"]

# SHFE黄金合约月份: 2/4/6/8/10/12 (偶数月)
CONTRACT_MONTHS = [2, 4, 6, 8, 10, 12]


def generate_contract_symbols(start_year=2010, end_year=2026):
    """生成所有可能的合约代码."""
    symbols = []
    for year in range(start_year, end_year + 1):
        for month in CONTRACT_MONTHS:
            symbols.append(f"AU{year % 100:02d}{month:02d}")
    return symbols


def parse_delivery_month(symbol):
    """AU2610 → 2026-10-01."""
    return pd.Timestamp(year=2000 + int(symbol[2:4]), month=int(symbol[4:6]), day=1)


def fetch_contract(symbol, max_retries=3):
    """获取单个合约数据."""
    for attempt in range(max_retries):
        try:
            df = ak.futures_zh_daily_sina(symbol=symbol)
            if df is not None and len(df) > 0:
                df = df.rename(columns={
                    "date": "trade_date",
                    "hold": "open_interest",
                    "settle": "settle",
                })
                df["contract"] = symbol
                df["trade_date"] = pd.to_datetime(df["trade_date"])
                return df[COLS]
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None
    return None


def table_exists(con):
    r = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [TABLE_NAME]
    ).fetchone()
    return r[0] > 0


def upsert_contract(con, symbol, df_new):
    """拉取成功才替换该合约: DELETE旧行 + INSERT新行 (per-contract事务)."""
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(f"DELETE FROM {TABLE_NAME} WHERE contract = ?", [symbol])
        con.execute(
            f"INSERT INTO {TABLE_NAME} ({', '.join(COLS)}) "
            f"SELECT {', '.join(COLS)} FROM df_new"
        )
        con.execute("COMMIT")
        return True
    except Exception as e:
        con.execute("ROLLBACK")
        print(f"    ✗ {symbol} 写入失败, 保留旧数据: {e}")
        return False


def main_full(con):
    """全量重建: DROP + CREATE (原行为)."""
    today = pd.Timestamp.today().normalize()
    symbols = generate_contract_symbols(2010, today.year + 1)
    print(f"待获取合约: {len(symbols)} 个")

    all_dfs = []
    success = 0
    empty = 0
    fail = 0

    for i, symbol in enumerate(symbols):
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(symbols)}")

        df = fetch_contract(symbol)
        if df is not None and len(df) > 0:
            all_dfs.append(df)
            success += 1
        elif df is not None:
            empty += 1
        else:
            fail += 1

        time.sleep(0.3)

    print(f"\n获取完成: 成功={success}, 空={empty}, 失败={fail}")

    if not all_dfs:
        print("✗ 没有获取到任何数据, 不写入 (保留旧表)")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["trade_date", "contract"]).reset_index(drop=True)

    print(f"总行数: {len(combined)}")
    print(f"合约数: {combined['contract'].nunique()}")
    print(f"日期范围: {combined['trade_date'].min().date()} ~ {combined['trade_date'].max().date()}")

    con.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    con.execute(f"CREATE TABLE {TABLE_NAME} AS SELECT * FROM combined")

    count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    contracts = con.execute(f"SELECT COUNT(DISTINCT contract) FROM {TABLE_NAME}").fetchone()[0]
    print(f"\n✓ 已写入 {TABLE_NAME}: {count} 行, {contracts} 个合约")


def main_incremental(con):
    """增量刷新: 只拉活跃窗口内的合约, per-contract UPSERT."""
    today = pd.Timestamp.today().normalize()
    stats = con.execute(
        f"SELECT contract, MAX(trade_date) AS max_date FROM {TABLE_NAME} GROUP BY contract"
    ).df()
    existing = set(stats["contract"])

    lo = today.replace(day=1) - pd.DateOffset(months=1)
    hi = today.replace(day=1) + pd.DateOffset(months=12)

    # 1. DB中活跃窗口内的合约 (数据可能还在更新)
    targets = [c for c in existing if lo <= parse_delivery_month(c) <= hi]
    # 2. 窗口内尚未入库的候选 (新上市/漏抓)
    candidates = generate_contract_symbols(today.year - 1, today.year + 1)
    targets += [c for c in candidates if c not in existing and lo <= parse_delivery_month(c) <= hi]
    targets = sorted(set(targets))

    print(f"活跃窗口: {lo.date()} ~ {hi.date()}")
    print(f"待刷新合约 {len(targets)} 个: {', '.join(targets)}")

    fetched = {}
    for symbol in targets:
        df = fetch_contract(symbol)
        if df is not None and len(df) > 0:
            fetched[symbol] = df
        time.sleep(0.3)

    print(f"拉取成功 {len(fetched)}/{len(targets)}")
    if not fetched:
        print("✗ 无新数据可写入 (网络失败或数据已最新)")
        return

    ok = 0
    for symbol, df in fetched.items():
        if upsert_contract(con, symbol, df):
            ok += 1
            print(f"  ✓ {symbol}: {len(df)} 行 "
                  f"({df['trade_date'].min().date()} ~ {df['trade_date'].max().date()})")

    result = con.execute(
        f"SELECT COUNT(*), COUNT(DISTINCT contract), MAX(trade_date) FROM {TABLE_NAME}"
    ).fetchone()
    print(f"\n✓ UPSERT {ok} 个合约")
    print(f"✓ {TABLE_NAME}: {result[0]} 行, {result[1]} 个合约, 最新日期 {result[2].date()}")


def main():
    parser = argparse.ArgumentParser(description="SHFE黄金合约数据获取")
    parser.add_argument("--full", action="store_true", help="全量重建 (默认增量)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"SHFE黄金合约数据获取 ({'全量' if args.full else '增量'}) → DuckDB")
    print("=" * 60)

    con = duckdb.connect(DB_PATH)
    try:
        if args.full or not table_exists(con):
            main_full(con)
        else:
            main_incremental(con)
    finally:
        con.close()
        print(f"\n✓ 已写入: {DB_PATH}")


if __name__ == "__main__":
    main()
