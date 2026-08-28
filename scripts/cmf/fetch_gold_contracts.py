#!/usr/bin/env python3
"""获取所有SHFE黄金(AU)合约的历史数据.

从akshare获取各合约日度数据, 存储到DuckDB.
用于后续渐进式换月价格构建.
"""
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import akshare as ak
import duckdb
import pandas as pd

DB_PATH = "data/cmf/fundamental.duckdb"
TABLE_NAME = "futures_all_contracts_au"

# SHFE黄金合约月份: 2/4/6/8/10/12 (偶数月)
# 每年6个合约, 从2010年开始
CONTRACT_MONTHS = [2, 4, 6, 8, 10, 12]


def generate_contract_symbols(start_year=2010, end_year=2026):
    """生成所有可能的合约代码."""
    symbols = []
    for year in range(start_year, end_year + 1):
        for month in CONTRACT_MONTHS:
            symbols.append(f"AU{year % 100:02d}{month:02d}")
    return symbols


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
                return df[["trade_date", "contract", "open", "high", "low",
                           "close", "volume", "open_interest", "settle"]]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return None
    return None


def main():
    print("=" * 60)
    print("SHFE黄金合约数据获取 → DuckDB")
    print("=" * 60)

    symbols = generate_contract_symbols(2010, 2026)
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

        # 避免请求过快
        time.sleep(0.3)

    print(f"\n获取完成: 成功={success}, 空={empty}, 失败={fail}")

    if not all_dfs:
        print("✗ 没有获取到任何数据")
        return

    # 合并所有数据
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["trade_date", "contract"]).reset_index(drop=True)

    print(f"总行数: {len(combined)}")
    print(f"合约数: {combined['contract'].nunique()}")
    print(f"日期范围: {combined['trade_date'].min().date()} ~ {combined['trade_date'].max().date()}")

    # 写入DuckDB
    con = duckdb.connect(DB_PATH)
    con.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    con.execute(f"""
        CREATE TABLE {TABLE_NAME} AS
        SELECT * FROM combined
    """)

    # 验证
    count = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    contracts = con.execute(f"SELECT COUNT(DISTINCT contract) FROM {TABLE_NAME}").fetchone()[0]
    print(f"\n✓ 已写入 {TABLE_NAME}: {count} 行, {contracts} 个合约")

    # 统计每个合约的数据量
    stats = con.execute(f"""
        SELECT contract, COUNT(*) as n,
               MIN(trade_date) as start_date,
               MAX(trade_date) as end_date,
               MAX(open_interest) as max_oi
        FROM {TABLE_NAME}
        GROUP BY contract
        ORDER BY contract
    """).df()

    print("\n各合约数据量:")
    print(stats.to_string(index=False))

    con.close()
    print(f"\n✓ 已写入: {DB_PATH}")


if __name__ == "__main__":
    main()
