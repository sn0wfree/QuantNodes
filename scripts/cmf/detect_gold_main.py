#!/usr/bin/env python3
"""识别主力合约与换月检测.

每天选择OI最大的合约作为主力, 检测换月起点.
输出: gold_main_switches 表 (switch_date, old_contract, new_contract)
"""
import sys
import warnings

warnings.filterwarnings("ignore")

import duckdb
import pandas as pd
import numpy as np

DB_PATH = "data/cmf/fundamental.duckdb"


def main():
    print("=" * 60)
    print("主力合约识别与换月检测")
    print("=" * 60)

    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("""
        SELECT trade_date, contract, open, close, volume, open_interest
        FROM futures_all_contracts_au
        ORDER BY trade_date, contract
    """).df()
    con.close()

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df[df["open_interest"] > 0].copy()

    # 每天选择OI最大的合约作为主力
    idx = df.groupby("trade_date")["open_interest"].idxmax()
    main_df = df.loc[idx].copy()
    main_df = main_df.sort_values("trade_date").reset_index(drop=True)

    print(f"主力合约数据: {len(main_df)} 天")
    print(f"主力合约分布:")
    print(main_df["contract"].value_counts().head(10).to_string())

    # 检测换月: 主力合约变化
    main_df["prev_contract"] = main_df["contract"].shift(1)
    main_df["is_switch"] = main_df["contract"] != main_df["prev_contract"]
    switches = main_df[main_df["is_switch"]].copy()
    switches = switches[switches["prev_contract"].notna()]

    print(f"\n检测到 {len(switches)} 次换月")
    print("\n换月记录:")
    for _, r in switches.iterrows():
        gap = (r["open"] / main_df[main_df["trade_date"] < r["trade_date"]].iloc[-1]["close"] - 1) * 100
        print(f"  {r['trade_date'].strftime('%Y-%m-%d')}  "
              f"{r['prev_contract']} → {r['contract']}  "
              f"OI={r['open_interest']:,}  gap={gap:+.2f}%")

    # 保存主力合约序列
    main_output = main_df[["trade_date", "contract", "open", "close",
                           "volume", "open_interest"]].copy()
    main_output.columns = ["trade_date", "main_contract", "open", "close",
                           "volume", "open_interest"]

    # 保存换月记录
    switch_output = switches[["trade_date", "prev_contract", "contract",
                              "open_interest"]].copy()
    switch_output.columns = ["switch_date", "old_contract", "new_contract", "oi_at_switch"]

    # 写入DuckDB
    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS gold_main_contracts")
    con.execute("DROP TABLE IF EXISTS gold_main_switches")
    con.execute("""
        CREATE TABLE gold_main_contracts AS
        SELECT * FROM main_output
    """)
    con.execute("""
        CREATE TABLE gold_main_switches AS
        SELECT * FROM switch_output
    """)

    count1 = con.execute("SELECT COUNT(*) FROM gold_main_contracts").fetchone()[0]
    count2 = con.execute("SELECT COUNT(*) FROM gold_main_switches").fetchone()[0]
    print(f"\n✓ gold_main_contracts: {count1} 行")
    print(f"✓ gold_main_switches: {count2} 行")

    con.close()


if __name__ == "__main__":
    main()
