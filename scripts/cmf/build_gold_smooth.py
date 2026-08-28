#!/usr/bin/env python3
"""渐进式换月价格构建 (4天线性插值).

检测换月期, 每天切换25%仓位到新合约, 生成平滑连续价格.
"""
import sys
import warnings

warnings.filterwarnings("ignore")

import duckdb
import pandas as pd
import numpy as np

DB_PATH = "data/cmf/fundamental.duckdb"
ROLLOVER_DAYS = 4  # 换月周期天数


def main():
    print("=" * 60)
    print("渐进式换月价格构建 (4天线性插值)")
    print("=" * 60)

    con = duckdb.connect(DB_PATH, read_only=True)

    # 加载所有合约数据
    all_contracts = con.execute("""
        SELECT trade_date, contract, open, close, volume, open_interest
        FROM futures_all_contracts_au
        ORDER BY trade_date, contract
    """).df()
    all_contracts["trade_date"] = pd.to_datetime(all_contracts["trade_date"])

    # 加载主力合约序列
    main_df = con.execute("""
        SELECT trade_date, main_contract, open, close, volume, open_interest
        FROM gold_main_contracts
        ORDER BY trade_date
    """).df()
    main_df["trade_date"] = pd.to_datetime(main_df["trade_date"])

    # 加载换月记录
    switches = con.execute("""
        SELECT switch_date, old_contract, new_contract, oi_at_switch
        FROM gold_main_switches
        ORDER BY switch_date
    """).df()
    switches["switch_date"] = pd.to_datetime(switches["switch_date"])

    con.close()

    print(f"主力合约数据: {len(main_df)} 天")
    print(f"换月记录: {len(switches)} 次")

    # 构建换月期窗口
    # 换月期: switch_date 前后各 ROLLOVER_DAYS/2 天
    rollover_windows = []
    for _, sw in switches.iterrows():
        switch_date = sw["switch_date"]
        old_contract = sw["old_contract"]
        new_contract = sw["new_contract"]

        # 找换月前后的交易日
        dates_before = main_df[main_df["trade_date"] < switch_date]["trade_date"].tail(ROLLOVER_DAYS // 2)
        dates_after = main_df[main_df["trade_date"] >= switch_date]["trade_date"].head(ROLLOVER_DAYS - ROLLOVER_DAYS // 2)

        window_dates = list(dates_before) + list(dates_after)
        if len(window_dates) < 2:
            continue

        rollover_windows.append({
            "switch_date": switch_date,
            "old_contract": old_contract,
            "new_contract": new_contract,
            "window_dates": window_dates,
        })

    print(f"换月窗口: {len(rollover_windows)} 个")

    # 构建调整后价格
    result_rows = []
    rollover_dates = set()

    for window in rollover_windows:
        switch_date = window["switch_date"]
        old_contract = window["old_contract"]
        new_contract = window["new_contract"]
        window_dates = window["window_dates"]

        # 获取旧合约和新合约的数据
        old_data = all_contracts[all_contracts["contract"] == old_contract].set_index("trade_date")
        new_data = all_contracts[all_contracts["contract"] == new_contract].set_index("trade_date")

        n_days = len(window_dates)
        for i, date in enumerate(window_dates):
            # 权重: 线性从1.0(旧)降到0.0(新)
            w_old = 1.0 - (i / (n_days - 1)) if n_days > 1 else 0.0
            w_new = 1.0 - w_old

            old_row = old_data.loc[date] if date in old_data.index else None
            new_row = new_data.loc[date] if date in new_data.index else None

            # 如果某一方没有数据, 用另一方
            if old_row is None and new_row is None:
                continue
            elif old_row is None:
                w_old, w_new = 0.0, 1.0
            elif new_row is None:
                w_old, w_new = 1.0, 0.0

            # 加权平均
            if old_row is not None and new_row is not None:
                eff_open = w_old * old_row["open"] + w_new * new_row["open"]
                eff_close = w_old * old_row["close"] + w_new * new_row["close"]
                eff_volume = int(w_old * old_row["volume"] + w_new * new_row["volume"])
                eff_oi = int(w_old * old_row["open_interest"] + w_new * new_row["open_interest"])
            elif old_row is not None:
                eff_open, eff_close = old_row["open"], old_row["close"]
                eff_volume, eff_oi = int(old_row["volume"]), int(old_row["open_interest"])
            else:
                eff_open, eff_close = new_row["open"], new_row["close"]
                eff_volume, eff_oi = int(new_row["volume"]), int(new_row["open_interest"])

            result_rows.append({
                "trade_date": date,
                "open": eff_open,
                "close": eff_close,
                "volume": eff_volume,
                "open_interest": eff_oi,
                "contract": f"{old_contract}→{new_contract}",
                "is_rollover": True,
                "w_old": w_old,
                "w_new": w_new,
            })
            rollover_dates.add(date)

    # 添加非换月期的数据
    non_rollover = main_df[~main_df["trade_date"].isin(rollover_dates)].copy()
    non_rollover["contract"] = non_rollover["main_contract"]
    non_rollover["is_rollover"] = False
    non_rollover["w_old"] = 1.0
    non_rollover["w_new"] = 0.0
    non_rollover = non_rollover[["trade_date", "open", "close", "volume",
                                  "open_interest", "contract", "is_rollover",
                                  "w_old", "w_new"]]

    # 合并
    result = pd.concat([pd.DataFrame(result_rows), non_rollover], ignore_index=True)
    result = result.sort_values("trade_date").reset_index(drop=True)

    # 计算收益率
    result["ret"] = result["close"].pct_change()
    result["gap"] = result["open"] / result["close"].shift(1) - 1

    print(f"\n调整后数据: {len(result)} 行")
    print(f"换月期天数: {result['is_rollover'].sum()}")
    print(f"非换月期天数: {(~result['is_rollover']).sum()}")

    # 对比: 用原始AU0数据计算收益率
    print("\n=== 换月期跳空对比 ===")
    rollover_gaps = result[result["is_rollover"]]["gap"]
    non_rollover_gaps = result[~result["is_rollover"]]["gap"]
    print(f"换月期 gap 均值: {rollover_gaps.mean()*100:.3f}%")
    print(f"换月期 gap 标准差: {rollover_gaps.std()*100:.3f}%")
    print(f"非换月期 gap 均值: {non_rollover_gaps.mean()*100:.3f}%")
    print(f"非换月期 gap 标准差: {non_rollover_gaps.std()*100:.3f}%")

    # 保存到DuckDB
    output = result[["trade_date", "open", "close", "volume", "open_interest",
                      "contract", "is_rollover"]].copy()

    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS futures_main_close_au_adj")
    con.execute("""
        CREATE TABLE futures_main_close_au_adj AS
        SELECT * FROM output
    """)
    count = con.execute("SELECT COUNT(*) FROM futures_main_close_au_adj").fetchone()[0]
    print(f"\n✓ futures_main_close_au_adj: {count} 行")
    con.close()


if __name__ == "__main__":
    main()
