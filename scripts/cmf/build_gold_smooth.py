#!/usr/bin/env python3
"""渐进式换月价格构建 (4天线性插值).

检测换月期, 每天切换25%仓位到新合约, 生成平滑连续价格.

核心函数:
    build_rollover_windows() — 构建换月窗口
    compute_smooth_prices() — 计算平滑价格
    build_smooth_continuous() — 主入口, 组合以上两步
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

DB_PATH = "data/cmf/fundamental.duckdb"
ROLLOVER_DAYS = 4  # 换月周期天数


@dataclass
class RolloverWindow:
    """换月窗口."""
    switch_date: pd.Timestamp
    old_contract: str
    new_contract: str
    window_dates: list[pd.Timestamp]


def build_rollover_windows(
    main_dates: pd.DatetimeIndex,
    switches: pd.DataFrame,
    rollover_days: int = ROLLOVER_DAYS,
) -> list[RolloverWindow]:
    """构建换月窗口.

    Args:
        main_dates: 主力合约的交易日序列
        switches: 换月记录, 需包含 switch_date, old_contract, new_contract 列
        rollover_days: 换月周期天数

    Returns:
        换月窗口列表
    """
    windows = []
    for _, sw in switches.iterrows():
        switch_date = sw["switch_date"]
        old_contract = sw["old_contract"]
        new_contract = sw["new_contract"]

        dates_series = pd.Series(main_dates)
        dates_before = dates_series[dates_series < switch_date].tail(rollover_days // 2)
        dates_after = dates_series[dates_series >= switch_date].head(rollover_days - rollover_days // 2)

        window_dates = list(dates_before) + list(dates_after)
        if len(window_dates) < 2:
            continue

        windows.append(RolloverWindow(
            switch_date=switch_date,
            old_contract=old_contract,
            new_contract=new_contract,
            window_dates=window_dates,
        ))
    return windows


def compute_rollover_weights(
    n_days: int,
    day_index: int,
) -> tuple[float, float]:
    """计算换月期某天的权重.

    Args:
        n_days: 换月窗口总天数
        day_index: 当前天索引 (0-based)

    Returns:
        (w_old, w_new) 权重对
    """
    if n_days <= 1:
        return 0.0, 1.0
    w_old = 1.0 - (day_index / (n_days - 1))
    w_new = 1.0 - w_old
    return w_old, w_new


def blend_row(
    old_row: Optional[pd.Series],
    new_row: Optional[pd.Series],
    w_old: float,
    w_new: float,
) -> Optional[dict]:
    """加权平均新旧合约数据.

    Args:
        old_row: 旧合约数据 (可能为None)
        new_row: 新合约数据 (可能为None)
        w_old: 旧合约权重
        w_new: 新合约权重

    Returns:
        加权平均后的数据字典, 或None (双方都无数据)
    """
    if old_row is None and new_row is None:
        return None
    elif old_row is None:
        return {
            "open": new_row["open"],
            "close": new_row["close"],
            "volume": int(new_row["volume"]),
            "open_interest": int(new_row["open_interest"]),
            "w_old": 0.0,
            "w_new": 1.0,
        }
    elif new_row is None:
        return {
            "open": old_row["open"],
            "close": old_row["close"],
            "volume": int(old_row["volume"]),
            "open_interest": int(old_row["open_interest"]),
            "w_old": 1.0,
            "w_new": 0.0,
        }
    else:
        return {
            "open": w_old * old_row["open"] + w_new * new_row["open"],
            "close": w_old * old_row["close"] + w_new * new_row["close"],
            "volume": int(w_old * old_row["volume"] + w_new * new_row["volume"]),
            "open_interest": int(w_old * old_row["open_interest"] + w_new * new_row["open_interest"]),
            "w_old": w_old,
            "w_new": w_new,
        }


def compute_smooth_prices(
    all_contracts: pd.DataFrame,
    main_df: pd.DataFrame,
    switches: pd.DataFrame,
    rollover_days: int = ROLLOVER_DAYS,
) -> pd.DataFrame:
    """计算渐进式换月平滑价格.

    Args:
        all_contracts: 所有合约数据, 需含 trade_date, contract, open, close, volume, open_interest
        main_df: 主力合约序列, 需含 trade_date, main_contract, open, close, volume, open_interest
        switches: 换月记录, 需含 switch_date, old_contract, new_contract
        rollover_days: 换月周期天数

    Returns:
        调整后的连续价格DataFrame
    """
    # 构建换月窗口
    windows = build_rollover_windows(
        main_dates=main_df["trade_date"],
        switches=switches,
        rollover_days=rollover_days,
    )

    # 构建调整后价格
    result_rows = []
    rollover_dates = set()

    for window in windows:
        old_contract = window.old_contract
        new_contract = window.new_contract
        window_dates = window.window_dates

        old_data = all_contracts[all_contracts["contract"] == old_contract].set_index("trade_date")
        new_data = all_contracts[all_contracts["contract"] == new_contract].set_index("trade_date")

        n_days = len(window_dates)
        for i, date in enumerate(window_dates):
            w_old, w_new = compute_rollover_weights(n_days, i)
            old_row = old_data.loc[date] if date in old_data.index else None
            new_row = new_data.loc[date] if date in new_data.index else None

            blended = blend_row(old_row, new_row, w_old, w_new)
            if blended is None:
                continue

            result_rows.append({
                "trade_date": date,
                "open": blended["open"],
                "close": blended["close"],
                "volume": blended["volume"],
                "open_interest": blended["open_interest"],
                "contract": f"{old_contract}→{new_contract}",
                "is_rollover": True,
                "w_old": blended["w_old"],
                "w_new": blended["w_new"],
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

    return result


def load_data_from_db(db_path: str = DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """从DuckDB加载数据.

    Returns:
        (all_contracts, main_df, switches) 三元组
    """
    con = duckdb.connect(db_path, read_only=True)

    all_contracts = con.execute("""
        SELECT trade_date, contract, open, close, volume, open_interest
        FROM futures_all_contracts_au
        ORDER BY trade_date, contract
    """).df()
    all_contracts["trade_date"] = pd.to_datetime(all_contracts["trade_date"])

    main_df = con.execute("""
        SELECT trade_date, main_contract, open, close, volume, open_interest
        FROM gold_main_contracts
        ORDER BY trade_date
    """).df()
    main_df["trade_date"] = pd.to_datetime(main_df["trade_date"])

    switches = con.execute("""
        SELECT switch_date, old_contract, new_contract, oi_at_switch
        FROM gold_main_switches
        ORDER BY switch_date
    """).df()
    switches["switch_date"] = pd.to_datetime(switches["switch_date"])

    con.close()
    return all_contracts, main_df, switches


def build_smooth_continuous(db_path: str = DB_PATH) -> pd.DataFrame:
    """主入口: 从数据库加载数据并构建平滑连续价格.

    Returns:
        调整后的连续价格DataFrame
    """
    all_contracts, main_df, switches = load_data_from_db(db_path)
    return compute_smooth_prices(all_contracts, main_df, switches)


def main():
    print("=" * 60)
    print("渐进式换月价格构建 (4天线性插值)")
    print("=" * 60)

    all_contracts, main_df, switches = load_data_from_db()
    print(f"主力合约数据: {len(main_df)} 天")
    print(f"换月记录: {len(switches)} 次")

    result = compute_smooth_prices(all_contracts, main_df, switches)

    print(f"\n调整后数据: {len(result)} 行")
    print(f"换月期天数: {result['is_rollover'].sum()}")
    print(f"非换月期天数: {(~result['is_rollover']).sum()}")

    rollover_gaps = result[result["is_rollover"]]["gap"]
    non_rollover_gaps = result[~result["is_rollover"]]["gap"]
    print(f"\n换月期 gap 均值: {rollover_gaps.mean()*100:.3f}%")
    print(f"换月期 gap 标准差: {rollover_gaps.std()*100:.3f}%")
    print(f"非换月期 gap 均值: {non_rollover_gaps.mean()*100:.3f}%")
    print(f"非换月期 gap 标准差: {non_rollover_gaps.std()*100:.3f}%")

    # 保存到DuckDB
    output = result[["trade_date", "open", "close", "volume", "open_interest",
                      "contract", "is_rollover"]].copy()
    con = duckdb.connect(DB_PATH)
    con.execute("DROP TABLE IF EXISTS futures_main_close_au_adj")
    con.execute("CREATE TABLE futures_main_close_au_adj AS SELECT * FROM output")
    count = con.execute("SELECT COUNT(*) FROM futures_main_close_au_adj").fetchone()[0]
    print(f"\n✓ futures_main_close_au_adj: {count} 行")
    con.close()


if __name__ == "__main__":
    main()
