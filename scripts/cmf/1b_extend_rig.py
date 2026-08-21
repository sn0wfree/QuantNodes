#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/1b_extend_rig.py — 补齐钻机数 (EIA 周度停更后人工延展).

akshare macro_usa_rig_count 只更新到 2024-02-23.
策略: 取最后观察值 (503) 作为最后 18 个月的占位, 在 DB 中添加占位行.
标记 source='rig_extended' 以便区分.
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import duckdb
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DB = REPO / "data" / "cmf" / "fundamental.duckdb"


def main():
    con = duckdb.connect(str(DB))

    last_date = con.execute("SELECT MAX(trade_date) FROM fundamentals_daily WHERE value_col='us_oil_rig'").fetchone()[0]
    last_val = con.execute("SELECT value FROM fundamentals_daily WHERE value_col='us_oil_rig' ORDER BY trade_date DESC LIMIT 1").fetchone()[0]
    print(f"原始最后值: {last_val} @ {last_date}")

    today = pd.Timestamp("2026-08-17").date()
    if last_date >= today:
        print("无需延展")
        return

    weeks = pd.date_range(start=pd.Timestamp(last_date) + pd.Timedelta(weeks=1),
                          end=today, freq="W-FRI")
    df = pd.DataFrame({
        "source": ["rig_extended"] * len(weeks),
        "trade_date": [d.date() for d in weeks],
        "value_col": ["us_oil_rig"] * len(weeks),
        "value": [float(last_val)] * len(weeks),
    })

    con.execute("DELETE FROM fundamentals_daily WHERE source = 'rig_extended'")
    con.append("fundamentals_daily", df, by_name=True)

    n = con.execute("SELECT COUNT(*) FROM fundamentals_daily WHERE value_col='us_oil_rig'").fetchone()[0]
    print(f"延展后 us_oil_rig 总行数: {n}")

    con.close()


if __name__ == "__main__":
    main()