#!/usr/bin/env python3.11
"""从 gold 数据库提取宏观因子数据到 QuantNodes 缓存.

数据源:
  - ~/Public/gold/database/gold_analysis.db
    - usd_index: 美元指数 (DXY)
    - fred_indicators: VIX, 实际利率 (rir)

输出:
  - data/high_freq_macro/macro_dxy_daily.parquet
  - data/high_freq_macro/macro_vix_daily.parquet
  - data/high_freq_macro/macro_real_rate_monthly.parquet
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
HF_DIR = REPO / "data" / "high_freq_macro"
HF_DIR.mkdir(parents=True, exist_ok=True)

GOLD_DB = Path.home() / "Public" / "gold" / "database" / "gold_analysis.db"


def extract_dxy() -> pd.DataFrame:
    """提取美元指数 (DXY) 日度数据."""
    cache = HF_DIR / "macro_dxy_daily.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    print(f"从 {GOLD_DB} 提取 DXY 数据...")
    conn = sqlite3.connect(str(GOLD_DB))
    df = pd.read_sql("SELECT date, value FROM usd_index ORDER BY date", conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.columns = ["dxy"]

    # 保存缓存
    df.to_parquet(cache)
    print(f"  DXY: {len(df)} 条, {df.index[0]} ~ {df.index[-1]}")
    return df


def extract_vix() -> pd.DataFrame:
    """提取 VIX 恐慌指数日度数据."""
    cache = HF_DIR / "macro_vix_daily.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    print(f"从 {GOLD_DB} 提取 VIX 数据...")
    conn = sqlite3.connect(str(GOLD_DB))
    df = pd.read_sql(
        "SELECT date, value FROM fred_indicators WHERE indicator_name = 'vix' ORDER BY date",
        conn,
    )
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.columns = ["vix"]

    # 保存缓存
    df.to_parquet(cache)
    print(f"  VIX: {len(df)} 条, {df.index[0]} ~ {df.index[-1]}")
    return df


def extract_real_rate() -> pd.DataFrame:
    """提取实际利率 (10年期) 月度数据."""
    cache = HF_DIR / "macro_real_rate_monthly.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    print(f"从 {GOLD_DB} 提取实际利率数据...")
    conn = sqlite3.connect(str(GOLD_DB))
    df = pd.read_sql(
        "SELECT date, value FROM fred_indicators WHERE indicator_name = 'rir' ORDER BY date",
        conn,
    )
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df.columns = ["real_rate"]

    # 保存缓存
    df.to_parquet(cache)
    print(f"  实际利率: {len(df)} 条, {df.index[0]} ~ {df.index[-1]}")
    return df


def extract_all_macro():
    """提取全部宏观因子数据."""
    print("=" * 60)
    print("从 gold 数据库提取宏观因子数据")
    print("=" * 60)

    dxy = extract_dxy()
    vix = extract_vix()
    real_rate = extract_real_rate()

    print("\n提取完成!")
    print(f"  DXY: {len(dxy)} 条")
    print(f"  VIX: {len(vix)} 条")
    print(f"  实际利率: {len(real_rate)} 条")

    return dxy, vix, real_rate


if __name__ == "__main__":
    extract_all_macro()
