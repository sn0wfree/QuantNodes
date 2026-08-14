#!/usr/bin/env python3
"""构建 VRP 面板: IV − HV + rolling 分位 → DuckDB vrp_daily 表.

docs/100 §六.2 规格:
    vrp_daily (product, trade_date, atm_iv, hv_20, vrp, vrp_percentile,
               vrp_positive, product_class)

数据来源: option_iv_daily (已有) JOIN volatility_hv_daily (已有)
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DB_PATH = str(Path.home() / "Public" / "DataCache" / "futures_options_daily.duckdb")
VRP_TABLE = "vrp_daily"
ROLLING_WINDOW = 252
ROLLING_MIN_PERIODS = 60

VRP_TABLE_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {VRP_TABLE} (
    product VARCHAR,
    trade_date DATE,
    atm_iv DOUBLE,
    hv_20 DOUBLE,
    hv_60 DOUBLE,
    hv_120 DOUBLE,
    vrp DOUBLE,
    vrp_percentile DOUBLE,
    iv_percentile DOUBLE,
    hv_percentile DOUBLE,
    vrp_positive INTEGER,
    product_class VARCHAR,   -- 'vrp_pos' | 'neutral' | 'vrp_neg'
    PRIMARY KEY (product, trade_date)
)
"""


def build_vrp_panel(duckdb_path: str = DB_PATH) -> pd.DataFrame:
    """join option_iv_daily + volatility_hv_daily → vrp panel."""
    con = duckdb.connect(duckdb_path)
    try:
        df = con.execute("""
            SELECT iv.product, iv.trade_date,
                   iv.atm_iv, iv.iv_percentile,
                   hv.hv_20, hv.hv_60, hv.hv_120
            FROM option_iv_daily iv
            JOIN volatility_hv_daily hv
              ON iv.product = hv.product AND iv.trade_date = hv.trade_date
            ORDER BY iv.product, iv.trade_date
        """).df()
    finally:
        con.close()

    if df.empty:
        return df

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["vrp"] = df["atm_iv"] - df["hv_20"]

    # rolling 分位 (per product)
    groups = df.groupby("product")
    df["vrp_percentile"] = (
        groups["vrp"]
        .transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS).rank(pct=True))
    )
    df["hv_percentile"] = (
        groups["hv_20"]
        .transform(lambda s: s.rolling(ROLLING_WINDOW, min_periods=ROLLING_MIN_PERIODS).rank(pct=True))
    )
    df["vrp_positive"] = (df["vrp"] > 0).astype(int)

    # 品种分类 (基于 mean VRP, per product)
    product_means = df.groupby("product")["vrp"].mean()
    product_pos_ratio = df.groupby("product")["vrp_positive"].mean()
    class_map = {}
    for prod in product_means.index:
        m = product_means[prod]
        p = product_pos_ratio[prod]
        if m > 0.01 and p > 0.55:
            class_map[prod] = "vrp_pos"
        elif m < -0.01 or p < 0.45:
            class_map[prod] = "vrp_neg"
        else:
            class_map[prod] = "neutral"
    df["product_class"] = df["product"].map(class_map)

    return df


def save_vrp_panel(df: pd.DataFrame, duckdb_path: str = DB_PATH) -> None:
    """写入 DuckDB."""
    con = duckdb.connect(duckdb_path)
    try:
        con.execute(VRP_TABLE_SCHEMA)
        con.execute(f"DELETE FROM {VRP_TABLE}")
        con.execute(f"""
            INSERT INTO {VRP_TABLE}
            (product, trade_date, atm_iv, hv_20, hv_60, hv_120,
             vrp, vrp_percentile, iv_percentile, hv_percentile,
             vrp_positive, product_class)
            SELECT product, trade_date, atm_iv, hv_20, hv_60, hv_120,
                   vrp, vrp_percentile, iv_percentile, hv_percentile,
                   vrp_positive, product_class
            FROM df
        """)
        n = con.execute(f"SELECT COUNT(*) FROM {VRP_TABLE}").fetchone()[0]
        print(f"{VRP_TABLE}: {n:,} rows saved")
    finally:
        con.close()


def print_stats(df: pd.DataFrame) -> None:
    """打印品种统计."""
    stats = (
        df.groupby("product")
        .agg(
            n=("vrp", "size"),
            avg_vrp=("vrp", "mean"),
            avg_iv=("atm_iv", "mean"),
            avg_hv=("hv_20", "mean"),
            pct_positive=("vrp_positive", "mean"),
        )
        .round(4)
    )
    stats = stats.sort_values("avg_vrp", ascending=False)
    print(f"\n{'prod':5s} {'n':>6s} {'avg_iv':>7s} {'avg_hv':>7s} "
          f"{'avg_vrp':>7s} {'pct_pos':>7s} {'class':>8s}")
    for prod, row in stats.iterrows():
        cls = df[df["product"] == prod]["product_class"].iloc[0]
        print(f"{prod:5s} {int(row['n']):6d} {row['avg_iv']:7.4f} "
              f"{row['avg_hv']:7.4f} {row['avg_vrp']:+7.4f} "
              f"{row['pct_positive']*100:6.1f}% {cls:>8s}")


def main():
    print("Building VRP panel ...")
    df = build_vrp_panel()
    print(f"Panel: {len(df):,} rows, {df['product'].nunique()} products")
    print_stats(df)
    save_vrp_panel(df)


if __name__ == "__main__":
    main()