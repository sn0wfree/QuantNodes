#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DuckDB 派生表生成脚本.

从 v_all_futures 计算:
  1. trading_calendar        — 交易日历
  2. main_contract_mapping   — 主力合约映射 (每日每产品 OI 最大)
  3. continuous_main_{PRODUCT}_daily — 主力连续面板 (92 张表)

Usage:
    python scripts/build_derived_tables.py              # 全部
    python scripts/build_derived_tables.py --only calendar   # 仅日历
    python scripts/build_derived_tables.py --only mapping    # 仅主力映射
    python scripts/build_derived_tables.py --only continuous # 仅连续面板
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import duckdb
from tqdm import tqdm

# ════════════════════════════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════════════════════════════

DUCKDB_PATH = Path("~/Public/DataCache/futures_options_daily.duckdb").expanduser()

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOG = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# 1. Trading Calendar
# ════════════════════════════════════════════════════════════════════════════

def build_trading_calendar(con) -> int:
    """生成 trading_calendar 表."""
    LOG.info("Building trading_calendar...")

    con.execute("DROP TABLE IF EXISTS trading_calendar")
    con.execute("""
        CREATE TABLE trading_calendar AS
        SELECT
            trade_date,
            EXTRACT(YEAR FROM trade_date)::INT AS year,
            EXTRACT(MONTH FROM trade_date)::INT AS month,
            EXTRACT(DOW FROM trade_date)::INT AS day_of_week,
            (trade_date = date_trunc('month', trade_date)) AS is_first_day_of_month,
            (trade_date = (date_trunc('month', trade_date) + INTERVAL '1 MONTH' - INTERVAL '1 DAY')) AS is_last_day_of_month
        FROM (
            SELECT DISTINCT trade_date
            FROM v_all_futures
            WHERE trade_date IS NOT NULL
            ORDER BY trade_date
        )
    """)

    cnt = con.execute("SELECT count() FROM trading_calendar").fetchone()[0]
    LOG.info(f"  trading_calendar: {cnt:,} rows")
    return cnt

# ════════════════════════════════════════════════════════════════════════════
# 2. Main Contract Mapping
# ════════════════════════════════════════════════════════════════════════════

def build_main_contract_mapping(con) -> int:
    """生成 main_contract_mapping 表 (每日每产品 OI 最大合约)."""
    LOG.info("Building main_contract_mapping...")

    con.execute("DROP TABLE IF EXISTS main_contract_mapping")
    con.execute("""
        CREATE TABLE main_contract_mapping AS
        WITH ranked AS (
            SELECT
                product,
                trade_date,
                symbol,
                open_interest,
                volume,
                close,
                ROW_NUMBER() OVER (
                    PARTITION BY product, trade_date
                    ORDER BY open_interest DESC NULLS LAST
                ) AS rn
            FROM v_all_futures
            WHERE open_interest IS NOT NULL
              AND open_interest > 0
        )
        SELECT product, trade_date, symbol, open_interest, volume, close
        FROM ranked
        WHERE rn = 1
        ORDER BY product, trade_date
    """)

    cnt = con.execute("SELECT count() FROM main_contract_mapping").fetchone()[0]
    n_products = con.execute("SELECT count(DISTINCT product) FROM main_contract_mapping").fetchone()[0]
    LOG.info(f"  main_contract_mapping: {cnt:,} rows, {n_products} products")
    return cnt

# ════════════════════════════════════════════════════════════════════════════
# 3. Continuous Main Panel
# ════════════════════════════════════════════════════════════════════════════

def get_products(con) -> list[str]:
    """获取 v_all_futures 中所有产品."""
    rows = con.execute(
        "SELECT DISTINCT product FROM v_all_futures ORDER BY product"
    ).fetchall()
    return [r[0] for r in rows]


def build_continuous_main(con, product: str) -> int:
    """生成单产品的主力连续面板 (原始拼接, 无换月平滑)."""
    table_name = f"continuous_main_{product.lower()}_daily"

    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    con.execute(f"""
        CREATE TABLE {table_name} AS
        SELECT
            f.trade_date,
            f.open,
            f.high,
            f.low,
            f.close,
            f.volume,
            f.open_interest
        FROM v_all_futures f
        JOIN main_contract_mapping m
            ON f.product = m.product
           AND f.trade_date = m.trade_date
           AND f.symbol = m.symbol
        WHERE f.product = '{product}'
        ORDER BY f.trade_date
    """)

    return con.execute(f"SELECT count() FROM {table_name}").fetchone()[0]


def build_smoothed_continuous(con, product: str) -> int:
    """生成单产品的平滑连续面板 (比例平滑, 消除换月跳价).

    算法: ratio adjustment (back-adjustment)
    - 换月日: ratio = new_open / prev_close
    - 历史价格: 乘以累积调整因子
    - 保持收益率连续, 消除价格跳变
    """
    table_name = f"smoothed_{product.lower()}_daily"

    # 获取原始连续合约和换月映射
    raw_table = f"continuous_main_{product.lower()}_daily"

    # 检查原始表是否存在
    exists = con.execute(
        f"SELECT count(*) FROM information_schema.tables "
        f"WHERE table_name = '{raw_table}'"
    ).fetchone()[0]
    if not exists:
        LOG.warning(f"  {product}: {raw_table} not found, skipping smoothed")
        return 0

    con.execute(f"DROP TABLE IF EXISTS {table_name}")

    # 使用窗口函数计算换月调整因子
    con.execute(f"""
        CREATE TABLE {table_name} AS
        WITH raw AS (
            SELECT trade_date, open, high, low, close, volume, open_interest,
                   LAG(close) OVER (ORDER BY trade_date) as prev_close,
                   LAG(symbol) OVER (ORDER BY trade_date) as prev_symbol
            FROM (
                SELECT f.trade_date, f.open, f.high, f.low, f.close,
                       f.volume, f.open_interest, m.symbol
                FROM v_all_futures f
                JOIN main_contract_mapping m
                    ON f.product = m.product
                   AND f.trade_date = m.trade_date
                   AND f.symbol = m.symbol
                WHERE f.product = '{product}'
            ) t
        ),
        -- 检测换月日: symbol变化
        rolls AS (
            SELECT trade_date,
                   CASE WHEN prev_symbol IS NOT NULL AND symbol != prev_symbol
                        THEN 1 ELSE 0 END as is_roll
            FROM (
                SELECT trade_date, symbol,
                       LAG(symbol) OVER (ORDER BY trade_date) as prev_symbol
                FROM (
                    SELECT f.trade_date, m.symbol
                    FROM v_all_futures f
                    JOIN main_contract_mapping m
                        ON f.product = m.product
                       AND f.trade_date = m.trade_date
                       AND f.symbol = m.symbol
                    WHERE f.product = '{product}'
                ) t
            ) t2
        ),
        -- 计算调整因子: 换月日的调整比率 (back-adjustment)
        -- ratio = new_open / prev_close
        factors AS (
            SELECT r.trade_date,
                   CASE WHEN r.is_roll = 1 AND raw.prev_close > 0
                        THEN raw.open / raw.prev_close
                        ELSE 1.0 END as roll_factor
            FROM rolls r
            JOIN raw ON r.trade_date = raw.trade_date
        ),
        -- 累积调整因子 (从最早日期开始累积)
        cum_factors AS (
            SELECT trade_date,
                   EXP(SUM(LN(roll_factor)) OVER (ORDER BY trade_date)) as cum_factor
            FROM factors
        )
        SELECT raw.trade_date,
               raw.open * cf.cum_factor as open,
               raw.high * cf.cum_factor as high,
               raw.low * cf.cum_factor as low,
               raw.close * cf.cum_factor as close,
               raw.volume,
               raw.open_interest
        FROM raw
        JOIN cum_factors cf ON raw.trade_date = cf.trade_date
        ORDER BY raw.trade_date
    """)

    return con.execute(f"SELECT count() FROM {table_name}").fetchone()[0]


def build_all_continuous_main(con) -> int:
    """生成全部主力连续面板."""
    products = get_products(con)
    LOG.info(f"Building continuous_main for {len(products)} products...")

    total = 0
    pbar = tqdm(products, desc="Continuous", unit="product", ncols=100)
    for product in pbar:
        pbar.set_postfix_str(product, refresh=False)
        try:
            cnt = build_continuous_main(con, product)
            total += cnt
            pbar.set_postfix_str(f"{product} ✓ {cnt:,}", refresh=False)
        except Exception as e:
            LOG.error(f"Failed {product}: {e}")
            pbar.set_postfix_str(f"{product} ✗", refresh=False)

    pbar.close()
    LOG.info(f"  continuous_main: {total:,} total rows across {len(products)} products")
    return total


def build_all_smoothed_continuous(con) -> int:
    """生成全部平滑连续面板 (比例平滑, 消除换月跳价)."""
    products = get_products(con)
    LOG.info(f"Building smoothed continuous for {len(products)} products...")

    total = 0
    pbar = tqdm(products, desc="Smoothed", unit="product", ncols=100)
    for product in pbar:
        pbar.set_postfix_str(product, refresh=False)
        try:
            cnt = build_smoothed_continuous(con, product)
            total += cnt
            pbar.set_postfix_str(f"{product} ✓ {cnt:,}", refresh=False)
        except Exception as e:
            LOG.error(f"Failed {product}: {e}")
            pbar.set_postfix_str(f"{product} ✗", refresh=False)

    pbar.close()
    LOG.info(f"  smoothed continuous: {total:,} total rows across {len(products)} products")
    return total

# ════════════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════════════

def run(only: str | None = None) -> None:
    """执行派生表生成."""
    if not DUCKDB_PATH.exists():
        LOG.error(f"DuckDB not found: {DUCKDB_PATH}")
        sys.exit(1)

    LOG.info(f"Connecting to DuckDB: {DUCKDB_PATH}")
    con = duckdb.connect(str(DUCKDB_PATH))

    try:
        t0 = time.time()

        if only is None or only == "calendar":
            build_trading_calendar(con)

        if only is None or only == "mapping":
            build_main_contract_mapping(con)

        if only is None or only == "continuous":
            build_all_continuous_main(con)

        if only is None or only == "smoothed":
            build_all_smoothed_continuous(con)

        elapsed = time.time() - t0
        LOG.info(f"Done in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    finally:
        con.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DuckDB 派生表生成",
    )
    p.add_argument(
        "--only", choices=["calendar", "mapping", "continuous", "smoothed"],
        help="仅生成指定表 (默认全部)",
    )
    p.add_argument(
        "--duckdb", type=str, default=None,
        help="DuckDB 路径",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    global DUCKDB_PATH
    if args.duckdb:
        DUCKDB_PATH = Path(args.duckdb)

    run(only=args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
