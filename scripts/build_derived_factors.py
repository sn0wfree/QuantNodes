#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DuckDB 因子层派生表生成脚本.

从 v_all_futures 计算:
  1. inter_contract_spread   — 跨期价差 (近月 vs 次月/季月)
  2. cross_product_spread    — 跨品种套利价差 (主流套利对)
  3. term_structure_daily    — 期限结构 (每日每产品全部合约)

Usage:
    python scripts/build_derived_factors.py                 # 全部
    python scripts/build_derived_factors.py --only inter    # 仅跨期
    python scripts/build_derived_factors.py --only cross    # 仅跨品种
    python scripts/build_derived_factors.py --only term     # 仅期限结构
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import duckdb

DUCKDB_PATH = Path("~/Public/DataCache/futures_options_daily.duckdb").expanduser()
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOG = logging.getLogger(__name__)

# ── 跨品种套利对定义 ──────────────────────────────────────────────────────────
# (pair_name, near_product, far_product, near_exchange, far_exchange)
CROSS_PAIRS = [
    # 黑色系
    ("RB-I",   "RB", "I",   "SHFE", "DCE"),
    ("HC-I",   "HC", "I",   "SHFE", "DCE"),
    ("I-J",    "I",  "J",   "DCE",  "DCE"),
    ("J-JM",   "J",  "JM",  "DCE",  "DCE"),
    ("RB-HC",  "RB", "HC",  "SHFE", "SHFE"),
    # 油粕
    ("Y-M",    "Y",  "M",   "DCE",  "DCE"),
    ("Y-P",    "Y",  "P",   "DCE",  "DCE"),
    ("OI-Y",   "OI", "Y",   "CZCE", "DCE"),
    ("OI-P",   "OI", "P",   "CZCE", "DCE"),
    ("RM-M",   "RM", "M",   "CZCE", "DCE"),
    ("A-M",    "A",  "M",   "DCE",  "DCE"),
    # 能化
    ("PP-L",   "PP", "L",   "DCE",  "DCE"),
    ("PP-V",   "PP", "V",   "DCE",  "DCE"),
    ("MA-FG",  "MA", "FG",  "CZCE", "CZCE"),
    ("MA-TA",  "MA", "TA",  "CZCE", "CZCE"),
    ("EG-TA",  "EG", "TA",  "DCE",  "CZCE"),
    ("BU-FU",  "BU", "FU",  "SHFE", "SHFE"),
    # 农产品
    ("CF-SR",  "CF", "SR",  "CZCE", "CZCE"),
    ("CS-C",   "CS", "C",   "DCE",  "DCE"),
    # 有色
    ("CU-AL",  "CU", "AL",  "SHFE", "SHFE"),
    ("ZN-PB",  "ZN", "PB",  "SHFE", "SHFE"),
    ("NI-SS",  "NI", "SS",  "SHFE", "SHFE"),
    # 贵金属
    ("AU-AG",  "AU", "AG",  "SHFE", "SHFE"),
    # 股指
    ("IF-IH",  "IF", "IH",  "CFFEX","CFFEX"),
    ("IF-IC",  "IF", "IC",  "CFFEX","CFFEX"),
    ("IH-IC",  "IH", "IC",  "CFFEX","CFFEX"),
]

# ── symbol → expiry 解析 ─────────────────────────────────────────────────────

def _parse_expiry(exchange: str, symbol: str):
    """返回 (expiry_year, expiry_month).

    非 CZCE: 后4位 YYMM, year=2000+YY (覆盖 2000-2099)
    CZCE:   后3位 YMM,  Y=0→2020, year=2020+Y (覆盖 2020-2029)
    """
    if exchange == "CZCE":
        y = int(symbol[-3])
        m = int(symbol[-2:])
        return 2020 + y, m
    yy = int(symbol[-4:-2])
    m = int(symbol[-2:])
    return 2000 + yy, m


def _expiry_date(exchange: str, symbol: str) -> str:
    """返回 YYYY-MM-DD 字符串（统一为每月15日作为近似到期日）."""
    y, m = _parse_expiry(exchange, symbol)
    return f"{y:04d}-{m:02d}-15"


# ════════════════════════════════════════════════════════════════════════════
# 1. 跨期价差 inter_contract_spread
# ════════════════════════════════════════════════════════════════════════════

BUILD_INTER_SQL = """
CREATE TABLE inter_contract_spread AS
WITH with_expiry AS (
    SELECT
        exchange,
        product,
        trade_date,
        symbol,
        close,
        open_interest,
        volume,
        -- 统一到期日期 (近似, 每月15日)
        {expiry_sql} AS expiry_date
    FROM v_all_futures
    WHERE close IS NOT NULL AND close > 0
),
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY product, trade_date
            ORDER BY expiry_date
        ) AS rank
    FROM with_expiry
)
SELECT
    exchange,
    product,
    trade_date,
    MAX(CASE WHEN rank=1 THEN symbol END)        AS sym_near,
    MAX(CASE WHEN rank=2 THEN symbol END)        AS sym_near2,
    MAX(CASE WHEN rank=3 THEN symbol END)        AS sym_near3,
    MAX(CASE WHEN rank=4 THEN symbol END)        AS sym_near4,
    MAX(CASE WHEN rank=1 THEN close END)         AS close_near,
    MAX(CASE WHEN rank=2 THEN close END)         AS close_near2,
    MAX(CASE WHEN rank=3 THEN close END)         AS close_near3,
    MAX(CASE WHEN rank=4 THEN close END)         AS close_near4,
    AVG(CASE WHEN rank=1 THEN close END) - AVG(CASE WHEN rank=2 THEN close END)    AS spread_1_2,
    AVG(CASE WHEN rank=1 THEN close END) - AVG(CASE WHEN rank=3 THEN close END)    AS spread_1_3,
    AVG(CASE WHEN rank=1 THEN close END) - AVG(CASE WHEN rank=4 THEN close END)    AS spread_1_4,
    MAX(CASE WHEN rank=1 THEN open_interest END) AS oi_near,
    MAX(CASE WHEN rank=2 THEN open_interest END) AS oi_near2,
    MAX(CASE WHEN rank=3 THEN open_interest END) AS oi_near3,
    MAX(CASE WHEN rank=4 THEN open_interest END) AS oi_near4
FROM ranked
WHERE rank <= 4
GROUP BY exchange, product, trade_date
ORDER BY product, trade_date
"""


def build_inter_contract_spread(con) -> int:
    LOG.info("Building inter_contract_spread ...")

    # Build expiry SQL expression
    expiry_case = """
        CASE
            WHEN exchange = 'CZCE' THEN
                MAKE_DATE(2020 + CAST(SUBSTR(symbol, -3, 1) AS INT),
                          CAST(SUBSTR(symbol, -2) AS INT), 15)
            ELSE
                MAKE_DATE(2000 + CAST(SUBSTR(symbol, -4, 2) AS INT),
                          CAST(SUBSTR(symbol, -2) AS INT), 15)
        END
    """
    sql = BUILD_INTER_SQL.replace("{expiry_sql}", expiry_case)

    con.execute("DROP TABLE IF EXISTS inter_contract_spread")
    con.execute(sql)

    cnt = con.execute("SELECT count() FROM inter_contract_spread").fetchone()[0]
    n_products = con.execute("SELECT count(DISTINCT product) FROM inter_contract_spread").fetchone()[0]
    LOG.info(f"  inter_contract_spread: {cnt:,} rows, {n_products} products")
    return cnt


# ════════════════════════════════════════════════════════════════════════════
# 2. 跨品种价差 cross_product_spread
# ════════════════════════════════════════════════════════════════════════════

def build_cross_product_spread(con) -> int:
    LOG.info("Building cross_product_spread ...")

    con.execute("DROP TABLE IF EXISTS cross_product_spread")

    con.execute("""
        CREATE TABLE cross_product_spread (
            pair_name       VARCHAR,
            near_product    VARCHAR,
            far_product     VARCHAR,
            trade_date      DATE,
            near_symbol     VARCHAR,
            far_symbol      VARCHAR,
            near_close      DOUBLE,
            far_close       DOUBLE,
            near_oi         DOUBLE,
            far_oi          DOUBLE,
            spread_raw      DOUBLE,
            spread_bp       DOUBLE
        )
    """)

    total = 0
    for pair_name, near_prod, far_prod, near_exch, far_exch in CROSS_PAIRS:
        sql = f"""
            INSERT INTO cross_product_spread
            SELECT
                '{pair_name}'             AS pair_name,
                '{near_prod}'             AS near_product,
                '{far_prod}'              AS far_product,
                COALESCE(n.trade_date, f.trade_date) AS trade_date,
                n.symbol                  AS near_symbol,
                f.symbol                  AS far_symbol,
                n.close                   AS near_close,
                f.close                   AS far_close,
                n.open_interest           AS near_oi,
                f.open_interest           AS far_oi,
                (n.close - f.close)       AS spread_raw,
                (n.close / NULLIF(f.close, 0) - 1.0) * 10000.0 AS spread_bp
            FROM main_contract_mapping n
            FULL JOIN main_contract_mapping f
                ON n.trade_date = f.trade_date
            WHERE (n.product = '{near_prod}' OR n.product IS NULL)
              AND (f.product = '{far_prod}' OR f.product IS NULL)
        """
        try:
            con.execute(sql)
            cnt = con.execute(
                "SELECT count() FROM cross_product_spread WHERE pair_name = ?",
                [pair_name]
            ).fetchone()[0]
            total += cnt
            LOG.info(f"  {pair_name}: {cnt:,} rows")
        except Exception as e:
            LOG.warning(f"  {pair_name}: FAILED — {e}")

    LOG.info(f"  cross_product_spread total: {total:,} rows, {len(CROSS_PAIRS)} pairs")
    return total


# ════════════════════════════════════════════════════════════════════════════
# 3. 期限结构 term_structure_daily
# ════════════════════════════════════════════════════════════════════════════

BUILD_TERM_SQL = """
CREATE TABLE term_structure_daily AS
WITH expiry AS (
    SELECT
        exchange,
        product,
        trade_date,
        symbol,
        close,
        open_interest,
        volume,
        {expiry_sql} AS expiry_date,
        {expiry_year_sql} AS expiry_year,
        {expiry_month_sql} AS expiry_month
    FROM v_all_futures
    WHERE close IS NOT NULL AND close > 0
),
ranked AS (
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY product, trade_date
            ORDER BY expiry_date
        ) AS rank,
        COUNT(*) OVER (PARTITION BY product, trade_date) AS total_contracts
    FROM expiry
)
SELECT
    exchange,
    product,
    trade_date,
    rank,
    symbol,
    expiry_year,
    expiry_month,
    close,
    open_interest,
    volume,
    total_contracts,
    DATEDIFF('day', trade_date, expiry_date) AS days_to_expiry
FROM ranked
WHERE rank <= 15
ORDER BY product, trade_date, rank
"""


def build_term_structure(con) -> int:
    LOG.info("Building term_structure_daily ...")

    expiry_date = """
        CASE
            WHEN exchange = 'CZCE' THEN
                MAKE_DATE(2020 + CAST(SUBSTR(symbol, -3, 1) AS INT),
                          CAST(SUBSTR(symbol, -2) AS INT), 15)
            ELSE
                MAKE_DATE(2000 + CAST(SUBSTR(symbol, -4, 2) AS INT),
                          CAST(SUBSTR(symbol, -2) AS INT), 15)
        END
    """
    expiry_year = """
        CASE
            WHEN exchange = 'CZCE' THEN 2020 + CAST(SUBSTR(symbol, -3, 1) AS INT)
            ELSE 2000 + CAST(SUBSTR(symbol, -4, 2) AS INT)
        END
    """
    expiry_month = """
        CAST(SUBSTR(symbol, CASE WHEN exchange = 'CZCE' THEN -2 ELSE -2 END) AS INT)
    """
    sql = BUILD_TERM_SQL.replace("{expiry_sql}", expiry_date)
    sql = sql.replace("{expiry_year_sql}", expiry_year)
    sql = sql.replace("{expiry_month_sql}", expiry_month)

    con.execute("DROP TABLE IF EXISTS term_structure_daily")
    con.execute(sql)

    cnt = con.execute("SELECT count() FROM term_structure_daily").fetchone()[0]
    n_products = con.execute("SELECT count(DISTINCT product) FROM term_structure_daily").fetchone()[0]
    LOG.info(f"  term_structure_daily: {cnt:,} rows, {n_products} products")
    return cnt


# ════════════════════════════════════════════════════════════════════════════
# 4. 历史波动率 volatility_hv_daily
# ════════════════════════════════════════════════════════════════════════════

def build_volatility_hv(con) -> int:
    LOG.info("Building volatility_hv_daily ...")

    con.execute("DROP TABLE IF EXISTS volatility_hv_daily")
    con.execute("""
        CREATE TABLE volatility_hv_daily AS
        WITH main_close AS (
            SELECT f.product, f.exchange, f.trade_date, f.close
            FROM v_all_futures f
            JOIN main_contract_mapping m
                ON f.product = m.product
               AND f.trade_date = m.trade_date
               AND f.symbol = m.symbol
            WHERE f.close IS NOT NULL AND f.close > 0
        ),
        with_returns AS (
            SELECT *,
                LN(close / LAG(close) OVER (
                    PARTITION BY product ORDER BY trade_date
                )) AS log_return
            FROM main_close
        )
        SELECT
            product,
            exchange,
            trade_date,
            close,
            ROUND(
                (STDDEV_SAMP(log_return) OVER (
                    PARTITION BY product ORDER BY trade_date
                    ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                ) * SQRT(252))::DOUBLE, 6
            ) AS hv_20,
            ROUND(
                (STDDEV_SAMP(log_return) OVER (
                    PARTITION BY product ORDER BY trade_date
                    ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
                ) * SQRT(252))::DOUBLE, 6
            ) AS hv_60,
            ROUND(
                (STDDEV_SAMP(log_return) OVER (
                    PARTITION BY product ORDER BY trade_date
                    ROWS BETWEEN 119 PRECEDING AND CURRENT ROW
                ) * SQRT(252))::DOUBLE, 6
            ) AS hv_120
        FROM with_returns
    """)

    cnt = con.execute("SELECT count() FROM volatility_hv_daily").fetchone()[0]
    n_products = con.execute("SELECT count(DISTINCT product) FROM volatility_hv_daily").fetchone()[0]
    LOG.info(f"  volatility_hv_daily: {cnt:,} rows, {n_products} products")
    return cnt


# ════════════════════════════════════════════════════════════════════════════
# 5. 期权 PCR 情绪 options_pcr_daily
# ════════════════════════════════════════════════════════════════════════════

def build_options_pcr(con) -> int:
    LOG.info("Building options_pcr_daily ...")

    con.execute("DROP TABLE IF EXISTS options_pcr_daily")
    con.execute("""
        CREATE TABLE options_pcr_daily AS
        SELECT
            exchange,
            product,
            trade_date,
            SUM(CASE WHEN option_type='C' THEN volume ELSE 0 END)::BIGINT
                AS call_volume,
            SUM(CASE WHEN option_type='P' THEN volume ELSE 0 END)::BIGINT
                AS put_volume,
            SUM(CASE WHEN option_type='C' THEN open_interest ELSE 0 END)::BIGINT
                AS call_oi,
            SUM(CASE WHEN option_type='P' THEN open_interest ELSE 0 END)::BIGINT
                AS put_oi,
            ROUND(
                CAST(SUM(CASE WHEN option_type='P' THEN volume ELSE 0 END) AS DOUBLE)
                / NULLIF(SUM(CASE WHEN option_type='C' THEN volume ELSE 0 END), 0),
                6
            ) AS pcr_volume,
            ROUND(
                CAST(SUM(CASE WHEN option_type='P' THEN open_interest ELSE 0 END) AS DOUBLE)
                / NULLIF(SUM(CASE WHEN option_type='C' THEN open_interest ELSE 0 END), 0),
                6
            ) AS pcr_oi,
            COUNT(*) AS total_contracts,
            COUNT(DISTINCT symbol) AS unique_contracts
        FROM v_all_options
        GROUP BY exchange, product, trade_date
    """)

    cnt = con.execute("SELECT count() FROM options_pcr_daily").fetchone()[0]
    n_products = con.execute("SELECT count(DISTINCT product) FROM options_pcr_daily").fetchone()[0]
    LOG.info(f"  options_pcr_daily: {cnt:,} rows, {n_products} products")
    return cnt


# ════════════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════════════

def run(only: str | None = None) -> None:
    if not DUCKDB_PATH.exists():
        LOG.error(f"DuckDB not found: {DUCKDB_PATH}")
        sys.exit(1)

    LOG.info(f"Connecting: {DUCKDB_PATH}")
    con = duckdb.connect(str(DUCKDB_PATH))

    try:
        t0 = time.time()

        if only is None or only == "inter":
            build_inter_contract_spread(con)

        if only is None or only == "cross":
            build_cross_product_spread(con)

        if only is None or only == "term":
            build_term_structure(con)

        if only is None or only == "hv":
            build_volatility_hv(con)

        if only is None or only == "pcr":
            build_options_pcr(con)

        elapsed = time.time() - t0
        LOG.info(f"Done in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    finally:
        con.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DuckDB 因子层派生表生成")
    p.add_argument(
        "--only", choices=["inter", "cross", "term", "hv", "pcr"],
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
