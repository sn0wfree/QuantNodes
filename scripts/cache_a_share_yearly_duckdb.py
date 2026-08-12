#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分年缓存全 A 股日线行情到离线 DuckDB (每年一个文件).

数据源: ClickHouse quote.stock_quote (data@localhost:8123).
输出:   ~/Public/DataCache/a_share_daily_{year}.duckdb, 每文件含 stock_quote 表
        (schema 镜像 ClickHouse 源表, 主键 ts_code+trade_date).

用法::

    python3 scripts/cache_a_share_yearly_duckdb.py                    # 全量 (所有年份)
    python3 scripts/cache_a_share_yearly_duckdb.py --start-year 2020  # 2020 起
    python3 scripts/cache_a_share_yearly_duckdb.py --years 2015 2016  # 指定年份
    python3 scripts/cache_a_share_yearly_duckdb.py --force            # 已存在也重建
    python3 scripts/cache_a_share_yearly_duckdb.py --out-dir /data/db

设计:
- 连接参数从 conn.ini [ClickHouse] 读取, 失败回退默认值 (与 cache_a_stock_data.py 一致)
- 逐年查询 + 逐年写文件, 避免单文件过大; 已存在的年份默认跳过 (--force 重建)
"""
from __future__ import annotations

import argparse
import configparser
import logging
import sys
from pathlib import Path

import clickhouse_connect
import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CONN_INI = ROOT / "conn.ini"
DEFAULT_OUT_DIR = Path("~/Public/DataCache").expanduser()

DEFAULT_CH = {
    "host": "localhost",
    "port": 8123,
    "user": "data",
    "passwd": "123456",
    "database": "quote",
}

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOG = logging.getLogger(__name__)


def _load_conn_ini() -> dict:
    cfg = dict(DEFAULT_CH)
    if not CONN_INI.exists():
        return cfg
    cp = configparser.ConfigParser()
    cp.read(CONN_INI, encoding="utf-8")
    if not cp.has_section("ClickHouse"):
        return cfg
    sec = cp["ClickHouse"]
    cfg["host"] = sec.get("host", cfg["host"])
    cfg["port"] = sec.getint("port", cfg["port"])
    cfg["user"] = sec.get("user", cfg["user"])
    cfg["passwd"] = sec.get("passwd", cfg["passwd"])
    cfg["database"] = sec.get("db", cfg["database"])
    return cfg


def list_years(client) -> list[int]:
    r = client.query(
        "SELECT DISTINCT toYear(trade_date) AS y FROM quote.stock_quote ORDER BY y"
    )
    return [row[0] for row in r.result_rows]


def fetch_year(client, year: int) -> pd.DataFrame:
    """拉取单年全部行情行 (DataFrame, DuckDB 原生替换扫描快速落盘)."""
    q = """
    SELECT ts_code,
           toDate(trade_date) AS trade_date,
           open, high, low, close,
           pre_close, change, pct_chg, vol, amount
    FROM quote.stock_quote
    WHERE trade_date >= '{y}-01-01' AND trade_date < '{y1}-01-01'
    ORDER BY ts_code, trade_date
    """.format(y=year, y1=year + 1)
    return client.query_df(q)


def write_year_db(out_dir: Path, year: int, df: pd.DataFrame) -> Path:
    path = out_dir / f"a_share_daily_{year}.duckdb"
    con = duckdb.connect(str(path))
    con.execute("DROP TABLE IF EXISTS stock_quote")
    con.execute(
        """
        CREATE TABLE stock_quote (
            ts_code    VARCHAR,
            trade_date DATE,
            open       DOUBLE,
            high       DOUBLE,
            low        DOUBLE,
            close      DOUBLE,
            pre_close  DOUBLE,
            change     DOUBLE,
            pct_chg    DOUBLE,
            vol        DOUBLE,
            amount     DOUBLE,
            PRIMARY KEY (ts_code, trade_date)
        )
        """
    )
    con.execute("INSERT INTO stock_quote SELECT * FROM df")
    con.close()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--years", type=int, nargs="+", help="显式指定年份列表")
    parser.add_argument("--force", action="store_true", help="文件已存在也重建")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _load_conn_ini()
    client = clickhouse_connect.get_client(
        host=cfg["host"], port=cfg["port"],
        username=cfg["user"], password=cfg["passwd"], database=cfg["database"],
    )

    years = args.years or [
        y for y in list_years(client) if y >= args.start_year
    ]
    LOG.info("计划年份: %s (共 %d 年)", years, len(years))

    for year in years:
        path = args.out_dir / f"a_share_daily_{year}.duckdb"
        if path.exists() and not args.force:
            LOG.info("跳过 %d: 已存在 %s", year, path)
            continue
        LOG.info("拉取 %d ...", year)
        rows = fetch_year(client, year)
        LOG.info("  %d 行 → %s", len(rows), path.name)
        write_year_db(args.out_dir, year, rows)
        LOG.info("完成 %d", year)

    LOG.info("全部完成: %s", args.out_dir)


if __name__ == "__main__":
    main()
