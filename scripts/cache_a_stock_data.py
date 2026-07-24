#!/usr/bin/env python3.11
# coding=utf-8
"""
cache_a_stock_data.py - 从 ClickHouse 缓存全 A 股日线到本地 parquet.

从 ClickHouse `quote.stock_quote` 拉取日线数据 (默认 2025-01-01 ~ 2026-07-23),
委托 ClickHouseDataLoader (database_node 兼容路径) 完成查询 + 清洗 + 落盘.

用法::

    python3.11 scripts/cache_a_stock_data.py
    python3.11 scripts/cache_a_stock_data.py --start 2025-01-01 --end 2026-07-23
    python3.11 scripts/cache_a_stock_data.py --no-cache
    python3.11 scripts/cache_a_stock_data.py --cache-parquet data/cache/full_a_2025_2026.parquet

设计:
- conn 参数优先从 conn.ini 的 [ClickHouse] section 读, 失败回退默认值
- 委托 ClickHouseDataLoader (QuantNodes.research.quant_alpha.evaluation) 而非重写,
  保证 schema / 清洗 / 缓存逻辑与 Table 4 pipeline 一致
- 缓存命中即跳过 (除非 --no-cache)
"""
from __future__ import annotations

import argparse
import configparser
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONN_INI = ROOT / "conn.ini"

DEFAULT_CH = {
    "host": "localhost",
    "port": 8123,
    "user": "data",
    "passwd": "123456",
    "database": "quote",
}


def _load_conn_ini() -> dict:
    """从 conn.ini 的 [ClickHouse] section 读连接参数, 失败回退默认值."""
    if not CONN_INI.exists():
        return dict(DEFAULT_CH)
    cp = configparser.ConfigParser()
    cp.read(CONN_INI, encoding="utf-8")
    if not cp.has_section("ClickHouse"):
        return dict(DEFAULT_CH)
    sec = cp["ClickHouse"]
    cfg = dict(DEFAULT_CH)
    cfg["host"] = sec.get("host", cfg["host"])
    cfg["port"] = sec.getint("port", cfg["port"])
    cfg["user"] = sec.get("user", cfg["user"])
    cfg["passwd"] = sec.get("passwd", cfg["passwd"])
    cfg["database"] = sec.get("db", cfg["database"])
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 ClickHouse 缓存全 A 股日线到本地 parquet",
    )
    parser.add_argument("--table", type=str, default="quote.stock_quote",
                        help="ClickHouse 表名 (default: quote.stock_quote)")
    parser.add_argument("--start", type=str, default="2025-01-01",
                        help="起始日期 (default: 2025-01-01)")
    parser.add_argument("--end", type=str, default="2026-07-23",
                        help="结束日期 (default: 2026-07-23)")
    parser.add_argument("--cache-parquet", type=str,
                        default="data/cache/full_a_2025_2026.parquet",
                        help="本地 parquet 缓存路径")
    parser.add_argument("--no-cache", action="store_true",
                        help="不使用本地 parquet 缓存 (强制重拉)")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    conn = _load_conn_ini()
    cache_path = ROOT / args.cache_parquet

    print(f"{'='*60}")
    print("A 股日线缓存 (ClickHouse → parquet)")
    print(f"{'='*60}")
    print(f"  连接: {conn['host']}:{conn['port']} (user={conn['user']}, db={conn['database']})")
    print(f"  表:   {args.table}")
    print(f"  区间: {args.start} ~ {args.end}")
    print(f"  路径: {cache_path}")
    if cache_path.exists():
        if args.no_cache:
            print("  模式: 强制重拉 (--no-cache, 已存在将被覆盖)")
        else:
            print("  模式: 命中缓存 (已存在, 跳过查询; 加 --no-cache 强制重拉)")
    print(f"{'='*60}\n")

    from QuantNodes.research.quant_alpha.evaluation import ClickHouseDataLoader

    loader = ClickHouseDataLoader(
        table=args.table,
        host=conn["host"],
        port=conn["port"],
        user=conn["user"],
        password=conn["passwd"],
        database=conn["database"],
        start_date=args.start,
        end_date=args.end,
        cache_parquet=None if args.no_cache else str(cache_path),
    )

    df = loader.load()

    print(f"\n{'='*60}")
    print("完成")
    print(f"  rows  : {df.height:,}")
    print(f"  codes : {df['code'].n_unique():,}")
    print(f"  date  : [{df['date'].min()} .. {df['date'].max()}]")
    print(f"  cols  : {df.columns}")
    print(f"  file  : {cache_path}")
    if cache_path.exists():
        size_mb = cache_path.stat().st_size / 1024 / 1024
        print(f"  size  : {size_mb:.2f} MB")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
