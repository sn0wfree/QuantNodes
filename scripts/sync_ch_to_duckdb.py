#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ClickHouse tick_quote → DuckDB 全量同步脚本.

直接查 tick_quote.{PRODUCT} 聚合日线 (绕过 FINAL, 极快),
写入 ~/Public/DataCache/futures_options_daily.duckdb.

支持:
  - 进度条 (tqdm)
  - 断点续传 (JSON checkpoint)
  - 增量/全量模式

Usage:
    python scripts/sync_ch_to_duckdb.py                  # 全量同步
    python scripts/sync_ch_to_duckdb.py --resume          # 断点续传
    python scripts/sync_ch_to_duckdb.py --products IF AU  # 指定产品
    python scripts/sync_ch_to_duckdb.py --dry-run         # 仅打印计划
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import clickhouse_connect
import duckdb
import pandas as pd
from tqdm import tqdm

# ════════════════════════════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DUCKDB_PATH = Path("~/Public/DataCache/futures_options_daily.duckdb").expanduser()
CHECKPOINT_PATH = DUCKDB_PATH.parent / "_sync_checkpoint.json"

CH_CONFIG = {
    "host": "localhost",
    "port": 8123,
    "username": "data",
    "password": "123456",
}

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOG = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# CH → DuckDB 列映射
# ════════════════════════════════════════════════════════════════════════════

CH_TO_DUCKDB_COLS = {
    "Exchange": "exchange",
    "Product": "product",
    "InstrumentID": "symbol",
    "TradingDay": "trade_date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Turnover": "turnover",
    "OpenInterest": "open_interest",
    "PreSettlement": "pre_settlement",
    "PreClose": "pre_close",
    "PreOpenInterest": "pre_open_interest",
}

# ════════════════════════════════════════════════════════════════════════════
# 断点续传
# ════════════════════════════════════════════════════════════════════════════

def load_checkpoint() -> dict:
    """加载断点文件."""
    if CHECKPOINT_PATH.exists():
        try:
            return json.loads(CHECKPOINT_PATH.read_text())
        except Exception:
            return {"completed": [], "failed": [], "stats": {}}
    return {"completed": [], "failed": [], "stats": {}}


def save_checkpoint(ckpt: dict) -> None:
    """保存断点文件."""
    CHECKPOINT_PATH.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2))


def mark_completed(ckpt: dict, product: str, rows: int, elapsed: float) -> None:
    """标记产品完成."""
    ckpt["completed"].append(product)
    ckpt["stats"][product] = {"rows": rows, "elapsed": round(elapsed, 2)}
    save_checkpoint(ckpt)


def mark_failed(ckpt: dict, product: str, error: str) -> None:
    """标记产品失败."""
    ckpt["failed"].append(product)
    ckpt["stats"][product] = {"error": error[:200]}
    save_checkpoint(ckpt)

# ════════════════════════════════════════════════════════════════════════════
# 核心同步逻辑
# ════════════════════════════════════════════════════════════════════════════

def get_product_exchange_map(duckdb_path: Path) -> dict[str, str]:
    """从 DuckDB contract_specs 读取 product→exchange 映射."""
    con = duckdb.connect(str(duckdb_path), read_only=True)
    try:
        rows = con.execute(
            "SELECT product, exchange FROM contract_specs ORDER BY product"
        ).fetchall()
        return {r[0]: r[1] for r in rows}
    finally:
        con.close()


def build_sync_sql(product: str, year: int | None = None) -> str:
    """构建 CH tick_quote 直接聚合 SQL (无 FINAL).
    
    Args:
        product: 产品代码
        year: 可选, 指定年份 (用于按年查询大表)
    """
    date_filter = ""
    if year is not None:
        date_filter = f"AND TradingDay >= '{year}-01-01' AND TradingDay < '{year+1}-01-01'"
    
    return f"""
SELECT
    '' AS Exchange,
    '' AS Product,
    InstrumentID AS Symbol,
    TradingDay AS TradeDate,
    argMin(LastPrice, TradingDateTimeMS) AS Open,
    max(LastPrice) AS High,
    min(LastPrice) AS Low,
    argMax(LastPrice, TradingDateTimeMS) AS Close,
    max(Volume) - min(Volume) AS Volume,
    max(Turnover) - min(Turnover) AS Turnover,
    argMax(OpenInterest, TradingDateTimeMS) AS OpenInterest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS PreSettlement,
    argMin(PreClosePrice, TradingDateTimeMS) AS PreClose,
    argMin(PreOpenInterest, TradingDateTimeMS) AS PreOpenInterest
FROM tick_quote.{product}
WHERE SessionType IN (2, 3)
  AND Exchange NOT LIKE '%\\_night%'
  {date_filter}
GROUP BY InstrumentID, TradingDay
ORDER BY TradingDay, InstrumentID
SETTINGS max_execution_time=120
"""


def sync_one_product(
    ch_client,
    duckdb_path: Path,
    product: str,
    exchange: str,
) -> int:
    """同步单个产品: CH → DuckDB. 返回写入行数."""
    sql = build_sync_sql(product)

    # CH 查询
    result = ch_client.query(sql).result_rows
    if not result:
        return 0

    # 转 DataFrame
    cols = list(CH_TO_DUCKDB_COLS.values())
    df = pd.DataFrame(result, columns=cols)

    # 填充 exchange/product (CH 返回空字符串)
    df["exchange"] = exchange
    df["product"] = product

    # 过滤期权 symbol (含 C/P + 数字的合约名)
    mask_opt = df["symbol"].str.contains(r"[CP]\d", case=False, na=False)
    df = df[~mask_opt].copy()

    if df.empty:
        return 0

    # 类型转换
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    for col in ["open", "high", "low", "close", "turnover", "open_interest",
                "pre_settlement", "pre_close", "pre_open_interest"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")

    # 写入 DuckDB
    table_name = f"{exchange.lower()}_{product.lower()}_futures_daily"
    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    finally:
        con.close()

    return len(df)


def sync_one_product_yearly(
    ch_client,
    duckdb_path: Path,
    product: str,
    exchange: str,
    start_year: int = 2018,
    end_year: int = 2027,
) -> int:
    """按年查询同步大表: CH → DuckDB. 返回写入行数.
    
    用于 AG/AL 等 tick 表 > 10 亿行的产品, 避免单次查询超时.
    """
    cols = list(CH_TO_DUCKDB_COLS.values())
    all_dfs = []
    
    for year in range(start_year, end_year):
        sql = build_sync_sql(product, year=year)
        result = ch_client.query(sql).result_rows
        if result:
            df_year = pd.DataFrame(result, columns=cols)
            all_dfs.append(df_year)
    
    if not all_dfs:
        return 0
    
    # 合并所有年份
    df = pd.concat(all_dfs, ignore_index=True)
    
    # 填充 exchange/product
    df["exchange"] = exchange
    df["product"] = product
    
    # 过滤期权 symbol
    mask_opt = df["symbol"].str.contains(r"[CP]\d", case=False, na=False)
    df = df[~mask_opt].copy()
    
    if df.empty:
        return 0
    
    # 类型转换
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    for col in ["open", "high", "low", "close", "turnover", "open_interest",
                "pre_settlement", "pre_close", "pre_open_interest"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    
    # 去重 (按年查询可能有重叠)
    df = df.drop_duplicates(subset=["symbol", "trade_date"], keep="last")
    df = df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    
    # 写入 DuckDB
    table_name = f"{exchange.lower()}_{product.lower()}_futures_daily"
    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")
    finally:
        con.close()
    
    return len(df)


def run_sync(
    products: list[str] | None = None,
    resume: bool = False,
    dry_run: bool = False,
    yearly: bool = False,
) -> None:
    """执行全量同步."""
    # 加载映射
    product_map = get_product_exchange_map(DUCKDB_PATH)
    all_products = sorted(product_map.keys())

    if products:
        # 过滤指定产品
        target = [p.upper() for p in products if p.upper() in product_map]
        missing = [p.upper() for p in products if p.upper() not in product_map]
        if missing:
            LOG.warning(f"Products not in contract_specs: {missing}")
    else:
        target = all_products

    # 断点续传
    ckpt = load_checkpoint() if resume else {"completed": [], "failed": [], "stats": {}}
    if resume and ckpt["completed"]:
        already = set(ckpt["completed"])
        target = [p for p in target if p not in already]
        LOG.info(f"Resuming: {len(already)} completed, {len(target)} remaining")

    if dry_run:
        LOG.info(f"Dry run: {len(target)} products to sync")
        for p in target[:10]:
            LOG.info(f"  {p} → {product_map[p]}")
        if len(target) > 10:
            LOG.info(f"  ... +{len(target)-10} more")
        return

    # 连接 CH
    LOG.info(f"Connecting to ClickHouse: {CH_CONFIG['host']}:{CH_CONFIG['port']}")
    ch = clickhouse_connect.get_client(**CH_CONFIG)

    # 同步
    total_rows = 0
    total_time = 0
    success = 0
    failed = 0

    pbar = tqdm(target, desc="Syncing", unit="product", ncols=100)
    for product in pbar:
        exchange = product_map[product]
        pbar.set_postfix_str(f"{exchange}.{product}", refresh=False)

        t0 = time.time()
        try:
            if yearly:
                rows = sync_one_product_yearly(ch, DUCKDB_PATH, product, exchange)
            else:
                rows = sync_one_product(ch, DUCKDB_PATH, product, exchange)
            elapsed = time.time() - t0
            total_rows += rows
            total_time += elapsed
            success += 1
            mark_completed(ckpt, product, rows, elapsed)
            pbar.set_postfix_str(f"{exchange}.{product} ✓ {rows:,} rows", refresh=False)
        except Exception as e:
            elapsed = time.time() - t0
            total_time += elapsed
            failed += 1
            mark_failed(ckpt, product, str(e))
            pbar.set_postfix_str(f"{exchange}.{product} ✗ {str(e)[:30]}", refresh=False)
            LOG.error(f"Failed {exchange}.{product}: {e}")

    pbar.close()

    # 汇总
    LOG.info("=" * 60)
    LOG.info(f"Sync complete: {success} success, {failed} failed")
    LOG.info(f"Total rows: {total_rows:,}")
    LOG.info(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    LOG.info(f"Avg per product: {total_time/len(target):.2f}s")
    LOG.info(f"Checkpoint saved: {CHECKPOINT_PATH}")

    if ckpt["failed"]:
        LOG.info(f"Failed products: {ckpt['failed']}")
        LOG.info("Run with --resume to retry failed products")


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ClickHouse tick_quote → DuckDB 全量同步",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--products", nargs="*", default=None,
        help="指定产品代码 (默认全部 92 个)",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="断点续传 (跳过已完成的产品)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="仅打印同步计划, 不实际执行",
    )
    p.add_argument(
        "--duckdb", type=str, default=None,
        help="DuckDB 路径 (默认 ~/Public/DataCache/futures_options_daily.duckdb)",
    )
    p.add_argument(
        "--yearly", action="store_true",
        help="按年查询 (用于 AG/AL 等大表, 避免超时)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    global DUCKDB_PATH
    if args.duckdb:
        DUCKDB_PATH = Path(args.duckdb)

    if not DUCKDB_PATH.exists():
        LOG.error(f"DuckDB not found: {DUCKDB_PATH}")
        return 1

    run_sync(
        products=args.products,
        resume=args.resume,
        dry_run=args.dry_run,
        yearly=args.yearly,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
