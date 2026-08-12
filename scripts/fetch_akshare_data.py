#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""akshare 补全数据抓取脚本.

从 akshare 补齐真实数据维度, 写入 DuckDB:
  1. spot_basis_daily     — 现货基差 (2018~2026, 52 品种, 近月/主力双基差)
  2. futures_rule_snapshot — 交易所交易规则快照 (保证金/涨跌停/乘数)

Usage:
    python scripts/fetch_akshare_data.py                    # 全量增量抓取基差 + 规则
    python scripts/fetch_akshare_data.py --only basis       # 仅基差
    python scripts/fetch_akshare_data.py --only rule        # 仅规则
    python scripts/fetch_akshare_data.py --start 2025-01-01 --end 2025-12-31  # 区间抓取
    python scripts/fetch_akshare_data.py --reset            # 删表重建 (全量重抓)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd

DUCKDB_PATH = Path("~/Public/DataCache/futures_options_daily.duckdb").expanduser()
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOG = logging.getLogger(__name__)

BATCH_DAYS = 15            # 每批抓取的天数窗口 (实测 2 周 1.3s)
RETRY = 2                  # 单批失败重试次数
SLEEP = 0.3                # 批间休眠 (秒), 避免过快


# ════════════════════════════════════════════════════════════════════════════
# 1. 现货基差 spot_basis_daily
# ════════════════════════════════════════════════════════════════════════════

SPOT_BASIS_SCHEMA = """
CREATE TABLE IF NOT EXISTS spot_basis_daily (
    trade_date               DATE,
    symbol                   VARCHAR,
    spot_price               DOUBLE,
    near_contract            VARCHAR,
    near_contract_price      DOUBLE,
    dominant_contract        VARCHAR,
    dominant_contract_price  DOUBLE,
    near_basis               DOUBLE,
    dom_basis                DOUBLE,
    near_basis_rate          DOUBLE,
    dom_basis_rate           DOUBLE
)
"""

SPOT_BASIS_COLS = [
    "trade_date", "symbol", "spot_price",
    "near_contract", "near_contract_price",
    "dominant_contract", "dominant_contract_price",
    "near_basis", "dom_basis", "near_basis_rate", "dom_basis_rate",
]


def _basis_fetch_batch(ak, start: date, end: date):
    """调用 akshare 拉一个日期区间的基差, 返回归一化 DataFrame."""
    df = ak.futures_spot_price_daily(
        start_day=start.strftime("%Y%m%d"),
        end_day=end.strftime("%Y%m%d"),
    )
    if df is None or df.empty:
        return df
    df = df.rename(columns={
        "date": "trade_date",
        "spot_price": "spot_price",
        "near_contract": "near_contract",
        "near_contract_price": "near_contract_price",
        "dominant_contract": "dominant_contract",
        "dominant_contract_price": "dominant_contract_price",
        "near_basis": "near_basis",
        "dom_basis": "dom_basis",
        "near_basis_rate": "near_basis_rate",
        "dom_basis_rate": "dom_basis_rate",
    })
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df[SPOT_BASIS_COLS]


def _get_calendar_dates(con) -> list[date]:
    """从 trading_calendar 取全部交易日."""
    rows = con.execute("SELECT trade_date FROM trading_calendar ORDER BY trade_date").fetchall()
    return [r[0] for r in rows]


def _get_existing_dates(con) -> set[date]:
    """spot_basis_daily 中已存在的交易日."""
    try:
        rows = con.execute("SELECT DISTINCT trade_date FROM spot_basis_daily").fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()


def build_spot_basis(con, ak, start: date | None, end: date | None, reset: bool = False) -> int:
    """抓取现货基差, 增量写入 spot_basis_daily."""
    LOG.info("Building spot_basis_daily ...")

    if reset:
        con.execute("DROP TABLE IF EXISTS spot_basis_daily")
    con.execute(SPOT_BASIS_SCHEMA)

    all_dates = _get_calendar_dates(con)
    if start:
        all_dates = [d for d in all_dates if d >= start]
    if end:
        all_dates = [d for d in all_dates if d <= end]

    existing = set()
    if not reset:
        existing = _get_existing_dates(con)

    pending = [d for d in all_dates if d not in existing]
    if not pending:
        LOG.info(f"  no new dates (already have {len(existing)} days)")
        return len(existing)
    LOG.info(f"  {len(pending)} dates to fetch (of {len(all_dates)})")

    # 按 BATCH_DAYS 窗口分批
    batches: list[tuple[date, date]] = []
    i = 0
    while i < len(pending):
        b_start = pending[i]
        b_end = pending[min(i + BATCH_DAYS - 1, len(pending) - 1)]
        batches.append((b_start, b_end))
        i += BATCH_DAYS

    total_rows = 0
    for bi, (b_start, b_end) in enumerate(batches):
        df = None
        for attempt in range(RETRY + 1):
            try:
                df = _basis_fetch_batch(ak, b_start, b_end)
                break
            except Exception as e:
                LOG.warning(f"  batch {bi}/{len(batches)} [{b_start}~{b_end}] attempt {attempt+1} failed: {e}")
                time.sleep(SLEEP * 3)
        if df is None:
            LOG.error(f"  batch {bi}/{len(batches)} [{b_start}~{b_end}] FAILED after {RETRY+1} tries, skip")
            continue
        if df.empty:
            continue
        try:
            con.executemany(
                f"INSERT INTO spot_basis_daily ({', '.join(SPOT_BASIS_COLS)}) "
                f"VALUES ({', '.join(['?'] * len(SPOT_BASIS_COLS))})",
                df[SPOT_BASIS_COLS].itertuples(index=False, name=None),
            )
            total_rows += len(df)
        except Exception as e:
            LOG.error(f"  batch {bi} insert failed: {e}")
        if bi % 20 == 0:
            LOG.info(f"  progress {bi}/{len(batches)} batches, {total_rows:,} rows")
        time.sleep(SLEEP)

    cnt = con.execute("SELECT count() FROM spot_basis_daily").fetchone()[0]
    n_sym = con.execute("SELECT count(DISTINCT symbol) FROM spot_basis_daily").fetchone()[0]
    n_day = con.execute("SELECT count(DISTINCT trade_date) FROM spot_basis_daily").fetchone()[0]
    LOG.info(f"  spot_basis_daily: {cnt:,} rows, {n_sym} symbols, {n_day} days")
    return cnt


# ════════════════════════════════════════════════════════════════════════════
# 2. 交易规则 futures_rule_snapshot
# ════════════════════════════════════════════════════════════════════════════

RULE_SCHEMA = """
CREATE TABLE IF NOT EXISTS futures_rule_snapshot (
    as_of_date             DATE,
    exchange               VARCHAR,
    product                VARCHAR,
    code                   VARCHAR,
    margin_ratio           DOUBLE,
    limit_ratio            DOUBLE,
    contract_multiplier    DOUBLE,
    tick_size              DOUBLE,
    max_order_vol          BIGINT,
    special_note           VARCHAR
)
"""


def build_rule_snapshot(con, ak, reset: bool = False) -> int:
    """抓取最新交易规则快照, 写入 futures_rule_snapshot."""
    LOG.info("Building futures_rule_snapshot ...")

    if reset:
        con.execute("DROP TABLE IF EXISTS futures_rule_snapshot")
    con.execute(RULE_SCHEMA)

    as_of = date.today()
    df = None
    # 若今日无数据, 回退到最近的可用交易日 (最多回退 14 天)
    for back in range(0, 15):
        probe = as_of - timedelta(days=back)
        df = None
        for attempt in range(RETRY + 1):
            try:
                df = ak.futures_rule(date=probe.strftime("%Y%m%d"))
                break
            except Exception as e:
                LOG.warning(f"  futures_rule {probe} attempt {attempt+1} failed: {e}")
                time.sleep(SLEEP)
        if df is not None:
            as_of = probe
            break
        if back % 3 == 0:
            LOG.info(f"  trying earlier date: {probe} ...")
    if df is None:
        LOG.error("  futures_rule fetch FAILED (no recent date available)")
        return 0

    df = df.rename(columns={
        "交易所": "exchange",
        "品种": "product",
        "代码": "code",
        "交易保证金比例": "margin_ratio",
        "涨跌停板幅度": "limit_ratio",
        "合约乘数": "contract_multiplier",
        "最小变动价位": "tick_size",
        "限价单每笔最大下单手数": "max_order_vol",
        "特殊合约参数调整": "special_note",
    })
    df["as_of_date"] = as_of
    rows = df[[
        "as_of_date", "exchange", "product", "code",
        "margin_ratio", "limit_ratio", "contract_multiplier",
        "tick_size", "max_order_vol", "special_note",
    ]].itertuples(index=False, name=None)

    con.execute(f"DELETE FROM futures_rule_snapshot WHERE as_of_date = DATE '{as_of}'")
    con.executemany(
        "INSERT INTO futures_rule_snapshot (as_of_date, exchange, product, code, "
        "margin_ratio, limit_ratio, contract_multiplier, tick_size, max_order_vol, special_note) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )

    cnt = con.execute("SELECT count() FROM futures_rule_snapshot").fetchone()[0]
    n_ex = con.execute("SELECT count(DISTINCT exchange) FROM futures_rule_snapshot").fetchone()[0]
    LOG.info(f"  futures_rule_snapshot: {cnt} rows, {n_ex} exchanges (as_of {as_of})")
    return cnt


# ════════════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════════════

def run(only: str | None = None, start: str | None = None, end: str | None = None,
        reset: bool = False) -> None:
    import warnings
    warnings.filterwarnings("ignore")
    import akshare as ak

    if not DUCKDB_PATH.exists():
        LOG.error(f"DuckDB not found: {DUCKDB_PATH}")
        sys.exit(1)

    LOG.info(f"Connecting: {DUCKDB_PATH}")
    con = duckdb.connect(str(DUCKDB_PATH))

    try:
        t0 = time.time()

        s_date = date.fromisoformat(start) if start else None
        e_date = date.fromisoformat(end) if end else None

        if only is None or only == "basis":
            build_spot_basis(con, ak, start=s_date, end=e_date, reset=reset)

        if only is None or only == "rule":
            build_rule_snapshot(con, ak, reset=reset)

        elapsed = time.time() - t0
        LOG.info(f"Done in {elapsed:.1f}s ({elapsed/60:.1f}min)")

    finally:
        con.close()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="akshare 补全数据抓取 (现货基差 + 交易规则)")
    p.add_argument("--only", choices=["basis", "rule"], help="仅抓取指定数据 (默认全部)")
    p.add_argument("--start", type=str, default=None, help="起始日期 YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="结束日期 YYYY-MM-DD")
    p.add_argument("--reset", action="store_true", help="删表重建 (全量重抓)")
    p.add_argument("--duckdb", type=str, default=None, help="DuckDB 路径")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    global DUCKDB_PATH
    if args.duckdb:
        DUCKDB_PATH = Path(args.duckdb)
    run(only=args.only, start=args.start, end=args.end, reset=args.reset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
