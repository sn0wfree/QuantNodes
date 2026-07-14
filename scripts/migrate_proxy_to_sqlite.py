# coding: utf-8
"""一次性: 把 iFinD 已落的 19 parquet 灌进 proxy_indices.db.

输入:
  data/ifind_cache/index_proxy/*.parquet  (19 个 iFinD 抓取)

输出:
  data/ifind_cache/proxy_indices.db
    └─ table: proxy_prices
        obs_date TEXT NOT NULL         -- 'YYYY-MM-DD'
        code     TEXT NOT NULL         -- '399324.SZ'
        name     TEXT                  -- '深证红利'
        close    REAL                  -- 主字段
        pre_close REAL / open REAL / high REAL / low REAL
        settle   REAL                  -- 商品指数
        vol      REAL / trans_amt REAL
        src      TEXT NOT NULL         -- 'ifind.quantapi' / 'wind.wsd'
        PRIMARY KEY (code, obs_date)

用法:
  python3.11 scripts/migrate_proxy_to_sqlite.py
"""
from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data/ifind_cache/proxy_indices.db"
SRC_PARQUET_DIR = PROJECT_ROOT / "data/ifind_cache/index_proxy"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS proxy_prices (
    obs_date  TEXT NOT NULL,
    code      TEXT NOT NULL,
    name      TEXT,
    close     REAL,
    pre_close REAL,
    open      REAL,
    high      REAL,
    low       REAL,
    settle    REAL,
    vol       REAL,
    trans_amt REAL,
    src       TEXT NOT NULL,
    PRIMARY KEY (code, obs_date)
);
CREATE INDEX IF NOT EXISTS idx_obs ON proxy_prices(obs_date);
"""

EXPECTED_COLS = [
    "obs_date", "code", "name", "close", "pre_close", "open",
    "high", "low", "settle", "vol", "trans_amt", "src",
]


def main() -> int:
    if not SRC_PARQUET_DIR.exists():
        logging.error("源目录不存在: %s", SRC_PARQUET_DIR)
        return 1

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_SQL)
    logging.info("SQLite 已就绪: %s", DB_PATH)

    n_files = 0
    n_rows = 0
    for p in sorted(SRC_PARQUET_DIR.glob("*.parquet")):
        df = pd.read_parquet(p).copy()
        if "code" not in df.columns:
            df["code"] = p.stem.replace("_", ".")
        if "src" not in df.columns:
            df["src"] = "ifind.quantapi"
        df["obs_date"] = pd.to_datetime(df["obs_date"]).dt.strftime("%Y-%m-%d")
        for col in EXPECTED_COLS:
            if col not in df.columns:
                df[col] = None
        df = df[EXPECTED_COLS]
        df.to_sql("proxy_prices", conn, if_exists="append", index=False)
        n_files += 1
        n_rows += len(df)
        logging.info("  %s rows=%d", p.name, len(df))

    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM proxy_prices").fetchone()[0]
    codes = conn.execute(
        "SELECT COUNT(DISTINCT code) FROM proxy_prices"
    ).fetchone()[0]
    srcs = conn.execute(
        "SELECT src, COUNT(*) FROM proxy_prices GROUP BY src"
    ).fetchall()
    earliest = conn.execute(
        "SELECT MIN(obs_date) FROM proxy_prices"
    ).fetchone()[0]
    latest = conn.execute(
        "SELECT MAX(obs_date) FROM proxy_prices"
    ).fetchone()[0]

    conn.close()

    logging.info("=" * 60)
    logging.info("迁移完成:")
    logging.info("  parquet 文件: %d", n_files)
    logging.info("  写入行数:     %d", n_rows)
    logging.info("  DB 总行数:    %d", total)
    logging.info("  DB 总 codes:  %d", codes)
    logging.info("  时间范围:     %s ~ %s", earliest, latest)
    for src, n in srcs:
        logging.info("  src=%-20s rows=%d", src, n)
    logging.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
