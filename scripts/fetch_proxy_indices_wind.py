# coding: utf-8
"""Wind 抓 17 个 ETF proxy 指数 → 写入 proxy_indices.db (SQLite).

前置:
  1. Wind 终端已启动并登录
  2. Python 环境含 WindPy (Wind 终端自带 / pip install WindPy)
  3. Wind 账号拥有指数日频 2018+ 历史数据权限

输出:
  data/ifind_cache/proxy_indices.db  (append 到 proxy_prices 表)

用法:
  python scripts/fetch_proxy_indices_wind.py
"""
from __future__ import annotations

import logging
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data/ifind_cache/proxy_indices.db"

try:
    from WindPy import w
except ImportError:
    raise SystemExit(
        "WindPy 未安装。请在 Wind 终端自带的 Python 环境下运行此脚本, "
        "或先 pip install WindPy（需 Wind 账号）"
    )

WIND_INDEX_TABLE: list[tuple[str, str, str, str]] = [
    # (code, name_zh, wind_code, target_etf)
    ("399324.SZ",    "深证红利",          "399324.SZ",    "159786"),
    ("000688.SH",    "上证科创板50",      "000688.SH",    "588000"),
    ("000906.SH",    "中证800",           "000906.SH",    "515900"),
    ("931151.CSI",   "中证光伏产业",      "931151.CSI",   "515790"),
    ("930850.CSI",   "中证智能制造",      "930850.CSI",   "515100"),
    ("399998.SZ",    "中证煤炭",          "399998.SZ",    "515220"),
    ("399976.SZ",    "中证新能源汽车",    "399976.SZ",    "515030"),
    ("931450.CSI",   "中证消费/新能车主题","931450.CSI",  "159996"),
    ("931152.CSI",   "中证创新药",        "931152.CSI",   "515080"),
    ("399989.SZ",    "中证医疗",          "399989.SZ",    "512170"),
    ("980017.SZ",    "国证半导体芯片",    "980017.SZ",    "512760/512480"),
    ("399987.SZ",    "中证酒",            "399987.SZ",    "512690"),
    ("H30269.CSI",   "中证红利低波动",    "H30269.CSI",   "512890"),
    ("000922.CSI",   "中证红利",          "000922.CSI",   "512260"),
    ("HSTECH.HK",    "恒生科技",          "HSTECH.HK",    "159740"),
    ("NDX.GI",       "纳斯达克100",       "NDX.GI",       "513300"),
    ("931865.CSI",   "中证半导体",        "931865.CSI",   "159995/补980017"),
]

EXPECTED_COLS = [
    "obs_date", "code", "name", "close", "pre_close", "open",
    "high", "low", "settle", "vol", "trans_amt", "src",
]

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


def main() -> int:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_SQL)
    logging.info("SQLite 已就绪: %s", DB_PATH)

    before_total = conn.execute(
        "SELECT COUNT(*) FROM proxy_prices"
    ).fetchone()[0]
    logging.info("现有行数: %d", before_total)

    logging.info("启动 Wind ...")
    start_ret = w.start()
    if start_ret.ErrorCode:
        raise SystemExit(f"Wind start 失败 ErrorCode={start_ret.ErrorCode}")

    start_date = "2018-01-01"
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    n_ok = 0
    n_err = 0
    for code, name_zh, wind_code, target_etf in WIND_INDEX_TABLE:
        logging.info("→ %s (%s)  wind_code=%s  target=%s",
                     code, name_zh, wind_code, target_etf)

        try:
            r = w.wsd(wind_code, "close,sec_name",
                      start_date, end_date, "PriceAdj=F")
        except Exception as exc:
            logging.warning("  ❌ 异常: %s, skip", exc)
            n_err += 1
            continue

        if r.ErrorCode != 0:
            logging.warning("  ❌ ErrorCode=%d skip", r.ErrorCode)
            n_err += 1
            continue

        if not r.Times or not r.Data:
            logging.warning("  ⚠ empty, skip")
            n_err += 1
            continue

        close_list = r.Data[0]
        name_list = r.Data[1] if len(r.Data) > 1 else [name_zh] * len(r.Times)

        df = pd.DataFrame({
            "obs_date": pd.to_datetime(r.Times).strftime("%Y-%m-%d"),
            "code":     [code] * len(r.Times),
            "name":     name_list,
            "close":    pd.to_numeric(close_list, errors="coerce"),
            "src":      "wind.wsd",
        })
        df = df.dropna(subset=["close"])
        df = df.drop_duplicates(subset=["obs_date"], keep="last")

        for col in EXPECTED_COLS:
            if col not in df.columns:
                df[col] = None
        df = df[EXPECTED_COLS]

        df.to_sql("proxy_prices", conn, if_exists="append", index=False)
        n_ok += 1
        logging.info(
            "  ✓ rows=%d  [%s ~ %s]  close NaN%%=%.2f%%",
            len(df),
            df["obs_date"].min(),
            df["obs_date"].max(),
            df["close"].isna().mean() * 100,
        )
        time.sleep(0.5)

    try:
        w.stop()
    except Exception:
        pass
    logging.info("Wind 已关闭.")
    conn.commit()

    after_total = conn.execute(
        "SELECT COUNT(*) FROM proxy_prices"
    ).fetchone()[0]
    wind_codes = conn.execute(
        "SELECT COUNT(DISTINCT code) FROM proxy_prices WHERE src='wind.wsd'"
    ).fetchone()[0]
    earliest = conn.execute(
        "SELECT MIN(obs_date) FROM proxy_prices WHERE src='wind.wsd'"
    ).fetchone()[0]
    latest = conn.execute(
        "SELECT MAX(obs_date) FROM proxy_prices WHERE src='wind.wsd'"
    ).fetchone()[0]
    n_dup = conn.execute(
        "SELECT COUNT(*) FROM ("
        "  SELECT code, obs_date, COUNT(*) c FROM proxy_prices "
        "  GROUP BY code, obs_date HAVING c>1)"
    ).fetchone()[0]

    conn.close()
    logging.info("=" * 60)
    logging.info("Wind 抓取完成:")
    logging.info("  成功:   %d / %d codes", n_ok, len(WIND_INDEX_TABLE))
    logging.info("  失败:   %d codes", n_err)
    logging.info("  新增行: %d", after_total - before_total)
    logging.info("  DB 总:  %d 行", after_total)
    logging.info("  Wind 代码去重: %d", wind_codes)
    logging.info("  Wind 时间范围: %s ~ %s", earliest, latest)
    logging.info("  重复行: %d (PRIMARY KEY 已保证无)", n_dup)
    logging.info("=" * 60)
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
