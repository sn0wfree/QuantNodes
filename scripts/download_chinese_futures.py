#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""中文期货/期权日线下载器 (公开数据版).

数据源 (按优先级):
  1. CFFEX 官网  http://www.cffex.com.cn/sj/historysj/{YYYYMM}/zip/{YYYYMM}.zip
  2. SHFE 官网   https://www.shfe.com.cn/reports/tradedata/datadownload/download.json
  3. AKShare     get_futures_daily / futures_zh_daily_sina / futures_main_sina

输出:
  - 本地 Parquet: data/chinese_futures/daily|main/{EXCHANGE}/{SYMBOL}.parquet
  - ClickHouse:  quote.fut_instruments / fut_daily_kline / fut_main_contract_mapping /
                 fut_trading_calendar / fut_download_log

Usage:
    python scripts/download_chinese_futures.py --exchanges CFFEX SHFE
    python scripts/download_chinese_futures.py --full
    python scripts/download_chinese_futures.py --test-auth
    python scripts/download_chinese_futures.py --main-only
"""
from __future__ import annotations

import argparse
import calendar as _cal
import configparser
import datetime as _dt
import gzip
import io
import json
import logging
import os
import random
import re
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
import requests
from tqdm import tqdm

# ════════════════════════════════════════════════════════════════════════════
# Region 1: 常量 / 配置
# ════════════════════════════════════════════════════════════════════════════

EXCHANGES = ["CFFEX", "SHFE", "DCE", "CZCE", "INE", "GFEX"]

# AKShare 主力合约品种映射 (from futures_display_main_sina 2026-07-22)
AKSHARE_MAIN_SYMBOLS = {
    "CFFEX": ["IF0", "TF0", "IH0", "IC0", "TS0", "IM0"],
    "SHFE":  ["FU0", "AL0", "RU0", "ZN0", "CU0", "AU0", "RB0", "PB0", "AG0",
              "BU0", "HC0", "SN0", "NI0", "SP0", "SS0", "AO0", "BR0", "AD0", "OP0"],
    "DCE":   ["V0", "P0", "B0", "M0", "I0", "JD0", "L0", "PP0", "FB0", "Y0",
              "C0", "A0", "J0", "JM0", "CS0", "EG0", "RR0", "EB0", "PG0",
              "LH0", "LG0", "BZ0"],
    "CZCE":  ["TA0", "OI0", "RS0", "RM0", "WH0", "JR0", "SR0", "CF0", "RI0",
              "MA0", "FG0", "LR0", "SF0", "SM0", "CY0", "AP0", "CJ0", "UR0",
              "SA0", "PF0", "PK0", "SH0", "PX0", "PR0", "PL0"],
    "INE":   ["SC0", "NR0", "LU0", "BC0", "EC0"],
    "GFEX":  ["SI0", "LC0", "PS0", "PT0", "PD0"],
}

# AKShare market 参数映射
AKSHARE_MARKET = {e: e for e in EXCHANGES}

# 限速 (min, max) 秒/请求
RATE_LIMITS = {
    "www.cffex.com.cn":          (3.0, 5.0),
    "www.shfe.com.cn":           (2.0, 4.0),
    "stock.finance.sina.com.cn": (1.0, 2.0),
    "hq.sinajs.cn":              (1.0, 2.0),
    "default":                   (1.5, 2.5),
}
RETRY_BACKOFF = [1, 2, 4, 8, 16]
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "chinese_futures"
META_DIR = DATA_ROOT / "_meta"
LOG_DIR = DATA_ROOT / "_log"
FAILED_FILE = META_DIR / "_failed_months.json"
LAST_RUN_FILE = META_DIR / "_last_run.json"
PRIORITY_FILE = META_DIR / "_source_priority.json"
LOG_FILE = LOG_DIR / "download.log"

CONN_INI = PROJECT_ROOT / "conn.ini"
DEFAULT_CH = {"host": "localhost", "port": 8123, "user": "default",
              "passwd": "", "database": "quote"}

# ClickHouse DDL (idempotent)
SCHEMA_SQL = {
    "fut_instruments": """
        CREATE TABLE IF NOT EXISTS {db}.fut_instruments (
            exchange          LowCardinality(String),
            product_id        LowCardinality(String),
            symbol            String,
            full_symbol       String,
            instrument_name   String DEFAULT '',
            ins_class         LowCardinality(String),
            option_class      LowCardinality(String) DEFAULT '',
            strike_price      Nullable(Decimal(18,4)),
            underlying_symbol String DEFAULT '',
            list_date         Nullable(Date),
            expire_date       Nullable(Date),
            volume_multiple   Nullable(Decimal(18,4)),
            price_tick        Nullable(Decimal(18,6)),
            is_main           UInt8 DEFAULT 0,
            download_ts       DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree(download_ts)
          ORDER BY (full_symbol)
          PARTITION BY exchange
    """,
    "fut_daily_kline": """
        CREATE TABLE IF NOT EXISTS {db}.fut_daily_kline (
            exchange       LowCardinality(String),
            product_id     LowCardinality(String) DEFAULT '',
            symbol         String,
            full_symbol    String,
            trade_date     Date,
            open           Nullable(Decimal(18,6)),
            high           Nullable(Decimal(18,6)),
            low            Nullable(Decimal(18,6)),
            close          Nullable(Decimal(18,6)),
            settle         Nullable(Decimal(18,6)),
            pre_settle     Nullable(Decimal(18,6)),
            pre_close      Nullable(Decimal(18,6)),
            volume         Nullable(Int64),
            open_interest  Nullable(Int64),
            amount         Nullable(Decimal(20,4)),
            delta          Nullable(Decimal(18,6)),
            source         LowCardinality(String),
            download_ts    DateTime DEFAULT now()
        ) ENGINE = ReplacingMergeTree(download_ts)
          ORDER BY (full_symbol, trade_date)
          PARTITION BY (exchange, toYear(trade_date))
    """,
    "fut_main_contract_mapping": """
        CREATE TABLE IF NOT EXISTS {db}.fut_main_contract_mapping (
            exchange      LowCardinality(String),
            product_id    LowCardinality(String),
            trade_date    Date,
            main_symbol   String,
            open_interest Nullable(Int64),
            volume        Nullable(Int64)
        ) ENGINE = ReplacingMergeTree()
          ORDER BY (exchange, product_id, trade_date)
    """,
    "fut_trading_calendar": """
        CREATE TABLE IF NOT EXISTS {db}.fut_trading_calendar (
            exchange    LowCardinality(String),
            trade_date  Date,
            is_trading  UInt8
        ) ENGINE = ReplacingMergeTree()
          ORDER BY (exchange, trade_date)
    """,
    "fut_download_log": """
        CREATE TABLE IF NOT EXISTS {db}.fut_download_log (
            exchange        LowCardinality(String),
            symbol          String,
            freq            LowCardinality(String),
            source          LowCardinality(String),
            rows            UInt64,
            last_trade_date Nullable(Date),
            started_at      DateTime,
            finished_at     DateTime,
            status          LowCardinality(String),
            error_msg       String DEFAULT ''
        ) ENGINE = MergeTree()
          ORDER BY (exchange, symbol, started_at)
          PARTITION BY toYYYYMM(started_at)
    """,
}

# ════════════════════════════════════════════════════════════════════════════
# Region 2: 日志 / 限速 / HTTP
# ════════════════════════════════════════════════════════════════════════════


def _setup_logging() -> logging.Logger:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("cn_futures")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


LOG = _setup_logging()


@dataclass
class RateLimiter:
    """每域名独立的滑动限速器."""
    rate_map: dict[str, tuple[float, float]] = field(default_factory=dict)
    _last: dict[str, float] = field(default_factory=lambda: defaultdict(lambda: 0.0))

    def __post_init__(self):
        self.rate_map = {**RATE_LIMITS, **self.rate_map}

    def _rate(self, url: str) -> tuple[float, float]:
        for host, rate in self.rate_map.items():
            if host in url:
                return rate
        return self.rate_map["default"]

    def wait(self, url: str) -> None:
        mn, mx = self._rate(url)
        now = time.monotonic()
        elapsed = now - self._last[url]
        sleep_for = max(0.0, mn - elapsed) + random.uniform(0, mx - mn)
        if sleep_for > 0:
            time.sleep(sleep_for)
            LOG.debug("rate wait %.2fs (range %.1f-%.1f) for %s",
                      sleep_for, mn, mx, url)
        self._last[url] = time.monotonic()


def http_get(url: str, *, rate: RateLimiter, retries: int = 5,
             timeout: int = 30, referer: str | None = None,
             stream: bool = False) -> requests.Response:
    """带限速 + 指数退避 + UA 轮换的 GET.

    Returns:
        requests.Response (with .status_code, .content, .text)

    Raises:
        requests.HTTPError on 4xx/5xx after retries exhausted
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            rate.wait(url)
            headers = {"User-Agent": random.choice(USER_AGENTS),
                       "Accept": "*/*",
                       "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
            if referer:
                headers["Referer"] = referer
            r = requests.get(url, headers=headers, timeout=timeout,
                             allow_redirects=True, stream=stream)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                # 不重试 404
                r.raise_for_status()
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} {url}")
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            if attempt < retries - 1:
                sleep = RETRY_BACKOFF[min(attempt, len(RETRY_BACKOFF) - 1)]
                LOG.warning(f"GET {url[:80]} 失败 (第{attempt+1}次): {e}; {sleep}s 后重试")
                time.sleep(sleep + random.uniform(0, 2))
    raise last_exc or RuntimeError(f"GET {url} failed after {retries} retries")


def _load_conn_ini() -> dict[str, Any]:
    """从 conn.ini 加载 ClickHouse 配置 (失败回退默认)."""
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


# ════════════════════════════════════════════════════════════════════════════
# Region 3: 失败月份管理 + 运行元数据
# ════════════════════════════════════════════════════════════════════════════


def _load_failed() -> dict[str, dict[str, dict[str, Any]]]:
    if not FAILED_FILE.exists():
        return {}
    try:
        return json.loads(FAILED_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_failed(data: dict[str, dict[str, dict[str, Any]]]) -> None:
    FAILED_FILE.parent.mkdir(parents=True, exist_ok=True)
    FAILED_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                           encoding="utf-8")


def _record_failure(source: str, key: str, error: str) -> None:
    """记录一次失败, 超过 3 次移入 abandoned."""
    data = _load_failed()
    data.setdefault(source, {}).setdefault(key, {"retries": 0, "errors": []})
    entry = data[source][key]
    entry["retries"] += 1
    entry["last_error"] = error[:200]
    entry["last_seen"] = _dt.datetime.now().isoformat()
    entry.setdefault("first_seen", entry["last_seen"])
    if entry["retries"] >= 3:
        entry["abandoned"] = True
    _save_failed(data)


def _clear_failure(source: str, key: str) -> None:
    data = _load_failed()
    if source in data and key in data[source]:
        data[source].pop(key, None)
        if not data[source]:
            data.pop(source, None)
        _save_failed(data)


def _write_last_run(stats: dict[str, Any]) -> None:
    LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    LAST_RUN_FILE.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                             encoding="utf-8")


# ════════════════════════════════════════════════════════════════════════════
# Region 4: 数据源 — CFFEX 官网
# ════════════════════════════════════════════════════════════════════════════

CFFEX_BASE = "http://www.cffex.com.cn"
CFFEX_HISTORY_URL = f"{CFFEX_BASE}/sj/historysj/{{ym}}/zip/{{ym}}.zip"


def _cffex_zip_url(ym: str) -> str:
    return CFFEX_HISTORY_URL.format(ym=ym)


def fetch_cffex_daily(start: _dt.date, end: _dt.date,
                      contracts: list[str] | None = None,
                      rate: RateLimiter | None = None) -> pd.DataFrame:
    """下载 CFFEX 月度 ZIP, 解析 GBK CSV.

    Columns returned (标准化后):
        trade_date, symbol, full_symbol, exchange, open, high, low, close,
        settle, pre_settle, volume, open_interest, amount, delta
    """
    rate = rate or RateLimiter()
    rows: list[pd.DataFrame] = []
    months = _months_between(start, end)
    for ym in tqdm(months, desc="CFFEX", unit="month", ncols=80, leave=False):
        url = _cffex_zip_url(ym)
        try:
            r = http_get(url, rate=rate, referer=CFFEX_BASE)
        except Exception as e:
            LOG.warning(f"CFFEX {ym}: 下载失败 {e}")
            _record_failure("cffex", ym, str(e))
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                for csv_name in zf.namelist():
                    if not csv_name.endswith(".csv"):
                        continue
                    with zf.open(csv_name) as f:
                        text = io.TextIOWrapper(f, encoding="gbk").read()
                        df_day = _parse_cffex_csv(text, csv_name)
                        if df_day is not None and not df_day.empty:
                            rows.append(df_day)
            _clear_failure("cffex", ym)
        except Exception as e:
            LOG.warning(f"CFFEX {ym}: 解析失败 {e}")
            _record_failure("cffex", ym, str(e))
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    if contracts:
        df = df[df["symbol"].isin(contracts)]
    df = df.drop_duplicates(subset=["full_symbol", "trade_date"], keep="last")
    return df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _parse_cffex_csv(text: str, csv_name: str) -> pd.DataFrame | None:
    """解析 CFFEX 单日 CSV.

    原始列: 合约代码,今开盘,最高价,最低价,成交量,成交金额,持仓量,持仓变化,
           今收盘,今结算,前结算,涨跌1,涨跌2,Delta
    """
    try:
        df = pd.read_csv(io.StringIO(text), dtype=str)
    except Exception:
        return None
    if df.empty or "合约代码" not in df.columns:
        return None
    df.columns = [c.strip() for c in df.columns]
    df["合约代码"] = df["合约代码"].str.strip()
    df = df[df["合约代码"].notna() & (df["合约代码"] != "")]
    if df.empty:
        return None

    trade_date = _parse_cffex_date_from_filename(csv_name)
    if trade_date is None:
        return None

    out = pd.DataFrame({
        "trade_date": trade_date,
        "symbol": df["合约代码"].values,
        "full_symbol": "CFFEX." + df["合约代码"].values,
        "exchange": "CFFEX",
        "open":   _num(df["今开盘"]),
        "high":   _num(df["最高价"]),
        "low":    _num(df["最低价"]),
        "close":  _num(df["今收盘"]),
        "settle": _num(df["今结算"]),
        "pre_settle": _num(df["前结算"]),
        "volume": _num(df["成交量"], int64=True),
        "open_interest": _num(df["持仓量"], int64=True),
        "amount": _num(df["成交金额"]),
        "delta": _num(df.get("Delta")),
    })
    # 标记产品代码 (IF, IC, IO, TS, ...)
    out["product_id"] = out["symbol"].apply(_extract_product_id)
    return out


_CFFEX_DATE_RE = re.compile(r"(\d{8})")


def _parse_cffex_date_from_filename(name: str) -> _dt.date | None:
    m = _CFFEX_DATE_RE.search(name)
    if not m:
        return None
    try:
        return _dt.datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def _extract_product_id(symbol: str) -> str:
    """从合约代码提取品种代码 (IF2509 -> IF, IO2509-C-4000 -> IO)."""
    if not isinstance(symbol, str) or not symbol:
        return ""
    m = re.match(r"^([A-Za-z]+)", symbol)
    return m.group(1).upper() if m else ""


def _num(series: pd.Series | None, int64: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    s = pd.to_numeric(series.replace({"": None, "-": None, ",": ""}),
                      errors="coerce")
    if int64:
        return s.astype("Int64")
    return s.astype("float64")


def _months_between(start: _dt.date, end: _dt.date) -> list[str]:
    """返回 YYYYMM 字符串列表 (含两端)."""
    months = []
    cur = _dt.date(start.year, start.month, 1)
    end_m = _dt.date(end.year, end.month, 1)
    while cur <= end_m:
        months.append(cur.strftime("%Y%m"))
        cur = _add_month(cur, 1)
    return months


def _add_month(d: _dt.date, n: int) -> _dt.date:
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    return _dt.date(y, m, 1)


# ════════════════════════════════════════════════════════════════════════════
# Region 5: 数据源 — SHFE 官网
# ════════════════════════════════════════════════════════════════════════════

SHFE_BASE = "https://www.shfe.com.cn"
SHFE_INDEX_URL = f"{SHFE_BASE}/reports/tradedata/datadownload/download.json"


@dataclass
class _ShfeZip:
    url: str
    year: int
    kind: str  # 'futures' | 'option'


def fetch_shfe_index(rate: RateLimiter) -> list[_ShfeZip]:
    """解析 SHFE download.json, 返回所有可用 ZIP 列表."""
    r = http_get(SHFE_INDEX_URL, rate=rate, referer=SHFE_BASE)
    data = json.loads(r.content)
    out: list[_ShfeZip] = []
    for item in data.get("data", {}).get("historicalData", []):
        kind = "futures" if item.get("type") == "1" else "option"
        url = SHFE_BASE + item["url"]
        out.append(_ShfeZip(url=url, year=int(item["year"]), kind=kind))
    for item in data.get("data", {}).get("latestData", []):
        kind = "futures" if item.get("type") == "1" else "option"
        url = SHFE_BASE + item["url"]
        out.append(_ShfeZip(url=url, year=int(item["year"]), kind=kind))
    return out


def fetch_shfe_daily(start: _dt.date, end: _dt.date,
                     kind: str = "futures",
                     rate: RateLimiter | None = None) -> pd.DataFrame:
    """下载 SHFE 历史 ZIP 并解析. kind: 'futures' | 'option'."""
    rate = rate or RateLimiter()
    try:
        zips = fetch_shfe_index(rate)
    except Exception as e:
        LOG.error(f"SHFE index: {e}")
        return pd.DataFrame()
    zips = [z for z in zips if z.kind == kind and start.year <= z.year <= end.year]
    rows: list[pd.DataFrame] = []
    zips_sorted = sorted(zips, key=lambda x: x.year)
    for z in tqdm(zips_sorted, desc=f"SHFE {kind}", unit="year", ncols=80, leave=False):
        try:
            r = http_get(z.url, rate=rate, referer=SHFE_BASE, timeout=60)
        except Exception as e:
            LOG.warning(f"SHFE {z.year} {kind}: 下载失败 {e}")
            _record_failure("shfe", f"{z.year}-{kind}", str(e))
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                for name in zf.namelist():
                    if not (name.endswith(".xls") or name.endswith(".xlsx")):
                        continue
                    with zf.open(name) as f:
                        df_xls = _parse_shfe_xls(f.read(), name, kind=kind)
                    if df_xls is not None and not df_xls.empty:
                        rows.append(df_xls)
            _clear_failure("shfe", f"{z.year}-{kind}")
        except Exception as e:
            LOG.warning(f"SHFE {z.year} {kind}: 解析失败 {e}")
            _record_failure("shfe", f"{z.year}-{kind}", str(e))
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)]
    df = df.drop_duplicates(subset=["full_symbol", "trade_date"], keep="last")
    return df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def _parse_shfe_xls(content: bytes, name: str, kind: str) -> pd.DataFrame | None:
    """解析 SHFE 单月 XLS.

    列(2 行表头, 中文行+英文行): 合约/Contract, 日期/Date, 前收盘/pre close,
      前结算/Pre settle, 开盘价/Open, 最高价/High, 最低价/Low,
      收盘价/Close, 结算价/Settle, 涨跌1/ch1, 涨跌2/ch2, 成交量/Volume,
      成交金额/Amount, 持仓量/OI
    """
    ext = name.split(".")[-1].lower()
    engine = "openpyxl" if ext == "xlsx" else "xlrd"
    try:
        df = pd.read_excel(io.BytesIO(content), engine=engine, header=None)
    except Exception as e:
        LOG.debug(f"SHFE {name}: read_excel failed: {e}")
        return None
    if df.empty or df.shape[1] < 13:
        return None
    # 找表头行: 包含 'Contract' 或 '合约' 的行
    hdr_idx = None
    for i in range(min(10, len(df))):
        vals = df.iloc[i].astype(str).str.strip().tolist()
        if "合约" in vals or "Contract" in vals:
            hdr_idx = i
            break
    if hdr_idx is None:
        return None
    # 取表头 (使用英文表头, 在 +1 行)
    en_hdr = df.iloc[hdr_idx + 1].astype(str).str.strip().tolist() \
        if hdr_idx + 1 < len(df) else []
    cn_hdr = df.iloc[hdr_idx].astype(str).str.strip().tolist()
    cols = en_hdr if any(h for h in en_hdr) else cn_hdr
    data = df.iloc[hdr_idx + 2:].copy()
    data.columns = cols
    # 必填列
    if "Contract" not in data.columns and "合约" not in data.columns:
        return None
    contract_col = "Contract" if "Contract" in data.columns else "合约"
    date_col = "Date" if "Date" in data.columns else "日期"

    sym = data[contract_col].ffill().astype(str).str.strip()
    dt = pd.to_datetime(data[date_col], errors="coerce")
    valid = (sym != "") & sym.notna() & dt.notna()
    if not valid.any():
        return None
    data = data.loc[valid].copy()
    data["__symbol"] = sym[valid].values
    data["__trade_date"] = dt[valid].values

    out = pd.DataFrame({
        "trade_date": pd.to_datetime(data["__trade_date"]).dt.date,
        "symbol": data["__symbol"].str.upper().values,
        "full_symbol": "SHFE." + data["__symbol"].str.upper().values,
        "exchange": "SHFE",
        "open":   _num(data.get("Open", data.get("开盘价"))),
        "high":   _num(data.get("High", data.get("最高价"))),
        "low":    _num(data.get("Low", data.get("最低价"))),
        "close":  _num(data.get("Close", data.get("收盘价"))),
        "settle": _num(data.get("Settle", data.get("结算价"))),
        "pre_settle": _num(data.get("Pre settle", data.get("前结算"))),
        "pre_close":  _num(data.get("pre close", data.get("前收盘"))),
        "volume": _num(data.get("Volume", data.get("成交量")), int64=True),
        "open_interest": _num(data.get("OI", data.get("持仓量")), int64=True),
        "amount": _num(data.get("Amount", data.get("成交金额"))),
    })
    out["product_id"] = out["symbol"].apply(_extract_product_id)
    out = out.dropna(subset=["trade_date", "symbol"])
    return out


# ════════════════════════════════════════════════════════════════════════════
# Region 6: 数据源 — AKShare (兜底 + 主力 + 其他交易所)
# ════════════════════════════════════════════════════════════════════════════


def _akshare_available() -> bool:
    try:
        import akshare as ak  # noqa
        return True
    except ImportError:
        return False


def fetch_akshare_daily(exchange: str, start: _dt.date, end: _dt.date,
                        rate: RateLimiter | None = None) -> pd.DataFrame:
    """AKShare get_futures_daily: 某交易所某日期区间的所有合约日线.

    仅支持 CFFEX/SHFE/CZCE/INE/GFEX (DCE 反爬, 数据不可用).
    """
    if not _akshare_available():
        return pd.DataFrame()
    if exchange == "DCE":
        LOG.warning("DCE: AKShare 反爬阻断, get_futures_daily 不可用, 跳过")
        return pd.DataFrame()
    import akshare as ak
    try:
        df = ak.get_futures_daily(
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            market=exchange,
        )
    except Exception as e:
        LOG.warning(f"AKShare get_futures_daily {exchange}: {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    # 列: symbol, date, open, high, low, close, volume, open_interest,
    #      turnover, settle, pre_settle, variety
    rename = {"symbol": "symbol", "date": "trade_date",
              "open": "open", "high": "high", "low": "low", "close": "close",
              "volume": "volume", "open_interest": "open_interest",
              "turnover": "amount", "settle": "settle",
              "pre_settle": "pre_settle", "variety": "_variety"}
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "symbol" not in df.columns:
        return pd.DataFrame()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["full_symbol"] = exchange + "." + df["symbol"].astype(str).str.upper()
    df["exchange"] = exchange
    df["product_id"] = df["symbol"].apply(_extract_product_id)
    cols = ["trade_date", "symbol", "full_symbol", "exchange", "product_id",
            "open", "high", "low", "close", "settle", "pre_settle",
            "volume", "open_interest", "amount"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    df = df.drop_duplicates(subset=["full_symbol", "trade_date"], keep="last")
    return df.sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def fetch_main_contract_daily(symbol: str, exchange: str,
                              rate: RateLimiter | None = None) -> pd.DataFrame:
    """通过 AKShare 拉取主力连续 (如 IF0, M0, RB0) 日线.

    数据源: futures_zh_daily_sina (新浪公开接口, 经 AKShare).
    """
    if not _akshare_available():
        return pd.DataFrame()
    import akshare as ak
    try:
        df = ak.futures_zh_daily_sina(symbol=symbol)
    except Exception as e:
        LOG.debug(f"AKShare futures_zh_daily_sina {symbol}: {e}")
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    # 列: date, open, high, low, close, volume, hold, settle
    df = df.rename(columns={"date": "trade_date", "hold": "open_interest"})
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["full_symbol"] = f"{exchange}.{symbol}"
    df["exchange"] = exchange
    df["symbol"] = symbol
    df["product_id"] = _extract_product_id(symbol)
    df["amount"] = None
    df["source"] = "akshare_main"
    cols = ["trade_date", "symbol", "full_symbol", "exchange", "product_id",
            "open", "high", "low", "close", "settle", "volume",
            "open_interest", "amount"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols].sort_values("trade_date").reset_index(drop=True)


# ════════════════════════════════════════════════════════════════════════════
# Region 7: Parquet 备份 + ClickHouse 写入
# ════════════════════════════════════════════════════════════════════════════


def _parquet_path(exchange: str, symbol: str, sub: str = "daily") -> Path:
    product = _extract_product_id(symbol)
    d = DATA_ROOT / sub / exchange / product
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{symbol}.parquet"


def save_parquet(df: pd.DataFrame, exchange: str, symbol: str,
                 sub: str = "daily") -> Path:
    """保存到 Parquet (snappy 压缩, 覆盖式).

    保留已有 Parquet 中的更早数据, 合并增量写入.
    """
    if df is None or df.empty:
        return _parquet_path(exchange, symbol, sub)
    path = _parquet_path(exchange, symbol, sub)
    if path.exists():
        try:
            old = pd.read_parquet(path)
            if not old.empty:
                df = pd.concat([old, df], ignore_index=True)
                df = df.drop_duplicates(
                    subset=["full_symbol", "trade_date"], keep="last")
                df = df.sort_values("trade_date").reset_index(drop=True)
        except Exception as e:
            LOG.warning(f"读旧 Parquet 失败 {path}: {e}; 覆盖写入")
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    return path


class ClickHouseWriter:
    """ClickHouse 写入器 (优雅降级: 连接失败时不抛错)."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or _load_conn_ini()
        self._node = None
        self._connected = False
        self._connect_error: str | None = None
        self._permission_warned = False

    def _ensure(self) -> bool:
        if self._connected:
            return True
        if self._connect_error:
            return False
        try:
            from QuantNodes.database_node import ClickHouseNode
            self._node = ClickHouseNode(
                host=self.cfg["host"], port=self.cfg["port"],
                user=self.cfg["user"], passwd=self.cfg["passwd"],
                database=self.cfg["database"],
            )
            self._node.connect()
            self._connected = True
            LOG.info(f"ClickHouse 连接成功: {self.cfg['host']}:{self.cfg['port']}/{self.cfg['database']}")
            return True
        except Exception as e:
            self._connect_error = str(e)
            LOG.warning(f"ClickHouse 连接失败, 后续跳过 CH 写入: {e}")
            return False

    def ensure_tables(self) -> None:
        if not self._ensure():
            return
        db = self.cfg["database"]
        host = self.cfg["host"]
        port = self.cfg["port"]
        user = self.cfg["user"]
        passwd = self.cfg["passwd"]
        url = f"http://{host}:{port}/"
        auth = (user, passwd) if user else None
        for tbl, ddl in SCHEMA_SQL.items():
            sql = ddl.format(db=db).strip()
            try:
                r = requests.post(url, params={"database": db, "query": sql},
                                  auth=auth, timeout=30)
                if r.status_code != 200:
                    err = r.text[:200]
                    if "Not enough privileges" in err or "ACCESS_DENIED" in err:
                        if not self._permission_warned:
                            LOG.warning(
                                f"ClickHouse 用户 {user} 缺少 CREATE 权限, "
                                f"DDL/INSERT 将被跳过 (Parquet 仍正常写入). "
                                f"授权方式: GRANT CREATE, INSERT ON {db}.* TO {user}"
                            )
                            self._permission_warned = True
                        return  # 后续 DDL 都跳过
                    LOG.warning(f"DDL {tbl} 失败: {r.status_code} {err}")
            except Exception as e:
                LOG.warning(f"DDL {tbl} 异常: {e}")

    def upsert_daily(self, df: pd.DataFrame, exchange: str, source: str) -> None:
        if df is None or df.empty or self._permission_warned or not self._ensure():
            return
        cols = ["exchange", "product_id", "symbol", "full_symbol",
                "trade_date", "open", "high", "low", "close", "settle",
                "pre_settle", "pre_close", "volume", "open_interest",
                "amount", "delta", "source"]
        for c in cols:
            if c not in df.columns:
                df[c] = None
        sub = df[cols].copy()
        sub["trade_date"] = pd.to_datetime(sub["trade_date"]).dt.strftime("%Y-%m-%d")
        try:
            self._insert_df_http(sub, "fut_daily_kline")
        except Exception as e:
            LOG.warning(f"ClickHouse insert fut_daily_kline 失败: {e}")

    def log_download(self, exchange: str, symbol: str, source: str,
                     rows: int, last_trade_date: _dt.date | None,
                     status: str, error: str = "",
                     started_at: _dt.datetime | None = None,
                     finished_at: _dt.datetime | None = None) -> None:
        if self._permission_warned or not self._ensure():
            return
        try:
            ts = lambda x: (x or _dt.datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
            last_dt = "NULL" if last_trade_date is None else f"'{last_trade_date}'"
            err = error[:200].replace("'", "''")
            sql = (
                f"INSERT INTO {self.cfg['database']}.fut_download_log "
                "(exchange, symbol, freq, source, rows, last_trade_date, "
                "started_at, finished_at, status, error_msg) VALUES "
                f"('{exchange}', '{symbol}', 'daily', '{source}', {rows}, "
                f"{last_dt}, '{ts(started_at)}', '{ts(finished_at)}', "
                f"'{status}', '{err}')"
            )
            self._node.execute(sql)
        except Exception as e:
            LOG.debug(f"ClickHouse log_download 失败: {e}")

    def _insert_df_http(self, df: pd.DataFrame, table: str) -> None:
        """通过 HTTP JSONEachRow 插入 DataFrame.

        CHBase 的 insert_df 在 HTTP 接口下会失败 (没有 .insert 方法),
        这里用 JSONEachRow 格式 POST 替代.
        """
        if not self._connected:
            return
        db = self.cfg["database"]
        records = []
        for _, row in df.iterrows():
            d = {}
            for c in df.columns:
                v = row[c]
                if v is None:
                    d[c] = None
                elif isinstance(v, float) and pd.isna(v):
                    d[c] = None
                elif hasattr(v, "isoformat"):
                    d[c] = v.isoformat()
                elif isinstance(v, pd.Timestamp):
                    d[c] = v.isoformat()
                else:
                    d[c] = v
            records.append(d)
        body = "\n".join(json.dumps(r, ensure_ascii=False, default=str)
                         for r in records)
        host = self.cfg["host"]
        port = self.cfg["port"]
        user = self.cfg["user"]
        passwd = self.cfg["passwd"]
        url = f"http://{host}:{port}/"
        params = {"database": db, "query": f"INSERT INTO {table} FORMAT JSONEachRow"}
        auth = (user, passwd) if user else None
        r = requests.post(url, params=params, data=body.encode("utf-8"),
                          auth=auth, timeout=60)
        if r.status_code != 200:
            err = r.text[:300]
            if "Not enough privileges" in err or "ACCESS_DENIED" in err:
                if not self._permission_warned:
                    LOG.warning(
                        f"ClickHouse 权限不足, INSERT 被跳过. "
                        f"请执行: GRANT INSERT ON {db}.* TO {user}"
                    )
                    self._permission_warned = True
                return
            raise RuntimeError(f"CH INSERT failed: {r.status_code} {err}")


# ════════════════════════════════════════════════════════════════════════════
# Region 8: 编排器
# ════════════════════════════════════════════════════════════════════════════

SOURCE_PRIORITY: dict[str, list[str]] = {
    "CFFEX": ["cffex", "akshare"],
    "SHFE":  ["shfe_futures", "akshare"],
    "DCE":   ["akshare_main"],   # DCE 主力只能用 Sina, 主力走专门路径
    "CZCE":  ["akshare"],
    "INE":   ["akshare"],
    "GFEX":  ["akshare"],
}


@dataclass
class RunStats:
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    rows: int = 0
    started: _dt.datetime = field(default_factory=_dt.datetime.now)


def _safe(fn: Callable, *args, **kwargs) -> pd.DataFrame:
    try:
        df = fn(*args, **kwargs)
        if df is None:
            return pd.DataFrame()
        return df
    except Exception as e:
        LOG.warning(f"{fn.__name__} 调用失败: {e}")
        return pd.DataFrame()


def run_one(exchanges: list[str], mode: str = "incremental",
            write_parquet: bool = True, write_clickhouse: bool = True,
            include_main: bool = True,
            skip_exchange_scan: bool = False,
            ch_writer: ClickHouseWriter | None = None,
            stats: RunStats | None = None) -> RunStats:
    """主入口: 按交易所 × 合约逐个下载."""
    stats = stats or RunStats()
    ch = ch_writer or ClickHouseWriter()
    if write_clickhouse:
        ch.ensure_tables()
    rate = RateLimiter()

    # 1. 主力连续合约
    if include_main:
        total_mains = sum(len(AKSHARE_MAIN_SYMBOLS.get(ex, [])) for ex in exchanges)
        pbar = tqdm(total=total_mains, desc="主力连续", unit="contract", ncols=80)
        for ex in exchanges:
            for sym in AKSHARE_MAIN_SYMBOLS.get(ex, []):
                stats.attempted += 1
                started_at = _dt.datetime.now()
                try:
                    df = fetch_main_contract_daily(sym, ex, rate)
                except Exception as e:
                    LOG.warning(f"main {ex}.{sym}: {e}")
                    df = pd.DataFrame()
                if df.empty:
                    stats.failed += 1
                    if write_clickhouse:
                        ch.log_download(ex, sym, "akshare_main", 0, None,
                                        "failed", str(df), started_at, _dt.datetime.now())
                    pbar.update(1)
                    pbar.set_postfix_str(f"{ex}.{sym} ❌")
                    continue
                last_dt = df["trade_date"].max()
                last_dt = last_dt.date() if hasattr(last_dt, "date") else last_dt
                if write_parquet:
                    save_parquet(df, ex, sym, sub="main")
                if write_clickhouse:
                    ch.upsert_daily(df, ex, source="akshare_main")
                stats.succeeded += 1
                stats.rows += len(df)
                if write_clickhouse:
                    ch.log_download(ex, sym, "akshare_main", len(df), last_dt,
                                    "success", "", started_at, _dt.datetime.now())
                pbar.update(1)
                pbar.set_postfix_str(f"{ex}.{sym} ✓")
        pbar.close()

    # 2. 交易所全合约日线
    if not skip_exchange_scan:
        for ex in exchanges:
            for src in SOURCE_PRIORITY.get(ex, []):
                if src == "cffex":
                    _run_cffex(ex, mode, rate, ch, write_parquet, write_clickhouse, stats)
                elif src == "shfe_futures":
                    _run_shfe("futures", ex, mode, rate, ch, write_parquet, write_clickhouse, stats)
                elif src == "shfe_option":
                    _run_shfe("option", ex, mode, rate, ch, write_parquet, write_clickhouse, stats)
                elif src == "akshare":
                    _run_akshare(ex, mode, rate, ch, write_parquet, write_clickhouse, stats)
                # shfe 处理会跑完所有年份, 跳出循环
                if src in ("cffex", "akshare", "shfe_futures"):
                    break

    stats.finished = _dt.datetime.now()
    _write_last_run({
        "last_run_at": _dt.datetime.now().isoformat(),
        "mode": mode,
        "exchanges": exchanges,
        "stats": {
            "attempted": stats.attempted,
            "succeeded": stats.succeeded,
            "failed": stats.failed,
            "rows": stats.rows,
            "duration_seconds": (stats.finished - stats.started).total_seconds(),
        },
    })
    LOG.info(f"全部完成: 成功={stats.succeeded} 失败={stats.failed} 行={stats.rows}")
    return stats


def _run_cffex(ex: str, mode: str, rate: RateLimiter, ch: ClickHouseWriter,
               write_parquet: bool, write_clickhouse: bool,
               stats: RunStats) -> None:
    if ex != "CFFEX":
        return
    start = _dt.date(2010, 4, 16)   # IF 上市日
    end = _dt.date.today()
    stats.attempted += 1
    started_at = _dt.datetime.now()
    df = _safe(fetch_cffex_daily, start, end, None, rate)
    if df.empty:
        stats.failed += 1
        LOG.warning("CFFEX: 全月份下载失败, 请检查网络")
        if write_clickhouse:
            ch.log_download("CFFEX", "ALL", "cffex", 0, None, "failed",
                            "empty result", started_at, _dt.datetime.now())
        return
    last_dt = df["trade_date"].max()
    last_dt = last_dt.date() if hasattr(last_dt, "date") else last_dt
    written = 0
    for symbol, grp in df.groupby("full_symbol"):
        if write_parquet:
            save_parquet(grp, "CFFEX", symbol, sub="daily")
        if write_clickhouse:
            ch.upsert_daily(grp, "CFFEX", source="cffex")
        written += 1
    stats.succeeded += 1
    stats.rows += len(df)
    if write_clickhouse:
        ch.log_download("CFFEX", "ALL", "cffex", len(df), last_dt,
                        "success", "", started_at, _dt.datetime.now())
    LOG.info(f"CFFEX 全量: {len(df)} rows / {written} symbols, last={last_dt}")


def _run_shfe(kind: str, ex: str, mode: str, rate: RateLimiter,
              ch: ClickHouseWriter, write_parquet: bool,
              write_clickhouse: bool, stats: RunStats) -> None:
    if ex != "SHFE":
        return
    start = _dt.date(2002, 1, 1) if kind == "futures" else _dt.date(2018, 1, 1)
    end = _dt.date.today()
    stats.attempted += 1
    started_at = _dt.datetime.now()
    df = _safe(fetch_shfe_daily, start, end, kind, rate)
    if df.empty:
        stats.failed += 1
        LOG.warning(f"SHFE {kind}: 全部失败")
        if write_clickhouse:
            ch.log_download("SHFE", kind, f"shfe_{kind}", 0, None,
                            "failed", "", started_at, _dt.datetime.now())
        return
    last_dt = df["trade_date"].max()
    last_dt = last_dt.date() if hasattr(last_dt, "date") else last_dt
    written = 0
    for symbol, grp in df.groupby("full_symbol"):
        if write_parquet:
            save_parquet(grp, "SHFE", symbol, sub="daily")
        if write_clickhouse:
            ch.upsert_daily(grp, "SHFE", source=f"shfe_{kind}")
        written += 1
    stats.succeeded += 1
    stats.rows += len(df)
    if write_clickhouse:
        ch.log_download("SHFE", kind, f"shfe_{kind}", len(df), last_dt,
                        "success", "", started_at, _dt.datetime.now())
    LOG.info(f"SHFE {kind} 全量: {len(df)} rows / {written} symbols, last={last_dt}")


def _run_akshare(ex: str, mode: str, rate: RateLimiter, ch: ClickHouseWriter,
                 write_parquet: bool, write_clickhouse: bool,
                 stats: RunStats) -> None:
    if ex == "CFFEX" or ex == "SHFE":
        return  # 已被官方源覆盖
    if ex == "DCE":
        LOG.info("DCE: 主力合约已通过 Sina 通道下载, 具体合约 AKShare 不可用, 跳过")
        return
    start = _dt.date(2005, 1, 1)
    end = _dt.date.today()
    stats.attempted += 1
    started_at = _dt.datetime.now()
    df = _safe(fetch_akshare_daily, ex, start, end, rate)
    if df.empty:
        stats.failed += 1
        LOG.warning(f"{ex} AKShare: 全部失败")
        if write_clickhouse:
            ch.log_download(ex, "ALL", "akshare", 0, None,
                            "failed", "", started_at, _dt.datetime.now())
        return
    last_dt = df["trade_date"].max()
    last_dt = last_dt.date() if hasattr(last_dt, "date") else last_dt
    written = 0
    for symbol, grp in df.groupby("full_symbol"):
        if write_parquet:
            save_parquet(grp, ex, symbol, sub="daily")
        if write_clickhouse:
            ch.upsert_daily(grp, ex, source="akshare")
        written += 1
    stats.succeeded += 1
    stats.rows += len(df)
    if write_clickhouse:
        ch.log_download(ex, "ALL", "akshare", len(df), last_dt,
                        "success", "", started_at, _dt.datetime.now())
    LOG.info(f"{ex} AKShare 全量: {len(df)} rows / {written} symbols, last={last_dt}")


# ════════════════════════════════════════════════════════════════════════════
# Region 9: CLI / 鉴权测试
# ════════════════════════════════════════════════════════════════════════════


def test_auth(rate: RateLimiter) -> bool:
    """测试 CFFEX / SHFE / AKShare 是否可用 (不下载数据)."""
    ok = True
    print("─" * 60)
    print("CFFEX 鉴权测试 (2024-09 月度 ZIP)")
    print("─" * 60)
    try:
        r = http_get(CFFEX_HISTORY_URL.format(ym="202409"),
                     rate=rate, referer=CFFEX_BASE)
        print(f"  ✓ 200 OK, {len(r.content):,} bytes")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        ok = False

    print("─" * 60)
    print("SHFE 鉴权测试 (download.json)")
    print("─" * 60)
    try:
        r = http_get(SHFE_INDEX_URL, rate=rate, referer=SHFE_BASE)
        data = json.loads(r.content)
        n = len(data.get("data", {}).get("historicalData", []))
        print(f"  ✓ 200 OK, {n} historicalData entries")
    except Exception as e:
        print(f"  ✗ 失败: {e}")
        ok = False

    print("─" * 60)
    print("AKShare 测试")
    print("─" * 60)
    if not _akshare_available():
        print("  ✗ akshare 未安装 (pip install akshare)")
        ok = False
    else:
        import akshare as ak
        try:
            df = ak.futures_zh_daily_sina(symbol="IF0")
            print(f"  ✓ futures_zh_daily_sina(IF0): {len(df)} rows")
        except Exception as e:
            print(f"  ✗ IF0: {e}")
            ok = False

    print("─" * 60)
    print(f"鉴权测试结果: {'全部通过' if ok else '部分失败'}")
    print("─" * 60)
    return ok


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="中文期货/期权日线下载器 (公开数据 + 限速版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --exchanges CFFEX SHFE              # 仅下载 CFFEX + SHFE
  %(prog)s --full                               # 全量 (默认是 incremental)
  %(prog)s --main-only                          # 只下载主力连续
  %(prog)s --test-auth                          # 测试鉴权/接口可用性
  %(prog)s --rate-min 1.0 --rate-max 2.0        # 自定义限速 (秒)
        """,
    )
    p.add_argument("--exchanges", nargs="+", default=EXCHANGES,
                   choices=EXCHANGES,
                   help="目标交易所 (默认全部 6 家)")
    p.add_argument("--mode", choices=["full", "incremental"],
                   default="incremental",
                   help="full=全量重下, incremental=智能增量 (默认)")
    p.add_argument("--main-only", action="store_true",
                   help="只下载主力连续合约")
    p.add_argument("--no-main", action="store_true",
                   help="跳过主力连续合约")
    p.add_argument("--no-parquet", action="store_true",
                   help="不写本地 Parquet 备份")
    p.add_argument("--no-clickhouse", action="store_true",
                   help="不写 ClickHouse")
    p.add_argument("--output-dir", type=str, default=None,
                   help="输出目录 (默认 data/chinese_futures)")
    p.add_argument("--rate-min", type=float, default=3.0,
                   help="限速下限秒/请求 (默认 3)")
    p.add_argument("--rate-max", type=float, default=5.0,
                   help="限速上限秒/请求 (默认 5)")
    p.add_argument("--dry-run", action="store_true",
                   help="只看会做什么, 不实际下载")
    p.add_argument("--test-auth", action="store_true",
                   help="测试鉴权/接口可用性, 不下载数据")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # 输出目录覆写
    global DATA_ROOT, META_DIR, LOG_DIR, FAILED_FILE
    global LAST_RUN_FILE, PRIORITY_FILE, LOG_FILE
    if args.output_dir:
        DATA_ROOT = Path(args.output_dir)
        META_DIR = DATA_ROOT / "_meta"
        LOG_DIR = DATA_ROOT / "_log"
        FAILED_FILE = META_DIR / "_failed_months.json"
        LAST_RUN_FILE = META_DIR / "_last_run.json"
        PRIORITY_FILE = META_DIR / "_source_priority.json"
        LOG_FILE = LOG_DIR / "download.log"

    # 自定义限速
    if args.rate_min > args.rate_max:
        args.rate_max = args.rate_min
    RATE_LIMITS["www.cffex.com.cn"] = (args.rate_min, args.rate_max)
    RATE_LIMITS["www.shfe.com.cn"] = (args.rate_min, args.rate_max)

    if args.test_auth:
        rate = RateLimiter()
        ok = test_auth(rate)
        return 0 if ok else 1

    LOG.info(f"开始运行: exchanges={args.exchanges} mode={args.mode} "
             f"main={'skip' if args.no_main else 'on'}")
    LOG.info(f"限速: {args.rate_min}-{args.rate_max} 秒/请求")

    if args.dry_run:
        LOG.info("Dry-run 模式: 仅打印计划, 不实际下载")
        for ex in args.exchanges:
            mains = AKSHARE_MAIN_SYMBOLS.get(ex, [])
            LOG.info(f"  {ex}: main={mains}, sources={SOURCE_PRIORITY.get(ex, [])}")
        return 0

    stats = run_one(
        exchanges=args.exchanges,
        mode=args.mode,
        write_parquet=not args.no_parquet,
        write_clickhouse=not args.no_clickhouse,
        include_main=not args.no_main,
        skip_exchange_scan=args.main_only,
    )
    return 0 if stats.failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())