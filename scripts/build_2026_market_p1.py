# coding: utf-8
"""build_2026_market_p1.py — CH tick → 临时库原始表 (期货+期权), 2025-10 起.

复用 sync_ch_to_duckdb 的聚合口径 (SessionType/夜盘过滤/argMin/argMax),
新增: 期权合约识别 + option_type/strike_price/underlying 解析.
输出: export/_tmp_2026.duckdb 的 {ex}_{p}_futures_daily / {ex}_{p}_options_daily

Usage:
    python scripts/build_2026_market_p1.py            # 全量 92 产品
    python scripts/build_2026_market_p1.py --resume   # 断点续传
    python scripts/build_2026_market_p1.py --products AU RB
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import clickhouse_connect
import duckdb
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DB = Path("~/Public/DataCache/futures_options_daily.duckdb").expanduser()
TMP_DB = PROJECT_ROOT / "export" / "_tmp_2026.duckdb"
CHECKPOINT = PROJECT_ROOT / "export" / "_tmp_2026_checkpoint.json"
START_DATE = "2018-01-01"  # 2018 起始, 2026-09-04 截止 (CH tick 起始 2018-01)

CH_CONFIG = {"host": "localhost", "port": 8123, "username": "data", "password": "123456"}

OPT_PATTERNS = [
    re.compile(r"^([a-z]+)(\d{4})([CP])(\d+(?:\.\d+)?)$"),        # SHFE/INE: au2604C816
    re.compile(r"^([a-z]+)(\d{4})-([CP])-(\d+(?:\.\d+)?)$"),      # DCE/GFEX: m2607-P-2700
    re.compile(r"^([A-Z]{1,2})(\d{3})([CP])(\d+(?:\.\d+)?)$"),    # CZCE: SR701P5500
    re.compile(r"^([A-Z]{2,4})(\d{4})-([CP])-(\d+(?:\.\d+)?)$"),  # CFFEX: IO2610-C-4100
]

# 标准期货合约: 品种(1-4字母) + 年月(3-4位). 剔除套利(SP a&b / MSC / MS-C) /
# EFP(xxefp) / TAS(sc2604TAS) / 期转现(l2603F) 等非标准合约
FUT_SYMBOL_RE = re.compile(r"^[A-Za-z]{1,4}\d{3,4}$")

FUT_COLS = ["exchange", "product", "symbol", "trade_date", "open", "high", "low",
            "close", "volume", "turnover", "open_interest",
            "pre_settlement", "pre_close", "pre_open_interest"]
OPT_COLS = ["exchange", "product", "symbol", "underlying", "option_type",
            "strike_price", "trade_date", "open", "high", "low", "close",
            "volume", "turnover", "open_interest",
            "pre_settlement", "pre_close", "pre_open_interest", "src_file"]

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
LOG = logging.getLogger(__name__)


def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        try:
            return json.loads(CHECKPOINT.read_text())
        except Exception:
            return {"completed": [], "failed": [], "stats": {}}
    return {"completed": [], "failed": [], "stats": {}}


def save_checkpoint(ckpt: dict) -> None:
    CHECKPOINT.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2))


def parse_option(symbol: str) -> tuple[str, str, float] | None:
    for pat in OPT_PATTERNS:
        m = pat.match(symbol)
        if m:
            underlying = f"{m.group(1)}{m.group(2)}"
            return underlying, m.group(3), float(m.group(4))
    return None


def fetch_product(ch: clickhouse_connect.Client, product: str) -> pd.DataFrame | None:
    """CH 聚合单产品全部合约 (期货+期权混合), 2025-10 起."""
    sql = f"""
SELECT
    InstrumentID AS symbol,
    TradingDay AS trade_date,
    argMin(LastPrice, TradingDateTimeMS) AS open,
    max(LastPrice) AS high,
    min(LastPrice) AS low,
    argMax(LastPrice, TradingDateTimeMS) AS close,
    max(Volume) - min(Volume) AS volume,
    max(Turnover) - min(Turnover) AS turnover,
    argMax(OpenInterest, TradingDateTimeMS) AS open_interest,
    argMin(PreSettlementPrice, TradingDateTimeMS) AS pre_settlement,
    argMin(PreClosePrice, TradingDateTimeMS) AS pre_close,
    argMin(PreOpenInterest, TradingDateTimeMS) AS pre_open_interest
FROM tick_quote.{product}
WHERE SessionType IN (2, 3)
  AND Exchange NOT LIKE '%\\_night%'
  AND TradingDay >= '{START_DATE}'
GROUP BY InstrumentID, TradingDay
SETTINGS max_execution_time=600
"""
    result = ch.query(sql).result_rows
    if not result:
        return None
    cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume",
            "turnover", "open_interest", "pre_settlement", "pre_close", "pre_open_interest"]
    return pd.DataFrame(result, columns=cols)


def write_product(df: pd.DataFrame, exchange: str, product: str) -> tuple[int, int]:
    """分类期货/期权并写入临时库. 返回 (期货行数, 期權行数)."""
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    num_cols = ["open", "high", "low", "close", "turnover", "open_interest",
                "pre_settlement", "pre_close", "pre_open_interest"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    opt_mask = df["symbol"].apply(lambda s: parse_option(str(s)) is not None)
    fut = df[~opt_mask].copy()
    fut = fut[fut["symbol"].apply(lambda s: FUT_SYMBOL_RE.match(str(s)) is not None)].copy()
    opt = df[opt_mask].copy()

    fut["exchange"] = exchange
    fut["product"] = product
    fut["volume"] = pd.to_numeric(fut["volume"], errors="coerce").fillna(0).astype("int64")
    fut = fut[FUT_COLS].sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    parsed = opt["symbol"].apply(parse_option)
    opt["underlying"] = [p[0] for p in parsed]
    opt["option_type"] = [p[1] for p in parsed]
    opt["strike_price"] = [p[2] for p in parsed]
    opt["exchange"] = exchange
    opt["product"] = product
    opt["src_file"] = "ch_tick_quote"
    opt["volume"] = pd.to_numeric(opt["volume"], errors="coerce").fillna(0).astype("int64")
    opt = opt[OPT_COLS].sort_values(["trade_date", "symbol"]).reset_index(drop=True)

    con = duckdb.connect(str(TMP_DB))
    try:
        ftab = f"{exchange.lower()}_{product.lower()}_futures_daily"
        otab = f"{exchange.lower()}_{product.lower()}_options_daily"
        con.execute(f"DROP TABLE IF EXISTS {ftab}")
        con.register("fut_v", fut)
        con.execute(f"CREATE TABLE {ftab} AS SELECT * FROM fut_v")
        con.unregister("fut_v")
        con.execute(f"DROP TABLE IF EXISTS {otab}")
        con.register("opt_v", opt)
        con.execute(f"CREATE TABLE {otab} AS SELECT * FROM opt_v")
        con.unregister("opt_v")
    finally:
        con.close()
    return len(fut), len(opt)


def run(products: list[str] | None, resume: bool) -> None:
    specs = duckdb.connect(str(SRC_DB), read_only=True).execute(
        "SELECT product, exchange FROM contract_specs ORDER BY exchange, product").fetchall()
    product_map = {p.upper(): e for p, e in specs}

    if products:
        target = [p.upper() for p in products if p.upper() in product_map]
    else:
        target = sorted(product_map)

    ckpt = load_checkpoint() if resume else {"completed": [], "failed": [], "stats": {}}
    target = [p for p in target if p not in set(ckpt["completed"])]

    ch = clickhouse_connect.get_client(**CH_CONFIG)
    pbar = tqdm(target, desc="P1 sync", unit="prod", ncols=100)
    for prod in pbar:
        exch = product_map[prod]
        pbar.set_postfix_str(f"{exch}.{prod}", refresh=False)
        t0 = time.time()
        try:
            df = fetch_product(ch, prod)
            nf = no = 0
            if df is not None and len(df):
                nf, no = write_product(df, exch, prod)
            ckpt["completed"].append(prod)
            ckpt["stats"][prod] = {"fut": nf, "opt": no, "sec": round(time.time() - t0, 1)}
            pbar.set_postfix_str(f"{exch}.{prod} ✓ f={nf:,} o={no:,}", refresh=False)
        except Exception as e:
            ckpt["failed"].append(prod)
            ckpt["stats"][prod] = {"error": str(e)[:200]}
            pbar.set_postfix_str(f"{exch}.{prod} ✗", refresh=False)
            LOG.error(f"{exch}.{prod}: {e}")
        save_checkpoint(ckpt)
    pbar.close()
    LOG.info(f"完成 {len(ckpt['completed'])} 产品, 失败 {len(ckpt['failed'])}: {ckpt['failed']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    TMP_DB.parent.mkdir(exist_ok=True)
    run(args.products, args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
