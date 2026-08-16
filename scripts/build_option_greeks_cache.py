#!/usr/bin/env python3
"""并行预计算期权 Greeks → parquet 缓存 (docs/103 加速回测用).

按 (product, symbol-chunk) 拆任务, multiprocessing.Pool 并行反解.
输出: data/cache/option_greeks.parquet
列: product / symbol / trade_date / strike_price / option_type / underlying
    / expiry_date / spot / delta / gamma / vega / theta / iv / valid

存储规模 (~2.4M option-days × ~14 列) 约 500MB-1GB, 一次性构建.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from QuantNodes.strategy.cta.data_loader import CtaDataLoader

PRODUCTS = ["RB", "I", "AG", "AU", "CU", "M", "TA", "SR", "RU", "PP"]
START = "2023-01-01"
END = "2025-12-31"
OUT = Path("/home/ll/Public/QuantNodes/data/cache/option_greeks.parquet")
CHUNK_SYMS = 25          # 每任务处理的 symbol 数 (负载均衡)
COLUMNS = [
    "product", "symbol", "trade_date", "strike_price", "option_type",
    "underlying", "expiry_date", "spot", "delta", "gamma",
    "vega", "theta", "iv", "valid",
]


def _compute_chunk(task: tuple) -> pd.DataFrame:
    """worker: 计算单个 symbol 块的全部 greeks (top-level, pickle-safe)."""
    from QuantNodes.strategy.cta.options.greeks import compute_greeks_for_option
    from QuantNodes.strategy.cta.options.underlying_map import UnderlyingMap

    product, symbols, start, end = task
    loader = CtaDataLoader()
    um = UnderlyingMap(loader)

    df = loader.load_v_all_options(products=[product], start=start, end=end)
    if df is None or df.empty:
        return pd.DataFrame(columns=COLUMNS)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["symbol"] = df["symbol"].str.upper()
    sub = df[df["symbol"].isin(set(symbols))].copy()
    if sub.empty:
        return pd.DataFrame(columns=COLUMNS)

    # 到期日映射 (数据驱动: 链内末次出现日期)
    last = sub.groupby("symbol")["trade_date"].max()
    sub["expiry_date"] = sub["symbol"].map(last)

    # underlying 价格
    us = sub["underlying"].dropna().unique().tolist()
    for u in us:
        ser = um.get(u)
        if not ser.empty:
            sub.loc[sub["underlying"] == u, "spot"] = (
                sub.loc[sub["underlying"] == u, "trade_date"].map(ser)
            )
    sub = sub[sub["spot"].notna() & (sub["spot"] > 0)].copy()
    if sub.empty:
        return pd.DataFrame(columns=COLUMNS)

    sub["_exp"] = pd.to_datetime(sub["expiry_date"], errors="coerce")
    ttm = (sub["_exp"] - sub["trade_date"]).dt.days / 365.0
    sub["_ttm"] = ttm
    work = sub[(sub["_ttm"] > 0) & (sub["close"] > 0)].copy()
    if work.empty:
        return pd.DataFrame(columns=COLUMNS)

    rows = []
    for _, r in work.iterrows():
        g = compute_greeks_for_option(
            symbol=r["symbol"], option_type=r["option_type"],
            strike_price=r["strike_price"], underlying_price=r["spot"],
            market_price=r["close"], time_to_maturity=r["_ttm"],
        )
        rows.append({
            "product": product, "symbol": r["symbol"],
            "trade_date": r["trade_date"], "strike_price": r["strike_price"],
            "option_type": r["option_type"], "underlying": r["underlying"],
            "expiry_date": r["expiry_date"], "spot": float(r["spot"]),
            "delta": g.delta, "gamma": g.gamma, "vega": g.vega,
            "theta": g.theta, "iv": g.iv, "valid": g.valid,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def _all_tasks():
    loader = CtaDataLoader()
    tasks = []
    for product in PRODUCTS:
        df = loader.load_v_all_options(products=[product], start=START, end=END)
        if df is None or df.empty:
            continue
        syms = sorted(df["symbol"].str.upper().unique().tolist())
        for i in range(0, len(syms), CHUNK_SYMS):
            tasks.append((product, syms[i:i + CHUNK_SYMS], START, END))
    return tasks


def main():
    nproc = max(1, min(len(PRODUCTS) * 8, 16))
    print(f"SCHEMA: {COLUMNS}")
    t0 = time.time()
    tasks = _all_tasks()
    print(f"tasks={len(tasks)}, cores={nproc}")

    frames = []
    with ProcessPoolExecutor(max_workers=nproc) as ex:
        futs = {ex.submit(_compute_chunk, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futs):
            product = futs[fut]
            try:
                df = fut.result()
            except Exception as e:
                print(f"  [{product}] task FAILED: {e}")
                continue
            frames.append(df)
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} chunks, {time.time()-t0:.0f}s", flush=True)

    if not frames:
        print("NO DATA")
        sys.exit(1)
    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    valid_n = int(out["valid"].sum())
    print(f"\ndone in {time.time()-t0:.0f}s")
    print(f"rows={len(out)}, valid={valid_n} ({valid_n/len(out)*100:.1f}%)")
    print(f"products={out['product'].nunique()}, symbols={out['symbol'].nunique()}")
    print(f"saved -> {OUT} ({OUT.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
