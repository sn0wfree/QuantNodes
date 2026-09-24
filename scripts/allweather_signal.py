#!/usr/bin/env python3
# coding=utf-8
"""全天候 v3 — 生成最新目标权重信号并落盘 (§生产化).

用法:
    python3 scripts/allweather_signal.py                    # 最新
    python3 scripts/allweather_signal.py --as-of 2026-06-30 # 指定截断日
    python3 scripts/allweather_signal.py --no-log           # 只打印不落盘

输出:
    data/allweather/signals/latest_target.csv   (asset, weight)
    data/allweather/signals/signal_log.parquet  (as_of + 各资产权重, 增量)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from QuantNodes.strategy.allweather.production import (
    CARRY_WEIGHT, latest_target, run_production, summarize)

OUT_DIR = Path("data/allweather/signals")


def main():
    ap = argparse.ArgumentParser(description="全天候 v3 目标权重信号")
    ap.add_argument("--as-of", default=None, help="截断日期 YYYY-MM-DD")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--no-log", action="store_true", help="不写信号日志")
    args = ap.parse_args()

    res = run_production(as_of=args.as_of)
    tgt = latest_target(res)
    as_of = res.nav.index[-1]
    m = summarize(res.nav)
    m19 = summarize(res.nav.loc['2019-01-01':])

    print(f"全天候 v3 | as_of {as_of.date()} | Carry {CARRY_WEIGHT:.0%} + D")
    print(f"  净值 {res.nav.iloc[-1]:.4f}")
    print(f"  CAGR {m19['CAGR']:.2%} | Sharpe {m19['Sharpe']:.2f} | "
          f"Vol {m19['Vol']:.2%} | MaxDD {m19['MaxDD']:.2%}")
    print(f"  当前回撤 {m['CurrDD']:.2%} | 组合 gross "
          f"{res.weights.iloc[-1].abs().sum():.3f} net "
          f"{res.weights.iloc[-1].sum():.3f}")
    print(f"\n最新目标权重 ({len(tgt)} 项):")
    for k, v in tgt.items():
        print(f"  {k:<12}{v:+.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tgt.rename("weight").to_frame().reset_index(names="asset").to_csv(
        out / "latest_target.csv", index=False)
    print(f"\n快照: {out / 'latest_target.csv'}")

    if not args.no_log:
        log_path = out / "signal_log.parquet"
        row = tgt.to_frame().T
        row.insert(0, "as_of", as_of)
        row = row.reset_index(drop=True)
        if log_path.exists():
            old = pd.read_parquet(log_path)
            old = old[old["as_of"] != as_of]
            row = pd.concat([old, row], ignore_index=True)
        row.to_parquet(log_path, index=False)
        print(f"信号日志: {log_path} ({len(row)} 期)")


if __name__ == "__main__":
    main()
