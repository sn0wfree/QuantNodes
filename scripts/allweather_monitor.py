#!/usr/bin/env python3
# coding=utf-8
"""全天候 v3 — 数据新鲜度 + 组合健康监控 (§生产化).

用法:
    python3 scripts/allweather_monitor.py
    python3 scripts/allweather_monitor.py --as-of 2026-06-30

告警阈值:
    - 当前回撤 < -8%          (目标 MaxDD -10%)
    - 任一数据源落后 > 5 交易日
    - carry sleeve 近 1 年 Sharpe < 0
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from QuantNodes.strategy.allweather.production import (
    DATA_DIR, V7_PATH, run_production, summarize)

DD_WARN = -0.08
LAG_WARN = 5
CARRY_SHARPE_WARN = 0.0

SOURCES = [
    ("panel_v5", DATA_DIR / "panel_v5.parquet"),
    ("carry", DATA_DIR / "carry_daily.parquet"),
    ("volume_oi", DATA_DIR / "volume_oi_cache.parquet"),
    ("v7_index", V7_PATH),
]


def _asof(path: Path):
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return pd.to_datetime(df.index).max()


def _sharpe(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 20 or x.std() == 0:
        return np.nan
    return x.mean() * 252 / (x.std() * np.sqrt(252))


def main():
    ap = argparse.ArgumentParser(description="全天候 v3 监控")
    ap.add_argument("--as-of", default=None, help="截断日期 YYYY-MM-DD")
    args = ap.parse_args()
    today = dt.date.today()
    alerts: list[str] = []

    print(f"=== 全天候 v3 监控 | {today} ===")

    print("\n数据新鲜度:")
    for name, path in SOURCES:
        a = _asof(path)
        if a is None:
            print(f"  {name:<11}缺失 ❌")
            alerts.append(f"{name} 数据缺失")
            continue
        lag = int(np.busday_count(a.date(), today))
        flag = "✅" if lag <= LAG_WARN else "⚠️"
        print(f"  {name:<11}{a.date()}  (落后 {lag} 交易日) {flag}")
        if lag > LAG_WARN:
            alerts.append(f"{name} 落后 {lag} 交易日")

    res = run_production(as_of=args.as_of)
    m19 = summarize(res.nav.loc['2019-01-01':])
    m = summarize(res.nav)
    print(f"\n组合指标 (2019+, as_of {m['end']}):")
    print(f"  CAGR {m19['CAGR']:.2%} | Sharpe {m19['Sharpe']:.2f} | "
          f"Vol {m19['Vol']:.2%} | MaxDD {m19['MaxDD']:.2%}")
    dd_flag = "✅" if m["CurrDD"] >= DD_WARN else "⚠️"
    print(f"  当前回撤 {m['CurrDD']:.2%} {dd_flag} "
          f"(告警线 {DD_WARN:.0%})")
    if m["CurrDD"] < DD_WARN:
        alerts.append(f"当前回撤 {m['CurrDD']:.2%} 低于告警线 {DD_WARN:.0%}")

    print("\nSleeve 健康:")
    sh_d19 = _sharpe(res.d_nav.pct_change(fill_method=None).loc['2019-01-01':])
    cs = res.carry_ret
    print(f"  D sleeve Sharpe (2019+)      {sh_d19:.2f}")
    for lab, lo in [("2019+", "2019-01-01"), ("2022+", "2022-01-01")]:
        print(f"  carry Sharpe ({lab:<5})         {_sharpe(cs.loc[lo:]):.2f}")
    sh_carry_1y = _sharpe(cs.iloc[-252:])
    print(f"  carry Sharpe (近1年)          {sh_carry_1y:.2f}")
    if not np.isnan(sh_carry_1y) and sh_carry_1y < CARRY_SHARPE_WARN:
        alerts.append(f"carry 近1年 Sharpe {sh_carry_1y:.2f} < 0")

    print("\n" + ("✅ 无告警" if not alerts else "⚠️ 告警:"))
    for a in alerts:
        print(f"  - {a}")
    return 1 if alerts else 0


if __name__ == "__main__":
    raise SystemExit(main())
