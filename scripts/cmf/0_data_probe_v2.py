#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/0_data_probe_v2.py — 用 getattr + 关键字匹配动态发现可用接口."""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import akshare as ak

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "cmf"
OUT.mkdir(parents=True, exist_ok=True)


def list_methods(obj, prefix: str) -> list[str]:
    return [n for n in dir(obj) if n.startswith(prefix) and callable(getattr(obj, n, None))]


def try_call(name: str, timeout_s: float = 10.0):
    import signal

    fn = getattr(ak, name, None)
    if fn is None or not callable(fn):
        return {"name": name, "status": "no-fn", "sig": ""}
    try:
        import inspect

        sig = str(inspect.signature(fn))[:80]
        return {"name": name, "status": "exists", "sig": sig}
    except Exception:
        return {"name": name, "status": "exists", "sig": "?"}


def main():
    print("=" * 70)
    print("akshare 接口发现 (v2)")
    print("=" * 70)

    found = []
    for prefix in [
        "futures_inventory",
        "futures_lme",
        "futures_comex",
        "futures_yangshan",
        "futures_yang",
        "futures_foreign",
        "futures_spot",
        "futures_basis",
        "futures_premium",
        "energy_",
        "macro_usa",
        "fx_",
        "futures_em_",
        "futures_czce_",
        "futures_dce_",
        "futures_shfe_",
        "futures_main_",
        "futures_pig",
        "futures_profit",
        "futures_market_",
        "pig_",
        "feed_",
        "futures_global_",
    ]:
        names = list_methods(ak, prefix)
        if names:
            print(f"\n[{prefix}*]  {len(names)} 个")
            for n in names[:20]:
                info = try_call(n)
                found.append({"prefix": prefix, **info})
                if len(names) <= 8 or len([f for f in found if f["prefix"] == prefix]) <= 3:
                    print(f"  {n}{info['sig']}")

    df = pd.DataFrame(found)
    df.to_csv(OUT / "akshare_api_inventory.csv", index=False)
    print(f"\n总计发现: {len(found)} 个接口")


if __name__ == "__main__":
    main()