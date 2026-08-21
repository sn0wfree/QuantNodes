#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/0_data_probe_v3.py — 用真实接口名探查基本面数据."""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import akshare as ak

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "cmf"
OUT.mkdir(parents=True, exist_ok=True)


def try_fetch(name: str, args=None, timeout_tag: str = ""):
    if args is None:
        args = {}
    fn = getattr(ak, name, None)
    if fn is None or not callable(fn):
        return {"factor": name, "status": "no-fn", "args": str(args), "n": 0, "start": None, "end": None}
    try:
        df = fn(**args)
        if df is None or (isinstance(df, pd.DataFrame) and len(df) == 0):
            return {"factor": name, "status": "empty", "args": str(args), "n": 0, "start": None, "end": None}
        if isinstance(df, pd.DataFrame):
            date_col = next(
                (c for c in df.columns if any(k in str(c).lower() for k in ["date", "time", "日期", "时间"])),
                None,
            )
            start = end = None
            if date_col:
                d = pd.to_datetime(df[date_col], errors="coerce").dropna()
                if len(d):
                    start = str(d.min().date())
                    end = str(d.max().date())
            return {"factor": name, "status": "OK", "args": str(args), "n": len(df), "start": start, "end": end}
        return {"factor": name, "status": f"non-df {type(df).__name__}", "args": str(args), "n": 0, "start": None, "end": None}
    except Exception as e:
        return {"factor": name, "status": f"FAIL:{type(e).__name__}", "args": str(args), "n": 0, "start": None, "end": None, "err": str(e)[:100]}


def main():
    print("=" * 70)
    print("CMF Phase 0 — akshare 真实接口可达性探查")
    print("=" * 70)

    probes = [
        # ── 铜库存 ──────────────────────────────────────────────────────
        ("futures_inventory_em", {"symbol": "cu"}, "铜.SHFE库存(cu)"),
        ("futures_inventory_em", {"symbol": "CU"}, "铜.SHFE库存(CU)"),
        ("futures_inventory_99", {"symbol": "cu"}, "铜.DCE库存(cu)"),
        ("futures_inventory_99", {"symbol": "CU"}, "铜.DCE库存(CU)"),
        ("futures_comex_inventory", {"symbol": "铜"}, "铜.COMEX库存"),
        ("futures_global_spot_em", {}, "全球商品实时价"),
        ("futures_spot_sys", {"symbol": "铜"}, "铜.现货"),

        # ── 原油 (INE-SC主力) ─────────────────────────────────────────
        ("futures_main_sina", {"symbol": "SC0"}, "原油.SC主力"),
        ("futures_main_sina", {"symbol": "SC"}, "原油.SC主连"),
        ("energy_oil_hist", {}, "原油.EIA日度"),
        ("energy_oil_detail", {"date": "20240101"}, "原油.EIA明细"),

        # ── 豆粕 ──────────────────────────────────────────────────────
        ("futures_inventory_em", {"symbol": "m"}, "豆粕.DCE库存(m)"),
        ("futures_inventory_em", {"symbol": "M"}, "豆粕.DCE库存(M)"),
        ("futures_inventory_99", {"symbol": "m"}, "豆粕.99库存(m)"),
        ("futures_main_sina", {"symbol": "M0"}, "豆粕.M主力"),

        # ── 宏观 ──────────────────────────────────────────────────────
        ("macro_usa_adp_employment", {}, "宏观.ADP就业"),
        ("fx_spot_quote", {}, "宏观.FX现汇"),
        ("fx_c_swap_cm", {}, "宏观.FX掉期"),

        # ── 国内宏观 ──────────────────────────────────────────────────
    ]

    print(f"\n测试 {len(probes)} 个真实接口:\n")
    results = []
    for fn_name, args, label in probes:
        r = try_fetch(fn_name, args)
        r["label"] = label
        results.append(r)
        marker = "✓" if r["status"] == "OK" else "✗"
        print(f"  {marker} {label:35s} {fn_name}({args}) → {r['status']:20s} n={r['n']:>5} [{r['start']}~{r['end']}]")

    df = pd.DataFrame(results)
    df.to_csv(OUT / "data_availability_v3.csv", index=False)
    ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\n汇总: {ok}/{len(results)} 可达 → {OUT / 'data_availability_v3.csv'}")


if __name__ == "__main__":
    main()