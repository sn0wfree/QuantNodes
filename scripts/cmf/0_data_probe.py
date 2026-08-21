#!/usr/bin/env python3
# coding=utf-8
"""scripts/cmf/0_data_probe.py — 基本面数据可达性探查.

对每个报告§2.2/§4需要的基本面因子，逐一测试akshare是否能拉到。
输出 reports/cmf/data_availability.md，标记可达/不可达/数据起点。
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "reports" / "cmf"
OUT.mkdir(parents=True, exist_ok=True)

results: list[dict] = []


def try_fetch(name: str, fn):
    try:
        df = fn()
        if df is None or len(df) == 0:
            return {"factor": name, "status": "empty", "n": 0, "start": None, "end": None}
        if isinstance(df, pd.DataFrame):
            date_col = next((c for c in ["date", "trade_date", "DATE", "时间", "日期"] if c in df.columns), None)
            start = end = None
            if date_col:
                d = pd.to_datetime(df[date_col], errors="coerce").dropna()
                if len(d):
                    start = str(d.min().date())
                    end = str(d.max().date())
            return {"factor": name, "status": "OK", "n": len(df), "start": start, "end": end}
        return {"factor": name, "status": f"non-df ({type(df).__name__})", "n": 0, "start": None, "end": None}
    except Exception as e:
        return {"factor": name, "status": f"FAIL: {type(e).__name__}", "n": 0, "start": None, "end": None, "err": str(e)[:120]}


def main():
    import akshare as ak

    print("=" * 70)
    print("CMF Phase 0 — 基本面数据可达性探查 (akshare 1.18.49)")
    print("=" * 70)

    probes = [
        # ── 铜 ─────────────────────────────────────────────────────────
        ("铜.SHFE库存",
         lambda: ak.futures_inventory_em(symbol="铜", exchange="shfe")
         if hasattr(ak, "futures_inventory_em") else None),
        ("铜.SHFE库存_v2",
         lambda: ak.futures_inventory_99(symbol="cu", exchange="SHFE")
         if hasattr(ak, "futures_inventory_99") else None),
        ("铜.LME库存",
         lambda: ak.futures_lme_stock(symbol="铜")
         if hasattr(ak, "futures_lme_stock") else None),
        ("铜.COMEX库存",
         lambda: ak.futures_comex_stock(symbol="HG")
         if hasattr(ak, "futures_comex_stock") else None),
        ("铜.洋山溢价",
         lambda: ak.futures_yangshan_premium(symbol="铜")
         if hasattr(ak, "futures_yangshan_premium") else None),

        # ── 原油 ────────────────────────────────────────────────────────
        ("原油.EIA美国商业库存_周",
         lambda: ak.energy_oil_eia_api(category="库存", name="美国")
         if hasattr(ak, "energy_oil_eia_api") else None),
        ("原油.EIA_总入口",
         lambda: ak.energy_oil_eia()),
        ("原油.炼厂开工率",
         lambda: ak.energy_oil_refinery_rate_us()
         if hasattr(ak, "energy_oil_refinery_rate_us") else None),
        ("原油.钻机数",
         lambda: ak.energy_oil_drilling_rig_us()
         if hasattr(ak, "energy_oil_drilling_rig_us") else None),
        ("原油.INE-SC主力合约",
         lambda: ak.futures_main_sina(symbol="SC0")
         if hasattr(ak, "futures_main_sina") else None),

        # ── 豆粕 ────────────────────────────────────────────────────────
        ("豆粕.国内库存",
         lambda: ak.futures_inventory_em(symbol="豆粕", exchange="dce")
         if hasattr(ak, "futures_inventory_em") else None),
        ("豆粕.压榨利润_华东",
         lambda: ak.futures_profit_ncp(symbol="m", exchange="dce")
         if hasattr(ak, "futures_profit_ncp") else None),
        ("豆粕.压榨利润_dce_m",
         lambda: ak.futures_profit_dce(symbol="m")
         if hasattr(ak, "futures_profit_dce") else None),
        ("豆粕.生猪养殖利润_自繁",
         lambda: ak.pig_price_national()
         if hasattr(ak, "pig_price_national") else None),
        ("豆粕.生猪养殖利润_总",
         lambda: ak.futures_pig_profit()
         if hasattr(ak, "futures_pig_profit") else None),

        # ── 宏观 ────────────────────────────────────────────────────────
        ("宏观.DXY美元指数",
         lambda: ak.fx_spot_us_dxy()),
        ("宏观.美国10Y TIPS_实际利率",
         lambda: ak.macro_usa_tips()),
        ("宏观.美国CPI",
         lambda: ak.macro_usa_cpi_monthly()),
        ("宏观.OVX_原油波动率",
         lambda: ak.futures_foreign_commodity_subscribe_exchange_symbol()),
    ]

    print(f"\n将测试 {len(probes)} 个数据接口...\n")
    for name, fn in probes:
        r = try_fetch(name, fn)
        results.append(r)
        marker = "✓" if r["status"] == "OK" else "✗"
        print(f"  {marker} {name:40s}  status={r['status']:30s}  n={r['n']:>6}  [{r['start']}~{r['end']}]")

    df = pd.DataFrame(results)
    df.to_csv(OUT / "data_availability.csv", index=False)

    ok_count = sum(1 for r in results if r["status"] == "OK")
    print(f"\n汇总: {ok_count}/{len(results)} 接口可达")
    print(f"明细: {OUT / 'data_availability.csv'}")


if __name__ == "__main__":
    main()