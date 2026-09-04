# coding=utf-8
"""cta_regime_filter_test.py — G2/G3 实验: regime filter (震荡市降仓).

G2: ER版     — regime_filter_enabled=True, type="er" (ER<0.4 → 仓位×0.4, 现有代码)
G3: ADX滞回版 — type="adx_hysteresis" (ADX>25满仓 / <20降仓 / 中间维持, 新建)

基线: cap=0.0 (weight_cap_test 对账通过, nav 已落盘 cap0.0.parquet)

决策规则 (预定义): 通过 = 2024 Sharpe 改善>0.3 AND IS+2022-23 恶化<10% AND MaxDD 不恶化
口径: 分段重定基 CAGR/Vol (与 weight_cap_test 一致)

输出: reports/cta/regime_filter_test.csv
      reports/cta/regime_filter_navs/{er,adx}.parquet
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")

from QuantNodes.strategy.cta.strategies.base import DEFAULT_PRODUCTS  # noqa: E402
from QuantNodes.strategy.cta.strategies.combo import (  # noqa: E402
    ComboConfig, FactorModuleComboStrategy,
)

REPORT_DIR = Path("reports/cta")
BASE_NAV = REPORT_DIR / "weight_cap_navs" / "cap0.0.parquet"

FINAL_LUMP_CFG = dict(
    aggregation="tvpr", module_weighting="inv_vol",
    continuous=True, cont_z_window=252, cont_z_clip=3.0,
    momentum_filter_window=60, momentum_filter_coef=0.0,
    target_vol=1.5, max_leverage=3.0, leverage_hard_cap=3.0,
    margin_call=True, position_ema_alpha=0.15,
    roll_cost_bps=10.0, roll_model="lump",
)
_cost = json.loads((REPORT_DIR / "cost_model.json").read_text())
CALIBRATED_COSTS = {
    "commission_rate": {p: v["commission_bps"] / 10000 for p, v in _cost.items()},
    "slippage_bps": {p: v["slippage_bps"] for p, v in _cost.items()},
}

VARIANTS = {
    "ER版(ER<0.4→x0.4)": {"regime_filter_enabled": True, "regime_filter_type": "er"},
    "ADX滞回(25/20→x0.4)": {"regime_filter_enabled": True,
                             "regime_filter_type": "adx_hysteresis",
                             "regime_adx_enter": 25.0, "regime_adx_exit": 20.0},
}
SEGMENTS = [("IS 2019-2021", "2019-01-01", "2022-01-01"),
            ("2022-2023", "2022-01-01", "2024-01-01"),
            ("2024", "2024-01-01", "2025-01-01"),
            ("2025", "2025-01-01", "2026-01-01"),
            ("全样本", None, None)]


def _run_one(label: str, extra: dict) -> tuple[str, pd.Series]:
    cfg_kw = dict(FINAL_LUMP_CFG)
    cfg_kw.update(CALIBRATED_COSTS)
    cfg_kw.update(extra)
    cfg = ComboConfig(**{f.name: cfg_kw[f.name] for f in dc_fields(ComboConfig)
                         if f.name in cfg_kw})
    s = FactorModuleComboStrategy(config=cfg, products=list(DEFAULT_PRODUCTS))
    r = s.run()
    return label, r.nav


def _seg_metrics(nav: pd.Series, lo: str | None, hi: str | None) -> dict:
    seg = nav
    if lo:
        seg = seg[seg.index >= lo]
    if hi:
        seg = seg[seg.index < hi]
    seg = seg / seg.iloc[0]
    rets = seg.pct_change().dropna()
    if len(rets) < 50:
        return {"sharpe": np.nan, "ann_ret": np.nan, "max_dd": np.nan}
    years = (seg.index[-1] - seg.index[0]).days / 365.25
    ann = seg.iloc[-1] ** (1 / years) - 1
    vol = rets.std() * np.sqrt(252)
    dd = (seg / seg.cummax() - 1).min()
    return {"sharpe": ann / vol if vol > 0 else np.nan,
            "ann_ret": ann, "max_dd": dd}


def main() -> None:
    print(f"G2/G3 实验: {list(VARIANTS)}  (2 workers)")
    results = {}
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(_run_one, k, v): k for k, v in VARIANTS.items()}
        for fut in as_completed(futs):
            label, nav = fut.result()
            results[label] = nav
            m = _seg_metrics(nav, None, None)
            print(f"  done {label}: Sharpe {m['sharpe']:.3f} "
                  f"AnnRet {m['ann_ret']:+.1%} MaxDD {m['max_dd']:+.1%}", flush=True)

    base_nav = pd.read_parquet(BASE_NAV)["nav"]
    results["基线(cap=0)"] = base_nav

    rows = []
    for label, nav in results.items():
        for name, lo, hi in SEGMENTS:
            m = _seg_metrics(nav, lo, hi)
            rows.append({"variant": label, "segment": name,
                         "sharpe": round(m["sharpe"], 3),
                         "ann_ret": round(m["ann_ret"], 4),
                         "max_dd": round(m["max_dd"], 4)})
    df = pd.DataFrame(rows)
    df.to_csv(REPORT_DIR / "regime_filter_test.csv", index=False)

    print("\n=== 分段指标 ===")
    for name, _, _ in SEGMENTS:
        seg = df[df["segment"] == name]
        print(f"{name:12s} " + "  ".join(
            f"{v.split('(')[0]}:{seg[seg['variant'] == v]['sharpe'].iloc[0]:.3f}"
            for v in results))

    # 预设决策规则评估
    print("\n=== 决策规则评估 (通过=2024改善>0.3 AND IS/22-23恶化<10% AND MaxDD不恶化) ===")
    base = {name: _seg_metrics(base_nav, lo, hi) for name, lo, hi in SEGMENTS}
    for label in VARIANTS:
        nav = results[label]
        m24 = _seg_metrics(nav, "2024-01-01", "2025-01-01")
        m_is = _seg_metrics(nav, "2019-01-01", "2022-01-01")
        m_mid = _seg_metrics(nav, "2022-01-01", "2024-01-01")
        m_all = _seg_metrics(nav, None, None)
        ok24 = m24["sharpe"] - base["2024"]["sharpe"] > 0.3
        ok_is = (m_is["sharpe"] >= base["IS 2019-2021"]["sharpe"] * 0.9
                 and m_mid["sharpe"] >= base["2022-2023"]["sharpe"] * 0.9)
        ok_dd = m_all["max_dd"] >= base["全样本"]["max_dd"] - 0.005
        verdict = "PASS" if (ok24 and ok_is and ok_dd) else "FAIL"
        print(f"  {label}: 2024Δ{m24['sharpe']-base['2024']['sharpe']:+.3f} "
              f"ISΔ{m_is['sharpe']-base['IS 2019-2021']['sharpe']:+.3f} "
              f"ok24={ok24} ok_is={ok_is} ok_dd={ok_dd} → {verdict}")

    nav_dir = REPORT_DIR / "regime_filter_navs"
    nav_dir.mkdir(exist_ok=True)
    for label, nav in results.items():
        safe = label.split("(")[0]
        nav.to_frame("nav").to_parquet(nav_dir / f"{safe}.parquet")
    print(f"\nnav 已落盘: {nav_dir}/")


if __name__ == "__main__":
    main()
