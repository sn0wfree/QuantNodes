# coding=utf-8
"""cta_weight_cap_test.py — G1 实验: inv_vol 模块权重上限 (module_weight_cap).

背景: inv_vol 把 IndustrialChain (窄覆盖低波模块) 推到80%权重,
TrendMomentum/VolumeFlow 被压到1-3% — 权重分配失真.
本实验测 cap ∈ {0.30, 0.40, 0.50} 对全样本/分段业绩的影响.

任务矩阵:
    cap=0.0   对账自检 (NAV 必须与 golden 逐位一致, ulp 内)
    cap=0.30 / 0.40 / 0.50

决策规则 (预定义, 防数据窥探):
    通过 = 2024 Sharpe 改善 > 0.3 AND 2019-2023 Sharpe 恶化 < 10% AND MaxDD 不恶化
口径: compute_metrics 官方口径 (Sharpe = 日历年CAGR / Vol)

输出: reports/cta/weight_cap_test.csv
      reports/cta/weight_cap_navs/cap{X}.parquet
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
GOLDEN = REPORT_DIR / "overfit_final_navs" / "基线_严格OOS0.5,60_2018.parquet"

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

CAPS = [0.0, 0.30, 0.40, 0.50]
SEGMENTS = [("IS 2019-2021", "2019-01-01", "2022-01-01"),
            ("2022-2023", "2022-01-01", "2024-01-01"),
            ("2024", "2024-01-01", "2025-01-01"),
            ("2025", "2025-01-01", "2026-01-01"),
            ("全样本", None, None)]


def _run_one(cap: float) -> tuple[float, pd.Series, dict]:
    cfg_kw = dict(FINAL_LUMP_CFG)
    cfg_kw["module_weight_cap"] = cap
    cfg_kw.update(CALIBRATED_COSTS)
    cfg = ComboConfig(**{f.name: cfg_kw[f.name] for f in dc_fields(ComboConfig)
                         if f.name in cfg_kw})
    s = FactorModuleComboStrategy(config=cfg, products=list(DEFAULT_PRODUCTS))
    r = s.run()
    # 模块权重重算 (与 combo._signals_raw inv_vol 分支同逻辑)
    cols = list(s._price_panel.columns)
    sigs = [s.module_signals[m.__class__.__name__] for m in s._modules]
    names = [m.__class__.__name__.replace("ModuleStrategy", "") for m in s._modules]
    w = pd.concat([sig.reindex(columns=cols).fillna(0)
                   .rolling(cfg.agg_window, min_periods=20).std().mean(axis=1)
                   for sig in sigs], axis=1)
    w.columns = names
    inv = 1.0 / w.replace(0, np.nan)
    w_norm = inv.div(inv.sum(axis=1), axis=0)
    if cap > 0:
        w_norm = w_norm.clip(upper=cap).div(w_norm.clip(upper=cap).sum(axis=1), axis=0)
    w_norm = w_norm.fillna(1 / len(sigs))
    return cap, r.nav, w_norm


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
    print(f"G1 实验: module_weight_cap ∈ {CAPS}  (3 workers)")
    results = {}
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(_run_one, c): c for c in CAPS}
        for fut in as_completed(futs):
            cap, nav, w_norm = fut.result()
            results[cap] = (nav, w_norm)
            m = _seg_metrics(nav, None, None)
            print(f"  done cap={cap}: Sharpe {m['sharpe']:.3f} "
                  f"AnnRet {m['ann_ret']:+.1%} MaxDD {m['max_dd']:+.1%}", flush=True)

    # 对账自检: cap=0 vs golden
    golden = pd.read_parquet(GOLDEN)["nav"]
    nav0 = results[0.0][0]
    align = golden.index.intersection(nav0.index)
    rel = ((golden.loc[align] - nav0.loc[align]).abs() / golden.loc[align]).max()
    print(f"\n对账自检 (cap=0 vs golden): 最大相对差 = {rel:.2e} "
          f"{'✓' if rel < 1e-12 else '✗ 超噪声底!'}")

    # 分段指标表
    rows = []
    for cap in CAPS:
        nav = results[cap][0]
        for name, lo, hi in SEGMENTS:
            m = _seg_metrics(nav, lo, hi)
            rows.append({"cap": cap, "segment": name, "sharpe": round(m["sharpe"], 3),
                         "ann_ret": round(m["ann_ret"], 4), "max_dd": round(m["max_dd"], 4)})
    df = pd.DataFrame(rows)
    df.to_csv(REPORT_DIR / "weight_cap_test.csv", index=False)
    print("\n=== 分段指标 ===")
    for name, _, _ in SEGMENTS:
        seg = df[df["segment"] == name]
        print(f"{name:12s} " + "  ".join(
            f"cap={c}:{seg[seg['cap'] == c]['sharpe'].iloc[0]:.3f}" for c in CAPS))

    # 2024-2025 模块权重 (G*候选: cap=0.40)
    print("\n=== 模块权重均值 (2024, cap=0.40 vs cap=0) ===")
    for cap in (0.0, 0.40):
        w24 = results[cap][1]
        w24 = w24[w24.index.year == 2024].mean()
        print(f"  cap={cap}: " + "  ".join(f"{c}:{v:.2f}" for c, v in w24.items()))

    # nav 落盘
    nav_dir = REPORT_DIR / "weight_cap_navs"
    nav_dir.mkdir(exist_ok=True)
    for cap, (nav, _) in results.items():
        nav.to_frame("nav").to_parquet(nav_dir / f"cap{cap}.parquet")
    print(f"\nnav 已落盘: {nav_dir}/cap{{0.0,0.3,0.4,0.5}}.parquet")


if __name__ == "__main__":
    main()
