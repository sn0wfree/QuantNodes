# coding=utf-8
"""cta_osc6_test.py — Layer2 实验: OscillatorAlpha 接入主CTA (5模块→6模块).

背景: OscillatorAlpha (15均值回归因子) 在 modules/ 中已构建但不在默认组合.
单跑 Sharpe 1.17 / 2024 +2.3%, 但与主CTA相关 +0.47 (非独立alpha).
本实验测 6模块 (module_classes 注入) vs golden (5模块) 的全样本/分段差异.

决策规则 (预定义): 接受 = 全样本 Sharpe 提升>0.1 AND MaxDD 恶化<0.5pp
口径: 分段重定基 CAGR/Vol (与 G 实验一致)

输出: reports/cta/osc6_test.csv + osc6_nav.parquet
"""
from __future__ import annotations

import json
import sys
from dataclasses import fields as dc_fields
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/ll/Public/QuantNodes")

from QuantNodes.strategy.cta.strategies.base import DEFAULT_PRODUCTS  # noqa: E402
from QuantNodes.strategy.cta.strategies.combo import (  # noqa: E402
    ComboConfig, DEFAULT_COMBO_MODULES, FactorModuleComboStrategy,
)
from QuantNodes.strategy.cta.strategies.modules.oscillator_alpha import (  # noqa: E402
    OscillatorAlphaModuleStrategy,
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
SEGMENTS = [("IS 2019-2021", "2019-01-01", "2022-01-01"),
            ("2022-2023", "2022-01-01", "2024-01-01"),
            ("2024", "2024-01-01", "2025-01-01"),
            ("2025", "2025-01-01", "2026-01-01"),
            ("全样本", None, None)]


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
    cfg_kw = dict(FINAL_LUMP_CFG)
    cfg_kw.update(CALIBRATED_COSTS)
    cfg_kw["module_classes"] = tuple(DEFAULT_COMBO_MODULES) + (OscillatorAlphaModuleStrategy,)
    cfg = ComboConfig(**{f.name: cfg_kw[f.name] for f in dc_fields(ComboConfig)
                         if f.name in cfg_kw})
    print("6模块回测: " + ", ".join(c.__name__ for c in cfg_kw["module_classes"]))
    s = FactorModuleComboStrategy(config=cfg, products=list(DEFAULT_PRODUCTS))
    r = s.run()
    nav6 = r.nav

    golden = pd.read_parquet(GOLDEN)["nav"]
    rows = []
    for label, nav in [("golden(5模块)", golden), ("6模块(+OscAlpha)", nav6)]:
        for name, lo, hi in SEGMENTS:
            m = _seg_metrics(nav, lo, hi)
            rows.append({"variant": label, "segment": name,
                         "sharpe": round(m["sharpe"], 3),
                         "ann_ret": round(m["ann_ret"], 4),
                         "max_dd": round(m["max_dd"], 4)})
    df = pd.DataFrame(rows)
    df.to_csv(REPORT_DIR / "osc6_test.csv", index=False)

    print("\n=== 分段对比 ===")
    for name, _, _ in SEGMENTS:
        g = df[(df["variant"] == "golden(5模块)") & (df["segment"] == name)].iloc[0]
        n = df[(df["variant"] == "6模块(+OscAlpha)") & (df["segment"] == name)].iloc[0]
        print(f"{name:12s} golden {g['sharpe']:.3f} → 6模块 {n['sharpe']:.3f} "
              f"(Δ{n['sharpe']-g['sharpe']:+.3f})  "
              f"DD {g['max_dd']:.1%}→{n['max_dd']:.1%}")

    # 预设决策规则
    g_all = _seg_metrics(golden, None, None)
    n_all = _seg_metrics(nav6, None, None)
    ok_shp = n_all["sharpe"] - g_all["sharpe"] > 0.1
    ok_dd = n_all["max_dd"] >= g_all["max_dd"] - 0.005
    verdict = "ACCEPT" if (ok_shp and ok_dd) else "REJECT"
    print(f"\n决策规则: Sharpe提升>0.1={ok_shp}, MaxDD恶化<0.5pp={ok_dd} → {verdict}")

    nav6.to_frame("nav").to_parquet(REPORT_DIR / "osc6_nav.parquet")
    print("nav 已落盘: osc6_nav.parquet")


if __name__ == "__main__":
    main()
