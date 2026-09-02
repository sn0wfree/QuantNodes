# coding=utf-8
"""cta_dual_basis.py — 定版配置双口径对比 (含EC/剔EC).

实盘监控脚本: 输出 Sharpe/AnnRet/MaxDD/Calmar/年度的 含EC vs 剔EC 两套口径.

定版配置: tv=1.5, ema=0.15, tvpr×inv_vol, 按品种校准成本.

用法:
    python3 scripts/cta_dual_basis.py
"""
from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.data_loader import CtaDataLoader
from QuantNodes.strategy.cta.strategies.factor_modules import (
    ComboConfig, FactorModuleComboStrategy,
)

REPORT_DIR = Path("reports/cta")

CFG_KW = dict(
    aggregation="tvpr", module_weighting="inv_vol",
    continuous=True, cont_z_window=252, cont_z_clip=3.0,
    momentum_filter_window=60, momentum_filter_coef=0.0,
    target_vol=1.5, max_leverage=3.0, leverage_hard_cap=3.0,
    margin_call=True, position_ema_alpha=0.15,
    roll_cost_bps=10.0, roll_model="lump",
)


def apply_calibrated_costs(cfg_kw: dict) -> dict:
    out = dict(cfg_kw)
    import json
    table = json.loads((REPORT_DIR / "cost_model.json").read_text())
    out["commission_rate"] = {p: v["commission_bps"] / 10000 for p, v in table.items()}
    out["slippage_bps"] = {p: v["slippage_bps"] for p, v in table.items()}
    return out


def run_once(exclude: list | None = None) -> dict:
    cfg_kw = apply_calibrated_costs(CFG_KW)
    cfg = ComboConfig(**{f.name: cfg_kw[f.name] for f in fields(ComboConfig)
                         if f.name in cfg_kw})
    if exclude:
        all_prods = CtaDataLoader().list_active_products()
        products = [p for p in all_prods if p not in exclude]
    else:
        products = None
    s = FactorModuleComboStrategy(config=cfg, products=products)
    r = s.run()
    nav = pd.Series(r.nav, index=pd.DatetimeIndex(r.dates))
    ret = nav.pct_change().dropna()
    n = len(ret)
    cagr = nav.iloc[-1] ** (252 / n) - 1
    vol = ret.std() * np.sqrt(252)
    dd = (nav / nav.cummax() - 1).min()
    yearly = ret.groupby(ret.index.year).apply(lambda x: (1 + x).prod() - 1)
    return {
        "sharpe": cagr / vol, "ann_ret": cagr, "vol": vol, "max_dd": dd,
        "calmar": cagr / abs(dd), "yearly": yearly,
    }


def main():
    rows = []
    configs = [
        ("含EC", None),
        ("剔EC", ["EC"]),
    ]
    yearly_df = pd.DataFrame()
    for label, exclude in configs:
        r = run_once(exclude)
        rows.append({"口径": label, **{k: r[k] for k in
                    ["sharpe", "ann_ret", "vol", "max_dd", "calmar"]}})
        yearly_df[label] = r["yearly"]

    df = pd.DataFrame(rows)
    print("=" * 70)
    print("双口径对比 (tv=1.5, ema=0.15, 按品种校准成本)")
    print("=" * 70)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n年度收益:")
    for col in yearly_df.columns:
        print(f"  {col}:")
    for yr, row in yearly_df.iterrows():
        print(f"  {yr}: " + "  ".join(f"{col}={row[col]:+.2%}" for col in yearly_df.columns))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_DIR / "dual_basis.csv", index=False)
    yearly_df.to_csv(REPORT_DIR / "dual_basis_yearly.csv")
    print(f"\n已保存: {REPORT_DIR}/dual_basis{{,_yearly}}.csv")


if __name__ == "__main__":
    main()
