# coding=utf-8
"""cta_turnover_experiment.py — 换手率改善实验 (Phase 1).

对比配置:
    baseline: position_ema_alpha=0, min_position_change=0.30 (当前默认)
    exp:      position_ema_alpha=0.15, min_position_change=0.15

输出:
    - 换手率 (年化, 双边)
    - Sharpe/AnnRet/Vol/MaxDD/Calmar (无成本 + 1bp滑点)
    - 结果落盘 reports/cta/turnover_experiment.csv

用法:
    python3 scripts/cta_turnover_experiment.py
"""
from __future__ import annotations

import argparse
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.strategies.factor_modules import (
    ComboConfig, FactorModuleComboStrategy,
)

REPORT_DIR = Path("reports/cta")

BASE_CFG = dict(
    aggregation="tvpr", module_weighting="inv_vol",
    continuous=True, cont_z_window=252, cont_z_clip=3.0,
    momentum_filter_window=60, momentum_filter_coef=0.0,
    target_vol=0.30, max_leverage=3.0, leverage_hard_cap=3.0,
    margin_call=True,
)

EXPERIMENTS = {
    "baseline(ema=0,mc=0.30)": {},
    "ema_a10(ema=0.10)": {"position_ema_alpha": 0.10},
    "ema_only(ema=0.15)": {"position_ema_alpha": 0.15},
    "ema_a20(ema=0.20)": {"position_ema_alpha": 0.20},
    "ema_a30(ema=0.30)": {"position_ema_alpha": 0.30},
    "mc_only(mc=0.15)": {"min_position_change": 0.15},
    "mc_high(mc=0.50)": {"min_position_change": 0.50},
    "combo(ema=0.15,mc=0.15)": {
        "position_ema_alpha": 0.15, "min_position_change": 0.15,
    },
    "final(ema=0.15)": {"position_ema_alpha": 0.15},
    "regime_on(ema=0.15)": {
        "position_ema_alpha": 0.15, "regime_filter_enabled": True,
    },
    "cap5(ema=0.15)": {
        # 5% equity上限 → coef = 0.05/max_leverage = 0.0167
        "position_ema_alpha": 0.15, "max_product_notional_coef": 0.05 / 3.0,
    },
    "regime+cap5(ema=0.15)": {
        "position_ema_alpha": 0.15, "regime_filter_enabled": True,
        "max_product_notional_coef": 0.05 / 3.0,
    },
    "lev_t40(tv=0.40)": {"position_ema_alpha": 0.15, "target_vol": 0.40},
    "lev_t50(tv=0.50)": {"position_ema_alpha": 0.15, "target_vol": 0.50},
    "lev_t55(tv=0.55)": {"position_ema_alpha": 0.15, "target_vol": 0.55},
    "lev_t60(tv=0.60)": {"position_ema_alpha": 0.15, "target_vol": 0.60},
    "lev_t100(tv=1.00)": {"position_ema_alpha": 0.15, "target_vol": 1.00},
    "lev_t150(tv=1.50)": {"position_ema_alpha": 0.15, "target_vol": 1.50},
    "lev_t200(tv=2.00)": {"position_ema_alpha": 0.15, "target_vol": 2.00},
    "lev_t300(tv=3.00)": {"position_ema_alpha": 0.15, "target_vol": 3.00},
    "lev_k17(tv=1.5,k=1.7)": {
        "position_ema_alpha": 0.15, "target_vol": 1.50, "equity_leverage_k": 1.7,
    },
    "lev_k20(tv=1.5,k=2.0)": {
        "position_ema_alpha": 0.15, "target_vol": 1.50, "equity_leverage_k": 2.0,
    },
    "final_calibrated(tv=1.5,按品种成本)": {
        "position_ema_alpha": 0.15, "target_vol": 1.50,
    },
    "final_calibrated+roll5(tv=1.5)": {
        "position_ema_alpha": 0.15, "target_vol": 1.50,
        "roll_cost_bps": 5.0,
    },
    "final_calibrated+roll10(tv=1.5)": {
        "position_ema_alpha": 0.15, "target_vol": 1.50,
        "roll_cost_bps": 10.0,
    },
    "final_calibrated+roll20(tv=1.5)": {
        "position_ema_alpha": 0.15, "target_vol": 1.50,
        "roll_cost_bps": 20.0,
    },
    "final_calibrated+roll50(tv=1.5)": {
        "position_ema_alpha": 0.15, "target_vol": 1.50,
        "roll_cost_bps": 50.0,
    },
    "final_lump10(tv=1.5,一次性换月10bp)": {
        "position_ema_alpha": 0.15, "target_vol": 1.50,
        "roll_cost_bps": 10.0, "roll_model": "lump",
    },
    "final_lump20(tv=1.5,一次性20bp)": {
        "position_ema_alpha": 0.15, "target_vol": 1.50,
        "roll_cost_bps": 20.0, "roll_model": "lump",
    },
    "final_lump30(tv=1.5,一次性30bp)": {
        "position_ema_alpha": 0.15, "target_vol": 1.50,
        "roll_cost_bps": 30.0, "roll_model": "lump",
    },
}


def run_one(name: str, overrides: dict, slippage: float | None,
            commission: float | None = None) -> dict:
    cfg_kw = dict(BASE_CFG)
    cfg_kw.update(overrides)
    # 任何含lump的变体都走按品种成本+真实换月成本
    use_calibrated = ("按品种成本" in name
                     or "+roll" in name
                     or "lump" in name.lower())
    if use_calibrated:
        import json
        table = json.loads((REPORT_DIR / "cost_model.json").read_text())
        cfg_kw["commission_rate"] = {
            p: v["commission_bps"] / 10000 for p, v in table.items()
        }
        cfg_kw["slippage_bps"] = {
            p: v["slippage_bps"] for p, v in table.items()
        }
    else:
        cfg_kw["slippage_bps"] = slippage
        if commission is not None:
            cfg_kw["commission_rate"] = commission
    cfg = ComboConfig(**{f: cfg_kw.get(f) for f in
                         (f.name for f in fields(ComboConfig)) if f in cfg_kw})
    t0 = time.time()
    s = FactorModuleComboStrategy(config=cfg)
    r = s.run()
    elapsed = time.time() - t0

    dates = pd.DatetimeIndex(r.dates)
    equity = pd.Series(r.equity, index=dates)
    pos = pd.DataFrame(r.positions, index=dates)
    m = r.metrics

    # 换手率 (名义/平均gross)
    prices = s._price_panel.reindex(dates)
    spec = {p: s._registry.get_or_default(p) for p in prices.columns}
    notional = pd.DataFrame(
        {p: pos[p].abs() * prices[p] * spec[p].contract_multiplier
         for p in prices.columns if p in pos.columns},
    )
    gross = notional.sum(axis=1)
    daily_turnover = notional.diff().abs().sum(axis=1) / gross.replace(0, np.nan)
    ann_turnover = daily_turnover.mean() * 252

    return {
        "exp": name,
        "slippage_bps": slippage or 0.0,
        "sharpe": m["Sharpe"], "ann_ret": m["AnnRet"], "vol": m["Vol"],
        "max_dd": m["MaxDD"], "calmar": m["Calmar"],
        "ann_turnover": ann_turnover,
        "avg_gross_lev": (gross / equity).mean(),
        "run_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exps", nargs="*", default=list(EXPERIMENTS),
                        help="要跑的实验名 (默认全部)")
    parser.add_argument("--slippages", nargs="*", type=float,
                        default=[None, 1.0],
                        help="滑点列表 (默认 无成本 + 1bp)")
    parser.add_argument("--commission", type=float, default=None,
                        help="手续费比例 (如0.001=10bp; 默认ComboConfig的1bp)")
    args = parser.parse_args()

    rows = []
    for name in args.exps:
        if name not in EXPERIMENTS:
            raise SystemExit(f"未知实验: {name}, 可选: {list(EXPERIMENTS)}")
        for slip in args.slippages:
            row = run_one(name, EXPERIMENTS[name], slip, commission=args.commission)
            rows.append(row)
            print(f"[{name} slip={slip}] Sharpe={row['sharpe']:.3f} "
                  f"AnnRet={row['ann_ret']:.2%} MaxDD={row['max_dd']:.2%} "
                  f"换手={row['ann_turnover']:.0f}x "
                  f"({row['run_sec']:.0f}s)")

    df = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPORT_DIR / "turnover_experiment.csv", index=False)
    print("\n=== 换手率实验汇总 ===")
    print(df.drop(columns=["run_sec"]).to_string(index=False))
    print(f"\n已保存: {REPORT_DIR}/turnover_experiment.csv")


if __name__ == "__main__":
    main()
