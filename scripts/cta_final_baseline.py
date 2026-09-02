# coding=utf-8
"""cta_final_baseline.py — 最终定版baseline回测.

最终定版配置 (lump口径, 按品种校准成本):
    tvpr × inv_vol, ema=0.15, mc=0.30, target_vol=1.5,
    roll_model='lump', roll_cost_bps=10.0,
    按品种手续费+滑点 (cost_model.json)

对比基准 (保留作参考):
    amortized口径 (旧低估口径) 在另一文件跑

输出:
    reports/cta/final_{nav,summary,yearly}_lump.csv

用法:
    python3 scripts/cta_final_baseline.py
"""
from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path

import pandas as pd

from QuantNodes.strategy.cta.strategies.factor_modules import (
    ComboConfig, FactorModuleComboStrategy,
)

REPORT_DIR = Path("reports/cta")

# 最终定版 (lump口径)
FINAL_LUMP_CFG = dict(
    aggregation="tvpr", module_weighting="inv_vol",
    continuous=True, cont_z_window=252, cont_z_clip=3.0,
    momentum_filter_window=60, momentum_filter_coef=0.0,
    target_vol=1.5, max_leverage=3.0, leverage_hard_cap=3.0,
    margin_call=True, position_ema_alpha=0.15,
    roll_cost_bps=10.0, roll_model="lump",
)


def apply_calibrated_costs(cfg_kw: dict) -> dict:
    """按品种成本: cost_model.json 的手续费+滑点表"""
    out = dict(cfg_kw)
    table = json.loads((REPORT_DIR / "cost_model.json").read_text())
    out["commission_rate"] = {
        p: v["commission_bps"] / 10000 for p, v in table.items()
    }
    out["slippage_bps"] = {p: v["slippage_bps"] for p, v in table.items()}
    return out


def run_final_lump() -> dict:
    cfg_kw = apply_calibrated_costs(FINAL_LUMP_CFG)
    cfg = ComboConfig(**{f.name: cfg_kw[f.name] for f in fields(ComboConfig)
                         if f.name in cfg_kw})
    s = FactorModuleComboStrategy(config=cfg)
    r = s.run()
    dates = pd.DatetimeIndex(r.dates)
    nav = r.nav
    ret = nav.pct_change().dropna()
    m = r.metrics

    # NAV序列
    pd.DataFrame({"nav": nav.values}, index=dates).to_csv(
        REPORT_DIR / "final_nav_lump.csv"
    )

    # 指标汇总
    summary = {
        "config": {
            "roll_model": cfg.roll_model,
            "roll_cost_bps": cfg.roll_cost_bps,
            "target_vol": cfg.target_vol,
            "position_ema_alpha": cfg.position_ema_alpha,
            "cost_basis": "per_product_calibrated",
        },
        "metrics": {k: float(v) for k, v in m.items()},
        "n_products": len(s._price_panel.columns),
        "date_range": [str(dates.min().date()), str(dates.max().date())],
    }
    (REPORT_DIR / "final_summary_lump.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    # 年度收益
    yearly = ret.groupby(ret.index.year).apply(lambda x: (1 + x).prod() - 1)
    yearly.rename("ann_ret").to_csv(REPORT_DIR / "final_yearly_lump.csv")

    print(f"=== 最终定版 (lump口径, 按品种校准成本) ===")
    print(f"Sharpe={m['Sharpe']:.3f} AnnRet={m['AnnRet']:.2%} "
          f"Vol={m['Vol']:.3f} MaxDD={m['MaxDD']:.2%} Calmar={m['Calmar']:.3f}")
    print(f"品种数={summary['n_products']}  区间={summary['date_range']}")
    print(f"年度收益:")
    for yr, r in yearly.items():
        print(f"  {yr}: {r:+.2%}")
    print(f"已保存: {REPORT_DIR}/final_*.csv")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="lump", help="输出文件标签")
    args = parser.parse_args()
    run_final_lump()


if __name__ == "__main__":
    main()