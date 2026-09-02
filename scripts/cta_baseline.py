# coding=utf-8
"""cta_baseline.py — 保存CTA策略baseline回测结果.

当前最优配置 (tvpr × inv_vol), 结果写入 reports/cta/.

用途:
    - 改进前保存基准 (对比修复效果)
    - 定期重跑追踪策略退化

用法:
    python3 scripts/cta_baseline.py            # 跑回测+保存CSV
    python3 scripts/cta_baseline.py --tag fix  # 输出文件加标签
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from QuantNodes.strategy.cta.strategies.factor_modules import (
    ComboConfig, FactorModuleComboStrategy,
)

REPORT_DIR = Path("reports/cta")

CFG = ComboConfig(
    aggregation="tvpr", module_weighting="inv_vol",
    continuous=True, cont_z_window=252, cont_z_clip=3.0,
    momentum_filter_window=60, momentum_filter_coef=0.0,
    target_vol=0.30, max_leverage=3.0, leverage_hard_cap=3.0,
    margin_call=True,
)


def run_baseline(tag: str = "") -> dict:
    s = FactorModuleComboStrategy(config=CFG)
    r = s.run()
    dates = pd.DatetimeIndex(r.dates)
    nav = r.nav
    ret = nav.pct_change().dropna()
    m = r.metrics

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""

    # 1. NAV序列
    pd.DataFrame({"nav": nav.values}, index=dates).to_csv(
        REPORT_DIR / f"baseline_nav{suffix}.csv"
    )

    # 2. 指标汇总
    summary = {
        "config": {
            "aggregation": CFG.aggregation,
            "module_weighting": CFG.module_weighting,
            "target_vol": CFG.target_vol,
            "max_leverage": CFG.max_leverage,
            "slippage_bps": getattr(CFG, "slippage_bps", None),
        },
        "metrics": {k: float(v) for k, v in m.items()},
        "n_products": len(s._price_panel.columns),
        "date_range": [str(dates.min().date()), str(dates.max().date())],
    }
    (REPORT_DIR / f"baseline_summary{suffix}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    # 3. 年度收益
    yearly = ret.groupby(ret.index.year).apply(
        lambda x: (1 + x).prod() - 1
    )
    yearly.rename("ann_ret").to_csv(REPORT_DIR / f"baseline_yearly{suffix}.csv")

    print(f"Sharpe={m['Sharpe']:.3f} AnnRet={m['AnnRet']:.3f} Vol={m['Vol']:.3f} "
          f"MaxDD={m['MaxDD']:.3f} Calmar={m['Calmar']:.3f}")
    print(f"品种数={summary['n_products']}  区间={summary['date_range']}")
    print(f"年度收益: {yearly.apply(lambda x: f'{x:+.1%}').to_dict()}")
    print(f"已保存: {REPORT_DIR}/baseline_*{suffix}.*")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="", help="输出文件标签")
    args = parser.parse_args()
    run_baseline(args.tag)


if __name__ == "__main__":
    main()
