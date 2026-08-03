"""AutoRiskWarning 自动学习框架演示.

模拟每周更新, 展示:
  1. Bayesian 权重自动调整
  2. 健康评分变化趋势
  3. 状态切换
  4. 版本历史对比
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.v10_3.risk_warning import (
    AutoRiskWarning, load_etf_data,
)
log = print


def simulate_weekly(aw: AutoRiskWarning, n_weeks: int = 12):
    log(f"{'Week':<6} {'Health':<8} {'Status':<10} {'Action':<12} "
        f"{'Clim':<8} {'HAR-RV':<8} {'EWMA':<8} {'CurVol':<8}")
    log("-" * 80)

    for w in range(n_weeks):
        detail = aw.weekly_update()
        wgt = detail["weights"]
        unc = detail["weight_uncertainty"]
        log(f"{w+1:<6} {detail['health_score']:<8} {detail['status']:<10} "
            f"{detail['action']:<12} "
            f"{wgt['climatology']:<8.1%} {wgt['harrv']:<8.1%} "
            f"{wgt['ewma']:<8.1%} {wgt['current_vol']:<8.1%}")
        if w > 0 and w % 4 == 0:
            log(f"  Uncertainty: clim={unc['climatology']:.3f} "
                f"harrv={unc['harrv']:.3f} ewma={unc['ewma']:.3f} "
                f"cur={unc['current_vol']:.3f}")


def main():
    log("=" * 60)
    log("AutoRiskWarning — 自动学习框架演示")
    log("=" * 60)

    etf = load_etf_data()

    for asset in ["equity", "commodity", "gold"]:
        log(f"\n--- {asset.upper()} ---")
        aw = AutoRiskWarning(etf, asset)
        simulate_weekly(aw, 8)

        versions = aw.versions.list_versions()
        if len(versions) > 0:
            log(f"\n  版本历史: {len(versions)} 个快照")
            log(f"  最新状态: {versions['status'].iloc[-1]} "
                f"(健康评分: {versions['health_score'].iloc[-1]})")

        pred = aw.predict("2024-01-04")
        log(f"  预测 2024-01-04: {pred:.1%}")

        log("")

    log("[Done]")


if __name__ == "__main__":
    main()