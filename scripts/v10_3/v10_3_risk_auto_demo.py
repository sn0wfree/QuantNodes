"""AutoRiskWarning 自动优化演示 — 展示 self-healing 全流程.

模拟每周更新, 展示:
  1. Bayesian 权重自动调整
  2. 健康评分变化趋势
  3. 状态切换 + 自动优化动作
  4. 优化前后对比 (校准/重建/回退)
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.risk import (
    AutoRiskWarning, load_etf_data, PlattCalibrator,
)
log = print


def show_optimization_impact(aw: AutoRiskWarning, label: str):
    """Show before/after metrics for the optimization action."""
    rw = aw.rw
    prob = rw._ensemble_prob.tail(120)
    yt = rw.y_true.reindex(prob.index).dropna()
    common = yt.index.intersection(prob.dropna().index)
    if len(common) < 10:
        return

    from sklearn.metrics import roc_auc_score, brier_score_loss
    yt_v = yt.loc[common].values
    p = prob.loc[common].values

    # Before calibration ECE
    bins = pd.cut(pd.Series(p), bins=10, labels=False)
    ece_before = 0.0
    for b in range(10):
        m = (bins == b)
        if m.sum() > 0:
            ece_before += abs(p[m].mean() - yt_v[m].mean()) * m.sum()
    ece_before /= len(p)

    log(f"  {label}: AUC={roc_auc_score(yt_v, p):.3f}  "
        f"Brier={brier_score_loss(yt_v, p):.3f}  ECE={ece_before:.3f}")


def simulate_with_optimization(aw: AutoRiskWarning, n_weeks: int = 12):
    log(f"{'Week':<6} {'Health':<8} {'Status':<10} {'Action':<16} {'Fallback':<9} {'Weights':<30}")
    log("-" * 90)

    for w in range(n_weeks):
        detail = aw.weekly_update()
        opt = detail.get("optimizer", {})
        wgt = detail["weights"]
        fb = "FALLBACK" if detail.get("fallback_active") else "normal"
        w_str = " ".join(f"{k}={v:.0%}" for k, v in wgt.items())
        log(f"{w+1:<6} {detail['health_score']:<8} {detail['status']:<10} "
            f"{opt.get('action_taken', '?'):<16} {fb:<9} {w_str:<30}")

        if w == 0:
            log("  ↑ 初始状态")
        if w > 0 and opt.get("changed"):
            log(f"  → 优化完成: {opt['action_taken']}")


def main():
    log("=" * 60)
    log("AutoRiskWarning — 自动优化演示")
    log("=" * 60)

    etf = load_etf_data()

    for asset in ["equity", "commodity", "gold"]:
        log(f"\n--- {asset.upper()} ---")
        aw = AutoRiskWarning(etf, asset)

        # Show baseline before any optimization
        log("\n  基线性能 (优化前):")
        rw = aw.rw
        for layer in ["ensemble", "climatology"]:
            m = rw._metrics.get(layer, {})
            if m:
                log(f"    {layer}: AUC={m['auc']:.3f}  Brier={m['brier']:.3f}  ECE={m['ece']:.3f}")

        simulate_with_optimization(aw, 8)

        # Show impact after optimization
        log("\n  优化后性能:")
        show_optimization_impact(aw, "ensemble")

        # Version history
        versions = aw.versions.list_versions()
        if len(versions) > 0:
            log(f"\n  版本历史: {len(versions)} 个快照")
            latest = versions.iloc[-1]
            log(f"  最新: {latest.get('status', '?')} "
                f"(健康评分: {latest.get('health_score', '?')})")

        log("")

    log("[Done]")


if __name__ == "__main__":
    main()