"""v10.3 Risk Warning — Demo: Evaluate all 3 assets + signal distribution.

Usage:
    python3 scripts/v10_3/v10_3_risk_warning_demo.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.risk import (
    EnsembleRiskWarning, ModelMonitor, load_etf_data, CONFIGS,
)
log = print


def run_demo(asset: str):
    log("=" * 60)
    log(f"Risk Warning Demo: {asset.upper()}")
    log("=" * 60)

    etf = load_etf_data()
    rw = EnsembleRiskWarning(etf, asset)

    log("\n[1] Full OOS Evaluation:")
    m = rw.evaluate()
    for layer, metrics in sorted(m.items()):
        log(f"  {layer:<15} AUC={metrics['auc']:.4f}  PR-AUC={metrics['prauc']:.4f}  "
            f"Brier={metrics['brier']:.4f}  ECE={metrics['ece']:.4f}  "
            f"PosRate={metrics['pos_rate']:.1%}")

    log("\n[2] Signal Distribution (OOS):")
    df = rw.predict_range(rw.oos_start, rw.oos_end)
    levels = [(0, 0.5, "Normal"), (0.5, 0.7, "Alert"),
              (0.7, 0.9, "Warning"), (0.9, 1.0, "Danger")]
    for lo, hi, lbl in levels:
        cnt = ((df["risk_prob"] >= lo) & (df["risk_prob"] < hi)).sum()
        pct = cnt / max(len(df), 1) * 100
        bar = "█" * int(pct / 3) + "░" * (33 - int(pct / 3))
        log(f"  {lbl:<10} {cnt:>4}d ({pct:>5.1f}%) {bar}")

    high = df[df["signal"] >= 3]
    if len(high) > 0:
        log(f"\n[3] High Risk Events (signal=3):")
        log(f"  Total: {len(high)} days")
        clusters = []
        for d in high.index:
            if not clusters or (d - clusters[-1][-1]).days > 5:
                clusters.append([d])
            else:
                clusters[-1].append(d)
        for c in clusters[:5]:
            start, end = c[0].date(), c[-1].date()
            t = rw.y_true.loc[start:end].sum()
            log(f"  {start} ~ {end}  ({len(c)}d, {int(t)} actual high-vol days)")

    log("\n[4] Monitor Check:")
    mon = ModelMonitor(rw)
    log(f"  Rolling metrics: {len(mon._rolling)} days")
    log(f"  Drift detected: {mon.drift_detected()}")
    alerts = mon.check_alerts()
    if alerts:
        log(f"  Alerts: {len(alerts)}")
        for a in alerts:
            log(f"    {a['rule']}")
    else:
        log(f"  Alerts: None")
    log(f"  Status: {mon.current_status.upper()}")

    log("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    for asset in ["equity", "commodity", "gold"]:
        run_demo(asset)
    log("[Done]")