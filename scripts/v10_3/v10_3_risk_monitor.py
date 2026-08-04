"""v10.3 Risk Warning — Monitoring: Generate evaluation reports for all assets.

Usage:
    python3 scripts/v10_3/v10_3_risk_monitor.py
    python3 scripts/v10_3/v10_3_risk_monitor.py --save
"""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.risk import (
    EnsembleRiskWarning, ModelMonitor, load_etf_data,
)
log = print

OUT_DIR = REPO / "reports" / "momentum_etf_rotation" / "v10_3"
MON_DIR = OUT_DIR / "monitoring"
MON_DIR.mkdir(exist_ok=True, parents=True)


def run_monitor(asset: str, save: bool = False):
    log("=" * 60)
    log(f"Monitor: {asset.upper()}")
    log("=" * 60)

    etf = load_etf_data()
    rw = EnsembleRiskWarning(etf, asset)

    full = rw.evaluate()
    log("\nFull OOS:")
    for layer, m in sorted(full.items()):
        log(f"  {layer:<12} AUC={m['auc']:.4f}  Brier={m['brier']:.4f}  ECE={m['ece']:.4f}")

    mon = ModelMonitor(rw)
    log(f"\nRolling window: {len(mon._rolling)} days")

    log("\n" + mon.report())

    if save:
        path = MON_DIR / f"{asset}_monitor.parquet"
        mon._rolling.to_parquet(path)
        log(f"\nSaved rolling metrics: {path}")

        card = pd.DataFrame([{
            "asset": asset, "created": pd.Timestamp.now(),
            "auc": full["ensemble"]["auc"],
            "prauc": full["ensemble"]["prauc"],
            "brier": full["ensemble"]["brier"],
            "ece": full["ensemble"]["ece"],
            "climatology_auc": full["climatology"]["auc"],
            "status": mon.current_status,
            "n_alerts": len(mon.check_alerts()),
        }])
        card_path = OUT_DIR / "model_card.csv"
        if card_path.exists():
            prev = pd.read_csv(card_path)
            card = pd.concat([prev, card], ignore_index=True)
        card.to_csv(card_path, index=False)
        log(f"Saved model card: {card_path}")

    return mon


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="save to reports/")
    args = parser.parse_args()

    for asset in ["equity", "commodity", "gold"]:
        run_monitor(asset, save=args.save)
        log("")