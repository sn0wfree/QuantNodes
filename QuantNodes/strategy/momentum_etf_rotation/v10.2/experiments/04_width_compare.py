"""Width timeseries + width-volatility correlation."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2"))

from _path import *  # noqa: F401,F403
from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import width_timeseries, width_volatility_correlation  # noqa: E402

DATA_DIR = ROOT / "data" / "high_freq_macro"
OUT_DIR = ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2" / "data" / "results"


def main() -> None:
    actual_full = pd.read_parquet(DATA_DIR / "v7_6_daily_etf_returns.parquet").dropna(thresh=int(2058 * 0.7), axis=1).ffill().fillna(0.0)
    sample = pd.read_parquet(OUT_DIR / "hw_Normal_Vol.parquet")
    actual = actual_full.loc[sample.index, sample.columns]
    realized_vol = actual.std(axis=1)

    methods = ["Normal-Vol", "PerAsset-CP", "Vol-CP", "Global-CP", "CA-GCP"]
    rows = []
    fig, ax = plt.subplots(figsize=(12, 5))
    for m in methods:
        hw = pd.read_parquet(OUT_DIR / f"hw_{m.replace('-', '_')}.parquet")
        w = width_timeseries(hw)
        rows.append({"method": m, "mean_bps": float(w.mean() * 1e4), "corr_vol": width_volatility_correlation(hw, realized_vol)})
        ax.plot(w.index, w.values * 1e4, label=m, alpha=0.7)
    ax.set_title("Mean Interval Width (bps) Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Width (bps)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "width_timeseries.png", dpi=120)

    df = pd.DataFrame(rows).set_index("method")
    print(df.round(3))
    df.to_csv(OUT_DIR / "width_summary.csv")


if __name__ == "__main__":
    main()