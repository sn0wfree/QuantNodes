"""Coverage during extreme-volatility days."""
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

DATA_DIR = ROOT / "data" / "high_freq_macro"
OUT_DIR = ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2" / "data" / "results"


def main() -> None:
    actual_full = pd.read_parquet(DATA_DIR / "v7_6_daily_etf_returns.parquet").dropna(thresh=int(2058 * 0.7), axis=1).ffill().fillna(0.0)
    sample = pd.read_parquet(OUT_DIR / "hw_Normal_Vol.parquet")
    actual = actual_full.loc[sample.index, sample.columns]
    cross_dispersion = actual.std(axis=1)
    vol_90 = cross_dispersion.quantile(0.9)
    extreme_mask = cross_dispersion > vol_90

    methods = ["Normal-Vol", "PerAsset-CP", "Vol-CP", "Global-CP", "CA-GCP"]

    rows = []
    for m in methods:
        lo = pd.read_parquet(OUT_DIR / f"lo_{m.replace('-', '_')}.parquet")
        up = pd.read_parquet(OUT_DIR / f"up_{m.replace('-', '_')}.parquet")
        covered = ((actual >= lo) & (actual <= up)).astype(float)
        daily_cov = covered.mean(axis=1)
        extreme_cov = daily_cov[extreme_mask].mean()
        rows.append({"method": m, "extreme_cov": float(extreme_cov), "extreme_days": int(extreme_mask.sum())})

    df = pd.DataFrame(rows).set_index("method")
    print(df.round(3))
    df.to_csv(OUT_DIR / "crisis_coverage.csv")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    rolling = covered.rolling(7, min_periods=1).mean()
    for m in methods:
        lo = pd.read_parquet(OUT_DIR / f"lo_{m.replace('-', '_')}.parquet")
        up = pd.read_parquet(OUT_DIR / f"up_{m.replace('-', '_')}.parquet")
        cov = ((actual >= lo) & (actual <= up)).astype(float).mean(axis=1)
        axes[0].plot(cov.index, cov.rolling(7, min_periods=1).mean(), label=m, alpha=0.7)
    for d in cross_dispersion.index[extreme_mask]:
        axes[0].axvspan(d, d + pd.Timedelta(days=1), alpha=0.2, color="red")
    axes[0].axhline(0.95, color="black", linestyle="--", label="95% target")
    axes[0].set_ylabel("Rolling 7d Coverage")
    axes[0].legend(loc="lower left", fontsize=8)
    axes[0].set_title("Daily Coverage on Test Set (red shading = extreme-volatility days)")
    axes[0].grid(alpha=0.3)
    axes[1].plot(cross_dispersion.index, cross_dispersion.values, color="black")
    axes[1].set_ylabel("Cross-section std (volatility)")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "crisis_coverage_timeseries.png", dpi=120)


if __name__ == "__main__":
    main()