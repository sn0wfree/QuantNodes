"""Coverage comparison across 5 methods + CA-GCP."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2"))

from _path import *  # noqa: F401,F403
from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import compute_coverage_metrics, width_bps  # noqa: E402

DATA_DIR = ROOT / "data" / "high_freq_macro"
OUT_DIR = ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2" / "data" / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib-end", default="2022-04-11")
    args = parser.parse_args()

    actual = pd.read_parquet(DATA_DIR / "v7_6_daily_etf_returns.parquet").dropna(thresh=int(2058 * 0.7), axis=1).ffill().fillna(0.0)
    sample = pd.read_parquet(OUT_DIR / "lo_Normal_Vol.parquet")
    actual = actual.loc[sample.index, sample.columns]

    cross_dispersion = actual.std(axis=1)
    vol_90 = cross_dispersion.quantile(0.9)
    extreme_vol = cross_dispersion > vol_90

    methods = ["Normal-Vol", "PerAsset-CP", "Vol-CP", "Global-CP", "CA-GCP"]
    rows = []
    for m in methods:
        lo = pd.read_parquet(OUT_DIR / f"lo_{m.replace('-', '_')}.parquet")
        up = pd.read_parquet(OUT_DIR / f"up_{m.replace('-', '_')}.parquet")
        hw = pd.read_parquet(OUT_DIR / f"hw_{m.replace('-', '_')}.parquet")
        metrics = compute_coverage_metrics(actual, lo, up, extreme_vol)
        metrics["width_bps"] = width_bps(hw)
        metrics["method"] = m
        rows.append(metrics)

    df = pd.DataFrame(rows).set_index("method")
    print(df.round(3))
    df.to_csv(OUT_DIR / "coverage_table.csv")


if __name__ == "__main__":
    main()