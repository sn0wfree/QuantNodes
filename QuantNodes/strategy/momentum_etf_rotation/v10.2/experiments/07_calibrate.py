"""Grid search over (k, eta, tau) using calib-period coverage criteria."""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2"))

from _path import *  # noqa: F401,F403
from ca_gcp import CAGCPConfig, CAGCPipeline  # noqa: E402
from ca_gcp.validators import compute_coverage_metrics, width_bps  # noqa: E402

DATA_DIR = ROOT / "data" / "high_freq_macro"
OUT_DIR = ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2" / "data" / "results"


def score(metrics: dict, width_bps_val: float) -> float:
    """Pareto score: prefer higher extr_cov, lower pa_std, narrower width."""
    if metrics["extreme"] < 0:
        return -1e9
    return metrics["extreme"] - 5.0 * metrics["pa_std"] - width_bps_val / 1000.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-end", default="2020-04-12")
    parser.add_argument("--calib-end", default="2021-04-12")
    parser.add_argument("--val-end", default="2022-04-12")
    args = parser.parse_args()

    df = pd.read_parquet(DATA_DIR / "v7_6_daily_etf_returns.parquet").dropna(thresh=int(2058 * 0.7), axis=1).ffill().fillna(0.0)
    train_end = pd.Timestamp(args.train_end)
    calib_end = pd.Timestamp(args.calib_end)
    val_end = pd.Timestamp(args.val_end)
    train = df.loc[:train_end].iloc[:-1]
    calib = df.loc[train_end:calib_end]
    val = df.loc[calib_end:val_end].iloc[1:]
    actual = val

    cross_dispersion = actual.std(axis=1)
    vol_90 = cross_dispersion.quantile(0.9)
    extreme_mask = cross_dispersion > vol_90

    grid = list(itertools.product([4, 8], [0.0, 0.3, 0.5], [40.0]))
    rows = []
    best = None
    for k, eta, tau in grid:
        cfg = CAGCPConfig(k=k, sensitivity_eta=eta, recency_tau=tau)
        pipe = CAGCPipeline(cfg)
        pipe.fit(train)
        out = pipe.predict(calib, val)
        m = compute_coverage_metrics(actual, out["lower"], out["upper"], extreme_mask)
        w_bps = width_bps(out["half_width"])
        s = score(m, w_bps)
        rows.append(
            {
                "k": k,
                "eta": eta,
                "tau": tau,
                "marginal": m["marginal"],
                "pa_std": m["pa_std"],
                "worst10": m["worst10"],
                "extreme": m["extreme"],
                "width_bps": w_bps,
                "score": s,
            }
        )
        if best is None or s > best["score"]:
            best = rows[-1]

    grid_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    grid_df.to_csv(OUT_DIR / "calibration_grid.csv", index=False)
    print("Top 5:")
    print(grid_df.head().round(3).to_string(index=False))
    print("\nBest:", best)


if __name__ == "__main__":
    main()