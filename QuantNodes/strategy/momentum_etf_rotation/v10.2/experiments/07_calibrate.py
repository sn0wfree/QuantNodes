"""Hyperparameter calibration: 343-combination grid search with early stop.

Uses the vectorized CAGCPipeline.predict_fast for ~100x speedup.
Saves full grid to calibration_grid.csv and Pareto-optimal params to
best_params.json.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2"))
sys.path.insert(0, str(ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation"))

from _path import *  # noqa: F401,F403
from ca_gcp import CAGCPConfig, CAGCPipeline  # noqa: E402
from ca_gcp.validators import compute_coverage_metrics, width_bps  # noqa: E402

DATA_DIR = ROOT / "data" / "high_freq_macro"
OUT_DIR = ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2" / "data" / "results"

K_GRID = [2, 4, 6, 8, 12, 16, 24]
ETA_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
TAU_GRID = [20, 30, 45, 60, 90, 120, 180]

EARLY_STOP_THRESHOLD = 50


def pareto_score(metrics: dict, width_bps_val: float) -> float:
    if np.isnan(metrics["extreme"]):
        return -1e9
    return (
        10.0 * metrics["extreme"]
        - 5.0 * metrics["pa_std"]
        - 1.0 * (width_bps_val / 1000.0)
    )


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "v7_6_daily_etf_returns.parquet")
    df = df.dropna(thresh=int(len(df) * 0.7), axis=1).ffill().fillna(0.0)

    train_end = pd.Timestamp("2020-04-12")
    calib_end = pd.Timestamp("2021-04-12")
    val_end = pd.Timestamp("2022-04-12")

    train = df.loc[:train_end].iloc[:-1]
    calib = df.loc[train_end:calib_end]
    val = df.loc[calib_end:val_end].iloc[1:]
    actual = val

    cross_dispersion = val.std(axis=1)
    extreme_mask = cross_dispersion > cross_dispersion.quantile(0.9)

    pipe = CAGCPipeline(CAGCPConfig(k=4))
    pipe.fit(train)

    rows = []
    best = None
    best_score = -np.inf
    no_improve_count = 0
    t0 = time.time()
    prev_k = None

    total = len(K_GRID) * len(ETA_GRID) * len(TAU_GRID)
    print(f"Grid size: {total} combinations")

    for idx, (k, eta, tau) in enumerate(itertools.product(K_GRID, ETA_GRID, TAU_GRID), start=1):
        pipe.config.k = k
        pipe.config.sensitivity_eta = eta
        pipe.config.recency_tau = tau

        # Rebuild graph when k changes (Bug 2 fix)
        if k != prev_k:
            pipe.fit(train)
            prev_k = k

        t_fit = time.time()
        out = pipe.predict_fast(calib, val)
        t_predict = time.time() - t_fit

        m = compute_coverage_metrics(actual, out["lower"], out["upper"], extreme_mask)
        w_bps = width_bps(out["half_width"])
        s = pareto_score(m, w_bps)

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
                "runtime_s": t_predict,
            }
        )

        if s > best_score:
            best_score = s
            best = rows[-1]
            no_improve_count = 0
            print(f"  [{idx}/{total}] k={k} η={eta} τ={tau}: NEW BEST score={s:.3f} (marginal={m['marginal']:.3f}, extreme={m['extreme']:.3f}, width={w_bps:.0f}bps, {t_predict:.1f}s)")
        else:
            no_improve_count += 1

        if no_improve_count >= EARLY_STOP_THRESHOLD and idx >= 100:
            print(f"  Early stop at {idx}/{total} (no improvement for {EARLY_STOP_THRESHOLD} combos)")
            break

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed / max(len(rows), 1):.2f}s/combo)")

    grid_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    grid_df.to_csv(OUT_DIR / "calibration_grid.csv", index=False)

    if best is not None:
        best_params = {
            "k": int(best["k"]),
            "eta": float(best["eta"]),
            "tau": float(best["tau"]),
            "marginal": float(best["marginal"]),
            "pa_std": float(best["pa_std"]),
            "extreme": float(best["extreme"]),
            "width_bps": float(best["width_bps"]),
            "score": float(best["score"]),
            "n_evaluated": len(rows),
        }
        with (OUT_DIR / "best_params.json").open("w") as f:
            json.dump(best_params, f, indent=2)
        print(f"\n=== BEST (Pareto score={best_score:.3f}) ===")
        print(f"  k={best_params['k']}, eta={best_params['eta']}, tau={best_params['tau']}")
        print(f"  marginal={best_params['marginal']:.3f}, pa_std={best_params['pa_std']:.3f}")
        print(f"  extreme={best_params['extreme']:.3f}, width={best_params['width_bps']:.0f} bps")
        print(f"\nTop 5:")
        print(grid_df.head(5).round(3).to_string(index=False))


if __name__ == "__main__":
    main()