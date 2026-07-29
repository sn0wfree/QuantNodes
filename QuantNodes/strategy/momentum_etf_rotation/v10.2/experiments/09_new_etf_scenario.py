"""Scarce-asset (new ETF) validation experiment.

Reproduces Parker & Zhang (2026) Table II: when an asset has only
H recent calibration samples (simulating a newly listed ETF), per-asset
CP is unstable but CA-GCP's neighbor pooling stabilizes coverage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2"))
sys.path.insert(0, str(ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation"))

from _path import *  # noqa: F401,F403
from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import (  # noqa: E402
    CAGCPConfig,
    CAGCPipeline,
    estimate_volatility,
)
from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import (  # noqa: E402
    compute_coverage_metrics,
    quality_dataframe,
    width_bps,
)

DATA_DIR = ROOT / "data" / "high_freq_macro"
OUT_DIR = ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2" / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def per_asset_cp_intervals(calib: pd.DataFrame, test: pd.DataFrame, alpha: float) -> dict:
    q = calib.abs().quantile(1 - alpha)
    half = pd.DataFrame(
        np.tile(q.reindex(test.columns).values, (len(test), 1)),
        index=test.index,
        columns=test.columns,
    )
    return {"lower": -half, "upper": half, "half_width": half}


def vol_cp_intervals(
    calib: pd.DataFrame, test: pd.DataFrame, sigma_calib: pd.DataFrame, sigma_test: pd.DataFrame, alpha: float
) -> dict:
    scores = calib.abs() / sigma_calib
    q = scores.quantile(1 - alpha)
    half = sigma_test.multiply(q.reindex(test.columns), axis=1)
    return {"lower": -half, "upper": half, "half_width": half}


def truncate_calib(calib: pd.DataFrame, scarce_codes: list[str], H: int) -> pd.DataFrame:
    """Truncate calibration set to last H days for scarce assets, drop earlier scores."""
    out = calib.copy()
    for code in scarce_codes:
        if code not in out.columns:
            continue
        nan_idx = out.index[:-H]
        out.loc[nan_idx, code] = np.nan
    return out.ffill().dropna(how="all")


def evaluate(
    name: str,
    actual: pd.DataFrame,
    intervals: dict,
    extreme_vol: pd.Series,
    scarce_codes: list[str],
) -> dict:
    m = compute_coverage_metrics(actual, intervals["lower"], intervals["upper"], extreme_vol)
    scarce_cov_per_asset = []
    for code in scarce_codes:
        if code in actual.columns:
            lo_c = intervals["lower"][code]
            up_c = intervals["upper"][code]
            cov = ((actual[code] >= lo_c) & (actual[code] <= up_c)).mean()
            scarce_cov_per_asset.append(float(cov))
    scarce_arr = np.array(scarce_cov_per_asset)
    m["scarce_cov"] = float(scarce_arr.mean())
    m["scarce_cov_std"] = float(scarce_arr.std())
    m["width_bps"] = width_bps(intervals["half_width"])
    m["method"] = name
    return m


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "v7_6_daily_etf_returns.parquet")
    df = df.dropna(thresh=int(len(df) * 0.7), axis=1).ffill().fillna(0.0)

    use_cols = df.columns[:20]
    df = df[use_cols]

    train_end = pd.Timestamp("2021-04-12")
    calib_end = pd.Timestamp("2022-04-11")
    train = df.loc[:train_end].iloc[:-1]
    calib = df.loc[train_end:calib_end]
    test = df.loc[calib_end:].iloc[1:253]

    rng = np.random.default_rng(42)
    scarce_codes = list(rng.choice(test.columns, size=5, replace=False))

    full_returns = pd.concat([train, calib, test])
    sigma = estimate_volatility(full_returns)
    sigma_calib = sigma.reindex(calib.index)
    sigma_test = sigma.reindex(test.index)

    cross_dispersion = test.std(axis=1)
    extreme_vol = cross_dispersion > cross_dispersion.quantile(0.9)

    rows = []
    quality_dfs = []
    pipe = None
    for H in (20, 40, 80):
        calib_trunc = truncate_calib(calib, scarce_codes, H)

        pa = per_asset_cp_intervals(calib_trunc, test, alpha=0.05)
        rows.append(evaluate("PerAsset-CP", test, pa, extreme_vol, scarce_codes))

        vc = vol_cp_intervals(calib_trunc, test, sigma_calib, sigma_test, alpha=0.05)
        rows.append(evaluate("Vol-CP", test, vc, extreme_vol, scarce_codes))

        if pipe is None:
            cfg = CAGCPConfig(k=8)
            pipe = CAGCPipeline(cfg)
            pipe.fit(train)
        out_full = pipe.predict(calib_trunc, test)
        rows.append(evaluate("CA-GCP", test, out_full, extreme_vol, scarce_codes))

        if H == 40:
            quality_dfs.append((H, quality_dataframe(pipe)))

    table = pd.DataFrame(rows)[["method", "marginal", "pa_std", "scarce_cov", "scarce_cov_std", "width_bps", "extreme"]]
    print("\n=== Scarcity Table (paper Table II) ===")
    print(table.to_string(index=False))
    table.to_csv(OUT_DIR / "scarce_table.csv", index=False)

    if quality_dfs:
        H, qdf = quality_dfs[0]
        qdf_out = qdf.copy()
        qdf_out["H"] = H
        qdf_out.to_csv(OUT_DIR / "borrow_recommendations.csv")
        print(f"\n=== Borrow Recommendations (H={H}) ===")
        print(qdf_out.round(3).to_string())


if __name__ == "__main__":
    main()