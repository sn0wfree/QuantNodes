"""Run 5 baselines + CA-GCP on ETF data; cache predictions to results/predictions.parquet."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2"))

from _path import *  # noqa: F401,F403
from ca_gcp import (  # noqa: E402
    CAGCPConfig,
    CAGCPipeline,
    estimate_volatility,
)

DATA_DIR = ROOT / "data" / "high_freq_macro"
OUT_DIR = ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2" / "data" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def baseline_normal_vol(actual: pd.DataFrame, sigma: pd.DataFrame) -> dict:
    half = 1.96 * sigma
    return {"lower": -half, "upper": half, "half_width": half, "stress": pd.Series(0, index=actual.index)}


def baseline_per_asset_cp(
    calib_returns: pd.DataFrame,
    test_returns: pd.DataFrame,
    sigma_test: pd.DataFrame,
    level: float = 0.95,
) -> dict:
    """Per-asset split CP on raw residuals with Gaussian rescaling."""
    residuals_calib = calib_returns.abs()
    thresholds = residuals_calib.quantile(level)
    half = thresholds.reindex(test_returns.columns).values * np.ones(test_returns.shape)
    half = pd.DataFrame(half, index=test_returns.index, columns=test_returns.columns)
    return {"lower": -half, "upper": half, "half_width": half, "stress": pd.Series(0, index=test_returns.index)}


def baseline_vol_cp(
    calib_returns: pd.DataFrame,
    test_returns: pd.DataFrame,
    sigma_calib: pd.DataFrame,
    sigma_test: pd.DataFrame,
    level: float = 0.95,
) -> dict:
    """Vol-CP (VAC-FF): per-asset CP on volatility-normalized scores."""
    scores_calib = calib_returns.abs() / sigma_calib
    thresholds = scores_calib.quantile(level)
    half = pd.DataFrame(
        np.outer(np.ones(len(test_returns)), thresholds.reindex(test_returns.columns).values),
        index=test_returns.index,
        columns=test_returns.columns,
    )
    half = half * sigma_test
    return {"lower": -half, "upper": half, "half_width": half, "stress": pd.Series(0, index=test_returns.index)}


def baseline_global_cp(
    calib_returns: pd.DataFrame,
    test_returns: pd.DataFrame,
    sigma_test: pd.DataFrame,
    level: float = 0.95,
) -> dict:
    """Global CP: single quantile over all (asset, day) pairs."""
    scores = calib_returns.abs().values.flatten()
    q = np.quantile(scores, level)
    half = pd.DataFrame(q, index=test_returns.index, columns=test_returns.columns)
    return {"lower": -half, "upper": half, "half_width": half, "stress": pd.Series(0, index=test_returns.index)}


def run_ca_gcp(
    train: pd.DataFrame,
    calib: pd.DataFrame,
    test: pd.DataFrame,
    cfg: CAGCPConfig,
) -> dict:
    pipeline = CAGCPipeline(cfg)
    pipeline.fit(train)
    return pipeline.predict(calib, test)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-end", default="2021-04-12")
    parser.add_argument("--calib-end", default="2022-04-11")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=60.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    df = pd.read_parquet(DATA_DIR / "v7_6_daily_etf_returns.parquet")
    df = df.dropna(thresh=int(len(df) * 0.7), axis=1)
    df = df.ffill().fillna(0.0)

    train_end = pd.Timestamp(args.train_end)
    calib_end = pd.Timestamp(args.calib_end)
    train = df.loc[:train_end].iloc[:-1]
    calib = df.loc[train_end:calib_end]
    test = df.loc[calib_end:].iloc[1:253]
    print(f"Train: {train.shape}, Calib: {calib.shape}, Test: {test.shape}")

    full = pd.concat([train, calib, test])
    sigma = estimate_volatility(full)
    sigma_calib = sigma.reindex(calib.index)
    sigma_test = sigma.reindex(test.index)

    methods: dict[str, dict] = {}

    methods["Normal-Vol"] = baseline_normal_vol(test, sigma_test)
    methods["PerAsset-CP"] = baseline_per_asset_cp(calib, test, sigma_test, level=1 - args.alpha)
    methods["Vol-CP"] = baseline_vol_cp(calib, test, sigma_calib, sigma_test, level=1 - args.alpha)
    methods["Global-CP"] = baseline_global_cp(calib, test, sigma_test, level=1 - args.alpha)

    cfg = CAGCPConfig(k=args.k, sensitivity_eta=args.eta, recency_tau=args.tau, alpha=args.alpha)
    methods["CA-GCP"] = run_ca_gcp(train, calib, test, cfg)

    rows = []
    for name, m in methods.items():
        m["half_width"].to_parquet(OUT_DIR / f"hw_{name.replace('-', '_')}.parquet")
        m["lower"].to_parquet(OUT_DIR / f"lo_{name.replace('-', '_')}.parquet")
        m["upper"].to_parquet(OUT_DIR / f"up_{name.replace('-', '_')}.parquet")
        rows.append({"method": name})

    pd.DataFrame(rows).to_csv(OUT_DIR / "methods_run.csv", index=False)
    print(f"Saved predictions for {list(methods.keys())} to {OUT_DIR}")


if __name__ == "__main__":
    main()