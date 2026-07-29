"""Theoretical coverage gap bound experiment (paper Eq. 9).

Computes per-asset theoretical coverage gap bound via TV distance,
compares against empirical coverage gaps from the test set.
"""
from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2"))

_V102_INIT = ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2" / "__init__.py"
_spec = _ilu.spec_from_file_location("v10_2_module", _V102_INIT)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
CAGCPipeline = _mod.CAGCPipeline
load_calibrated_config = _mod.load_calibrated_config

from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import (  # noqa: E402
    compute_coverage_metrics,
    theoretical_coverage_bound,
    total_variation_distance_ecdf,
)

DATA_DIR = ROOT / "data" / "high_freq_macro"
OUT_DIR = ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2" / "data" / "results"


def tv_heatmap(pipeline: CAGCPipeline, scores_calib: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise TV distance matrix."""
    codes = pipeline.codes
    n = len(codes)
    arr = np.zeros((n, n))
    for i in range(n):
        si = scores_calib[codes[i]].dropna().values
        for j in range(n):
            sj = scores_calib[codes[j]].dropna().values
            arr[i, j] = total_variation_distance_ecdf(si, sj)
    return pd.DataFrame(arr, index=codes, columns=codes)


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "v7_6_daily_etf_returns.parquet")
    df = df.dropna(thresh=int(len(df) * 0.7), axis=1).ffill().fillna(0.0)

    train_end = pd.Timestamp("2021-04-12")
    calib_end = pd.Timestamp("2022-04-11")
    val_end = pd.Timestamp("2023-04-12")

    train = df.loc[:train_end].iloc[:-1]
    calib = df.loc[train_end:calib_end]
    val = df.loc[calib_end:val_end].iloc[1:]

    cfg = load_calibrated_config()
    pipe = CAGCPipeline(cfg)
    pipe.fit(train)

    out = pipe.predict_fast(calib, val)

    cross_dispersion = val.std(axis=1)
    extreme_mask = cross_dispersion > cross_dispersion.quantile(0.9)
    metrics = compute_coverage_metrics(val, out["lower"], out["upper"], extreme_mask)
    per_asset_cov = ((val >= out["lower"]) & (val <= out["upper"])).mean(axis=0)

    print(f"Empirical marginal: {metrics['marginal']:.3f}")
    print(f"Empirical per-asset mean: {per_asset_cov.mean():.3f}")
    print(f"Empirical per-asset min: {per_asset_cov.min():.3f}")
    print(f"Empirical per-asset max: {per_asset_cov.max():.3f}")

    from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import estimate_volatility
    full_returns = pd.concat([calib, val])
    sigma = estimate_volatility(full_returns)
    sigma_calib = sigma.reindex(calib.index)
    calib_resid = calib.abs()
    scores_calib = calib_resid / sigma_calib

    print(f"\nComputing TV bounds for {len(pipe.codes)} assets...")
    bounds = theoretical_coverage_bound(pipe, scores_calib)
    bounds["empirical_gap"] = (1.0 - per_asset_cov).reindex(bounds.index)
    bounds["ratio"] = bounds["bound"] / bounds["empirical_gap"].replace(0, np.nan)
    bounds["bound_satisfied"] = bounds["bound"] >= bounds["empirical_gap"]

    print(f"\n=== Theoretical vs Empirical Coverage Gaps ===")
    print(f"  Bound: mean={bounds['bound'].mean():.4f}, max={bounds['bound'].max():.4f}")
    print(f"  Empirical gap: mean={bounds['empirical_gap'].mean():.4f}, max={bounds['empirical_gap'].max():.4f}")
    print(f"  Ratio (bound/empirical): mean={bounds['ratio'].mean():.2f}x, min={bounds['ratio'].min():.2f}x")
    print(f"  Bound satisfied for {(bounds['bound_satisfied']).sum()}/{len(bounds)} assets")

    bounds.to_csv(OUT_DIR / "theoretical_bound.csv")

    tv_mat = tv_heatmap(pipe, scores_calib)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(tv_mat.values, cmap="viridis", aspect="auto", vmin=0, vmax=tv_mat.values.max())
    plt.colorbar(im, ax=ax, label="TV distance")
    ax.set_xticks(range(len(tv_mat.columns)))
    ax.set_xticklabels(tv_mat.columns, rotation=90, fontsize=6)
    ax.set_yticks(range(len(tv_mat.index)))
    ax.set_yticklabels(tv_mat.index, fontsize=6)
    ax.set_title(f"Pairwise TV distance of normalized scores ({len(tv_mat)} assets)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tv_distance_heatmap.png", dpi=120)
    print(f"\nSaved TV distance heatmap to {OUT_DIR / 'tv_distance_heatmap.png'}")

    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(bounds["bound"], bins=20, alpha=0.7, label="Theoretical bound")
    axes[0].hist(bounds["empirical_gap"].dropna(), bins=20, alpha=0.7, label="Empirical gap")
    axes[0].set_xlabel("Coverage gap")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].set_title("Distribution comparison")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(bounds["bound"], bounds["empirical_gap"], alpha=0.6)
    lim = max(bounds["bound"].max(), bounds["empirical_gap"].max())
    axes[1].plot([0, lim], [0, lim], "k--", alpha=0.5, label="bound = empirical")
    axes[1].set_xlabel("Theoretical bound")
    axes[1].set_ylabel("Empirical gap")
    axes[1].set_title("Per-asset: bound vs empirical")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "theoretical_bound_vs_empirical.png", dpi=120)
    print(f"Saved scatter to {OUT_DIR / 'theoretical_bound_vs_empirical.png'}")


if __name__ == "__main__":
    main()