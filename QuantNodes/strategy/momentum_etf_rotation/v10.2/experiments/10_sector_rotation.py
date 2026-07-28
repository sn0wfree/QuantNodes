"""Sector-clustered CA-GCP experiment.

Compares three approaches:
1. Global CA-GCP (cross-sector pooling)
2. Sector CA-GCP (intra-sector pooling only)
3. Sector + Correlation Hybrid (intra + highly-correlated cross-sector)

Reports per-sector coverage and width to highlight the trade-off between
coverage tightness and cross-sector independence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2"))
sys.path.insert(0, str(ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation"))

from _path import *  # noqa: F401,F403
from ca_gcp import CAGCPConfig, CAGCPipeline  # noqa: E402
from ca_gcp.cluster import (  # noqa: E402
    build_sector_groups,
    fit_sector_ca_gcp,
    fit_sector_hybrid_ca_gcp,
    load_sector_map,
    predict_sector_ca_gcp,
)
from ca_gcp.validators import compute_coverage_metrics, width_bps  # noqa: E402

DATA_DIR = ROOT / "data" / "high_freq_macro"
V102_DIR = ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2"
SECTOR_MAP = V102_DIR / "data" / "etf_sector_map.csv"
OUT_DIR = V102_DIR / "data" / "results"


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "v7_6_daily_etf_returns.parquet")
    df = df.dropna(thresh=int(len(df) * 0.7), axis=1).ffill().fillna(0.0)

    sector_map = load_sector_map(SECTOR_MAP)
    sectors = build_sector_groups(list(df.columns), sector_map, min_size=3)
    print(f"Sectors: {[(s, len(c)) for s, c in sectors.items()]}")

    target_sectors = {"科技", "宽基", "周期"}
    sectors = {s: c for s, c in sectors.items() if s in target_sectors}
    print(f"Using: {[(s, len(c)) for s, c in sectors.items()]}")

    train_end = pd.Timestamp("2021-04-12")
    calib_end = pd.Timestamp("2022-04-11")
    train = df.loc[:train_end].iloc[:-1]
    calib = df.loc[train_end:calib_end]
    test = df.loc[calib_end:].iloc[1:253]

    cfg = CAGCPConfig(k=4)

    global_pipe = CAGCPipeline(cfg)
    global_pipe.fit(train)
    global_out = global_pipe.predict(calib, test)

    sector_pipes = fit_sector_ca_gcp(train, sectors, cfg)
    sector_out = predict_sector_ca_gcp(sector_pipes, calib, test)

    hybrid_pipes = fit_sector_hybrid_ca_gcp(
        train, sectors, cfg, cross_sector_threshold=0.5
    )
    hybrid_out = predict_sector_ca_gcp(hybrid_pipes, calib, test)

    cross_dispersion = test.std(axis=1)
    extreme_vol = cross_dispersion > cross_dispersion.quantile(0.9)

    rows = []

    rows.append(
        {
            "method": "Global CA-GCP",
            "scope": "all",
            **{k: v for k, v in compute_coverage_metrics(test, global_out["lower"], global_out["upper"], extreme_vol).items()},
            "width_bps": width_bps(global_out["half_width"]),
        }
    )

    for sec, codes in sectors.items():
        sec_codes_present = [c for c in codes if c in sector_out.lower.columns]
        common_idx = test.index.intersection(sector_out.lower.index)
        sec_test = test.loc[common_idx, sec_codes_present]
        sec_lo = sector_out.lower.loc[common_idx, sec_codes_present]
        sec_up = sector_out.upper.loc[common_idx, sec_codes_present]
        sec_hw = sector_out.half_width.loc[common_idx, sec_codes_present]
        m = compute_coverage_metrics(sec_test, sec_lo, sec_up, extreme_vol)
        rows.append(
            {
                "method": "Sector CA-GCP",
                "scope": sec,
                **m,
                "width_bps": width_bps(sec_hw),
            }
        )

        hy_codes_present = [c for c in codes if c in hybrid_out.lower.columns]
        if not hy_codes_present:
            continue
        hy_test = test.loc[common_idx, hy_codes_present]
        hy_lo = hybrid_out.lower.loc[common_idx, hy_codes_present]
        hy_up = hybrid_out.upper.loc[common_idx, hy_codes_present]
        hy_hw = hybrid_out.half_width.loc[common_idx, hy_codes_present]
        m_hy = compute_coverage_metrics(hy_test, hy_lo, hy_up, extreme_vol)
        rows.append(
            {
                "method": "Hybrid Sector CA-GCP",
                "scope": sec,
                **m_hy,
                "width_bps": width_bps(hy_hw),
            }
        )

    df_out = pd.DataFrame(rows)
    print("\n=== Sector CA-GCP Comparison ===")
    print(df_out.round(3).to_string(index=False))
    df_out.to_csv(OUT_DIR / "sector_comparison.csv", index=False)


if __name__ == "__main__":
    main()