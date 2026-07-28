"""Sector-clustered CA-GCP.

When the universe contains ETFs from different sectors (tech, healthcare,
finance, ...), cross-sector correlations are weak (~0.3) while intra-sector
correlations are strong (~0.85). Pooling across sectors injects noise.

This module trains and predicts with **one independent CA-GCP per sector**.
A hybrid variant additionally allows borrowing from the cross-sector universe
when the cross-sector correlation exceeds a threshold.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from ca_gcp.core.pipeline import CAGCPConfig, CAGCPipeline


@dataclass
class SectorCAGCPResult:
    """Concatenated outputs from sector-level predictions."""

    lower: pd.DataFrame
    upper: pd.DataFrame
    half_width: pd.DataFrame
    stress: pd.Series
    sector_stress: dict[str, pd.Series]


def load_sector_map(csv_path: str | Path) -> dict[str, str]:
    """Load code -> sector mapping from CSV."""
    df = pd.read_csv(csv_path)
    if not {"code", "sector"}.issubset(df.columns):
        raise ValueError(f"CSV must have columns code,sector, got {df.columns.tolist()}")
    return dict(zip(df["code"].astype(str), df["sector"].astype(str)))


def build_sector_groups(
    codes: list[str], sector_map: dict[str, str], min_size: int = 2
) -> dict[str, list[str]]:
    """Group ETF codes by sector, filtering out tiny sectors."""
    sector_to_codes: dict[str, list[str]] = {}
    for c in codes:
        sec = sector_map.get(c)
        if sec is None:
            continue
        sector_to_codes.setdefault(sec, []).append(c)
    return {sec: cs for sec, cs in sector_to_codes.items() if len(cs) >= min_size}


def fit_sector_ca_gcp(
    returns_train: pd.DataFrame,
    sectors: dict[str, list[str]],
    config: CAGCPConfig | None = None,
    sector_overrides: dict[str, CAGCPConfig] | None = None,
) -> dict[str, CAGCPipeline]:
    """Train one independent CAGCPipeline per sector.

    Args:
        returns_train: Full training returns (T, N).
        sectors: sector -> list of codes (subset of returns_train.columns).
        config: Default config for all sectors.
        sector_overrides: Optional per-sector CAGCPConfig overrides
            (e.g. smaller k for tiny sectors).
    """
    cfg = config or CAGCPConfig()
    out: dict[str, CAGCPipeline] = {}
    for sec, codes in sectors.items():
        sec_returns = returns_train[codes]
        sec_cfg = (sector_overrides or {}).get(sec, cfg)
        if sec_cfg.k >= len(codes):
            sec_cfg = CAGCPConfig(**{**sec_cfg.__dict__, "k": max(1, len(codes) - 1)})
        pipe = CAGCPipeline(sec_cfg)
        pipe.fit(sec_returns)
        out[sec] = pipe
    return out


def predict_sector_ca_gcp(
    pipelines: dict[str, CAGCPipeline],
    returns_calib: pd.DataFrame,
    returns_test: pd.DataFrame,
) -> SectorCAGCPResult:
    """Predict intervals per sector; concatenate outputs.

    Each pipeline's outputs are stored under a per-sector key in
    `sector_stress`, with the primary `lower` / `upper` / `half_width`
    concatenating only the **first occurrence** of each code (avoiding
    duplicate columns from cross-sector borrowing).
    """
    sector_frames: dict[str, dict[str, pd.DataFrame]] = {}
    sector_stress: dict[str, pd.Series] = {}
    seen_codes: set[str] = set()

    for sec, pipe in pipelines.items():
        sec_codes = [c for c in pipe.codes if c in returns_calib.columns and c in returns_test.columns]
        new_codes = [c for c in sec_codes if c not in seen_codes]
        seen_codes.update(new_codes)
        if not new_codes:
            sector_frames[sec] = {}
            continue
        sec_calib = returns_calib[new_codes]
        sec_test = returns_test[new_codes]
        out = pipe.predict(sec_calib, sec_test)
        sector_frames[sec] = {
            "lower": out["lower"],
            "upper": out["upper"],
            "half_width": out["half_width"],
        }
        sector_stress[sec] = out["stress"]

    lower_dfs = [v["lower"] for v in sector_frames.values() if v]
    upper_dfs = [v["upper"] for v in sector_frames.values() if v]
    hw_dfs = [v["half_width"] for v in sector_frames.values() if v]

    lower = pd.concat(lower_dfs, axis=1) if lower_dfs else pd.DataFrame()
    upper = pd.concat(upper_dfs, axis=1) if upper_dfs else pd.DataFrame()
    hw = pd.concat(hw_dfs, axis=1) if hw_dfs else pd.DataFrame()

    lower = lower.loc[:, ~lower.columns.duplicated()]
    upper = upper.loc[:, ~upper.columns.duplicated()]
    hw = hw.loc[:, ~hw.columns.duplicated()]

    stress_df = pd.concat(sector_stress, axis=1)
    stress = stress_df.mean(axis=1) if not stress_df.empty else pd.Series(dtype=float)

    return SectorCAGCPResult(
        lower=lower,
        upper=upper,
        half_width=hw,
        stress=stress,
        sector_stress=sector_stress,
    )


def fit_sector_hybrid_ca_gcp(
    returns_train: pd.DataFrame,
    sectors: dict[str, list[str]],
    config: CAGCPConfig | None = None,
    cross_sector_threshold: float = 0.5,
) -> dict[str, CAGCPPipeline]:
    """Sector CA-GCP with optional cross-sector neighbor borrowing.

    For each sector, augment the universe with cross-sector codes whose
    correlation with any in-sector asset exceeds `cross_sector_threshold`,
    then train as a single CA-GCP. The resulting pipeline's neighbors may
    include out-of-sector assets, but each sector still trains independently.
    """
    cfg = config or CAGCPConfig()
    corr_full = returns_train.corr()

    out: dict[str, CAGCPipeline] = {}
    for sec, codes in sectors.items():
        in_sec = list(codes)
        cross_codes: list[str] = []
        for c in in_sec:
            for c2 in returns_train.columns:
                if c2 in in_sec or c2 in cross_codes:
                    continue
                if corr_full.at[c, c2] >= cross_sector_threshold:
                    cross_codes.append(c2)
        augmented = in_sec + cross_codes
        sec_returns = returns_train[augmented]
        pipe = CAGCPipeline(cfg)
        pipe.fit(sec_returns)
        out[sec] = pipe
    return out