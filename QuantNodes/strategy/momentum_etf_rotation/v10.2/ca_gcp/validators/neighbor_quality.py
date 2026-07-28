"""Borrow-strength scoring for cross-asset calibration pooling.

Implements the scarcity-aware neighbor quality assessment from
Parker & Zhang (2026) Sec. 5.4 Table II.

When an asset has few calibration samples (e.g. a newly listed ETF),
pooling neighbors' scores stabilizes coverage. The decision of
*how much* to borrow depends on neighbor correlation quality.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.pipeline import CAGCPipeline


@dataclass
class NeighborQuality:
    """Borrow-strength score for one target asset."""

    target: str
    target_idx: int
    n_neighbors: int
    weighted_corr_sum: float
    mean_corr: float
    effective_sample_size: float
    borrow_recommendation: str

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "n_neighbors": self.n_neighbors,
            "weighted_corr_sum": self.weighted_corr_sum,
            "mean_corr": self.mean_corr,
            "effective_sample_size": self.effective_sample_size,
            "borrow_recommendation": self.borrow_recommendation,
        }


def compute_neighbor_quality(
    pipeline: CAGCPipeline,
    target_idx: int,
    calib_days: int = 252,
) -> NeighborQuality:
    """Score neighbor-borrow quality for target asset.

    Args:
        pipeline: A fitted CAGCPipeline (provides corr_matrix + neighbors).
        target_idx: Index into pipeline.codes.
        calib_days: Length of the calibration window for the target itself.

    Returns:
        NeighborQuality with the borrow recommendation.
    """
    if pipeline.corr_matrix is None:
        raise ValueError("Pipeline must be fitted before computing neighbor quality.")

    nbr_idx = pipeline.neighbors[target_idx]
    nbr_idx = [i for i in nbr_idx if i != target_idx]
    if not nbr_idx:
        return NeighborQuality(
            target=pipeline.codes[target_idx],
            target_idx=target_idx,
            n_neighbors=0,
            weighted_corr_sum=0.0,
            mean_corr=0.0,
            effective_sample_size=float(calib_days),
            borrow_recommendation="weak",
        )

    p = pipeline.config.sharpness_p
    corrs = np.array([max(float(pipeline.corr_matrix[target_idx, i]), 0.0) for i in nbr_idx])
    weighted = (corrs ** p).sum()
    mean_corr = float(corrs.mean())

    borrowed_days = calib_days * len(nbr_idx) * (weighted / max(len(nbr_idx), 1))
    ess = calib_days + borrowed_days

    if weighted >= 5.0:
        rec = "strong"
    elif weighted >= 2.0:
        rec = "moderate"
    else:
        rec = "weak"

    return NeighborQuality(
        target=pipeline.codes[target_idx],
        target_idx=target_idx,
        n_neighbors=len(nbr_idx),
        weighted_corr_sum=float(weighted),
        mean_corr=mean_corr,
        effective_sample_size=float(ess),
        borrow_recommendation=rec,
    )


def recommend_borrow_strategy(
    pipeline: CAGCPipeline,
    calib_days_per_asset: dict[str, int] | None = None,
    strong_threshold: float = 5.0,
    moderate_threshold: float = 2.0,
) -> dict[str, str]:
    """Return per-asset borrow recommendation (strong / moderate / weak).

    Args:
        pipeline: Fitted CAGCPipeline.
        calib_days_per_asset: Optional dict mapping code -> calib length.
            Defaults to 252 for all assets.
        strong_threshold: weighted_corr_sum above this -> "strong".
        moderate_threshold: weighted_corr_sum above this -> "moderate".
    """
    if calib_days_per_asset is None:
        calib_days_per_asset = {c: 252 for c in pipeline.codes}

    out = {}
    for v_idx, code in enumerate(pipeline.codes):
        nq = compute_neighbor_quality(pipeline, v_idx, calib_days_per_asset.get(code, 252))
        if nq.weighted_corr_sum >= strong_threshold:
            out[code] = "strong"
        elif nq.weighted_corr_sum >= moderate_threshold:
            out[code] = "moderate"
        else:
            out[code] = "weak"
    return out


def quality_dataframe(
    pipeline: CAGCPipeline,
    calib_days_per_asset: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with one row per asset summarizing borrow quality."""
    if calib_days_per_asset is None:
        calib_days_per_asset = {c: 252 for c in pipeline.codes}

    rows = []
    for v_idx in range(len(pipeline.codes)):
        nq = compute_neighbor_quality(pipeline, v_idx, calib_days_per_asset.get(pipeline.codes[v_idx], 252))
        rows.append(nq.to_dict())
    return pd.DataFrame(rows).set_index("target")