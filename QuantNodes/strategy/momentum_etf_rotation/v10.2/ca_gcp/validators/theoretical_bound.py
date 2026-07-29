"""Theoretical coverage gap bound via TV distance (paper Eq. 9).

Implements the Barber et al. (2023) bound:
    |coverage_gap(v)| ≤ Σ_{v' ∈ N(v)} w_{v',v} × TV(P_{s̃_{v'}}, P_{s̃_v})

TV distance is estimated via empirical CDF differences (Q4=c, the most
theoretically rigorous of the three options).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core.pipeline import CAGCPipeline


@dataclass
class TheoreticalBound:
    """Per-asset theoretical coverage gap bound."""

    code: str
    n_neighbors: int
    max_tv: float
    weighted_tv_sum: float
    bound: float

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "n_neighbors": self.n_neighbors,
            "max_tv": self.max_tv,
            "weighted_tv_sum": self.weighted_tv_sum,
            "bound": self.bound,
        }


def total_variation_distance_ecdf(scores_p: np.ndarray, scores_q: np.ndarray) -> float:
    """TV distance via empirical CDF differences.

    TV(P, Q) = sup_x |F_P(x) - F_Q(x)|
             ≈ max over sorted unique points of |CDF_P - CDF_Q|

    Args:
        scores_p: 1D array of samples from P.
        scores_q: 1D array of samples from Q.

    Returns:
        TV distance estimate in [0, 1].
    """
    p = np.sort(np.asarray(scores_p, dtype=float))
    q = np.sort(np.asarray(scores_q, dtype=float))

    n_p = len(p)
    n_q = len(q)
    if n_p == 0 or n_q == 0:
        return 1.0

    all_pts = np.unique(np.concatenate([p, q]))
    F_p = np.searchsorted(p, all_pts, side="right") / n_p
    F_q = np.searchsorted(q, all_pts, side="right") / n_q

    return float(np.max(np.abs(F_p - F_q)))


def theoretical_coverage_bound(
    pipeline: CAGCPipeline,
    scores_calib: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Per-asset theoretical coverage gap bound (paper Eq. 9).

    For each target asset v:
        bound(v) = Σ_{v' ∈ N(v)} w_{v',v} × TV(P_{s̃_{v'}}, P_{s̃_v})

    Args:
        pipeline: Fitted CAGCPipeline (provides corr_matrix + neighbors).
        scores_calib: (T_calib, N) normalized nonconformity scores.
        alpha: 1 - target_coverage (unused for bound, kept for API parity).

    Returns:
        DataFrame with columns: code, n_neighbors, max_tv, weighted_tv_sum, bound.
    """
    if pipeline.corr_matrix is None:
        raise ValueError("Pipeline must be fitted before computing theoretical bound.")

    p = pipeline.config.sharpness_p
    rows: list[dict] = []

    target_scores_arr = []
    for code in pipeline.codes:
        s = scores_calib[code].dropna().values
        target_scores_arr.append(s)

    for v_idx, code in enumerate(pipeline.codes):
        nbr_idx = pipeline.neighbors[v_idx]
        nbr_idx = [i for i in nbr_idx if i != v_idx]
        if not nbr_idx:
            rows.append(
                {
                    "code": code,
                    "n_neighbors": 0,
                    "max_tv": 0.0,
                    "weighted_tv_sum": 0.0,
                    "bound": 0.0,
                }
            )
            continue

        target_s = target_scores_arr[v_idx]
        weighted_tv_sum = 0.0
        max_tv = 0.0

        for src_idx in nbr_idx:
            src_s = target_scores_arr[src_idx]
            tv = total_variation_distance_ecdf(target_s, src_s)
            max_tv = max(max_tv, tv)

            corr_v = max(float(pipeline.corr_matrix[v_idx, src_idx]), 0.0)
            if corr_v <= 0:
                corr_v = 1e-6
            w = corr_v ** p
            weighted_tv_sum += w * tv

        bound = weighted_tv_sum / max(len(nbr_idx), 1)
        rows.append(
            {
                "code": code,
                "n_neighbors": len(nbr_idx),
                "max_tv": max_tv,
                "weighted_tv_sum": weighted_tv_sum,
                "bound": bound,
            }
        )

    return pd.DataFrame(rows).set_index("code")


def compare_bound_to_empirical(
    bounds: pd.DataFrame,
    empirical_gaps: pd.Series,
) -> pd.DataFrame:
    """Compare theoretical bounds against empirical coverage gaps.

    Args:
        bounds: DataFrame from theoretical_coverage_bound (indexed by code).
        empirical_gaps: Series of empirical (target - realized) coverage gaps, indexed by code.

    Returns:
        DataFrame with columns: code, bound, empirical_gap, ratio (bound / empirical_gap).
    """
    common = bounds.index.intersection(empirical_gaps.index)
    df = pd.DataFrame(
        {
            "code": list(common),
            "bound": [float(bounds.loc[c, "bound"]) for c in common],
            "empirical_gap": [float(empirical_gaps[c]) for c in common],
        }
    )
    df["ratio"] = df["bound"] / df["empirical_gap"].replace(0, np.nan)
    df["bound_satisfied"] = df["bound"] >= df["empirical_gap"]
    return df.set_index("code")