"""v10.2 CA-GCP risk-filter hook.

Applies CA-GCP prediction intervals as a position-sizing adjustment:
  - Width explosion (> threshold) → reduce position size
  - System pressure (γ > 0.6) → pause new entries / scale down
  - Neighbor divergence (> threshold) → reduce most volatile legs

The function is independent of v10 source; it accepts weights and returns
adjusted weights, preserving v10's relative ranking.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import sys
from pathlib import Path

_V102 = Path(__file__).resolve().parent.parent
if str(_V102) not in sys.path:
    sys.path.insert(0, str(_V102))

from ca_gcp.core import CAGCPConfig, CAGCPipeline  # noqa: E402


@dataclass
class RiskFilterRules:
    """Threshold rules for v10.2 risk overlay."""

    width_z_yellow: float = 2.0
    width_z_red: float = 3.0
    stress_yellow: float = 0.6
    stress_red: float = 0.85
    yellow_scale: float = 0.7
    red_scale: float = 0.4
    panic_scale: float = 0.0


def ca_gcp_risk_filter(
    weights: pd.Series,
    intervals: dict[str, pd.DataFrame],
    rules: RiskFilterRules | None = None,
) -> tuple[pd.Series, dict]:
    """Apply CA-GCP risk filter to v10 target weights.

    Args:
        weights: v10 target weights (indexed by asset code).
        intervals: dict from CAGCPipeline.predict() with keys:
            'lower', 'upper', 'half_width', 'thresholds', 'stress'.
        rules: RiskFilterRules configuration.

    Returns:
        (adjusted_weights, diagnostics_dict)
    """
    rules = rules or RiskFilterRules()
    hw = intervals["half_width"]
    stress = intervals["stress"]

    width_ts = (2.0 * hw).mean(axis=1)
    roll = width_ts.rolling(60, min_periods=10)
    width_z = (width_ts - roll.mean()) / roll.std().replace(0, np.nan)

    diag: dict = {
        "width_z_today": float(width_z.iloc[-1]) if pd.notna(width_z.iloc[-1]) else 0.0,
        "stress_today": float(stress.iloc[-1]),
        "alert_level": "green",
    }
    scale = 1.0
    if width_z.iloc[-1] > rules.width_z_red or stress.iloc[-1] > rules.stress_red:
        scale = rules.red_scale
        diag["alert_level"] = "red"
    elif width_z.iloc[-1] > rules.width_z_yellow or stress.iloc[-1] > rules.stress_yellow:
        scale = rules.yellow_scale
        diag["alert_level"] = "yellow"

    diag["applied_scale"] = scale
    adjusted = weights * scale

    if scale < 1.0:
        residual = (1.0 - adjusted.sum()) if adjusted.sum() < 1.0 else 0.0
        if residual > 0:
            weights_min = weights.abs().idxmin()
            adjusted[weights_min] += residual

    return adjusted, diag


def build_v10_2_pipeline(
    returns_history: pd.DataFrame,
    config: CAGCPConfig | None = None,
) -> CAGCPipeline:
    """Build a fresh CA-GCP pipeline from training returns."""
    pipe = CAGCPipeline(config)
    pipe.fit(returns_history)
    return pipe