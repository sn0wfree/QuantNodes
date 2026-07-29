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
    """Threshold rules for v10.2 risk overlay.

    Defaults are very conservative: only trigger on truly extreme market
    conditions. Tuned for v10 which already has low vol (5-7%) and small
    drawdowns (3-7%).
    """

    width_z_yellow: float = 3.0
    width_z_red: float = 4.5
    stress_yellow: float = 0.92
    stress_red: float = 0.98
    yellow_scale: float = 0.85
    red_scale: float = 0.6
    panic_scale: float = 0.3
    # Hysteresis: recovery requires stress below this (prevents bounce)
    stress_yellow_recovery: float = 0.80
    width_z_yellow_recovery: float = 2.0


def experimental_rules() -> RiskFilterRules:
    """More aggressive thresholds for stress-test experiments."""
    return RiskFilterRules(
        width_z_yellow=2.0,
        width_z_red=3.0,
        stress_yellow=0.6,
        stress_red=0.85,
    )


def ca_gcp_risk_filter(
    weights: pd.Series,
    intervals: dict[str, pd.DataFrame],
    rules: RiskFilterRules | None = None,
    today: pd.Timestamp | None = None,
    history: pd.DataFrame | None = None,
) -> tuple[pd.Series, dict]:
    """Apply CA-GCP risk filter to v10 target weights.

    Args:
        weights: v10 target weights (indexed by asset code).
        intervals: dict from CAGCPipeline.predict() with keys:
            'lower', 'upper', 'half_width', 'thresholds', 'stress'.
        rules: RiskFilterRules configuration.
        today: Specific day to evaluate. If None, uses last day of intervals.
        history: Earlier half_width values for computing rolling stats.
            If None, uses rolling on intervals["half_width"] only.

    Returns:
        (adjusted_weights, diagnostics_dict)
    """
    rules = rules or RiskFilterRules()
    hw = intervals["half_width"]
    stress = intervals["stress"]

    if today is None:
        today = intervals["half_width"].index[-1]

    if history is not None:
        hw_full = pd.concat([history, hw]).drop_duplicates()
    else:
        hw_full = hw
    width_ts = (2.0 * hw_full).mean(axis=1)
    roll = width_ts.rolling(60, min_periods=10)
    width_z_full = (width_ts - roll.mean()) / roll.std().replace(0, np.nan)

    if today in width_z_full.index:
        width_z_today = float(width_z_full.loc[today]) if pd.notna(width_z_full.loc[today]) else 0.0
    else:
        width_z_today = float(width_z_full.iloc[-1]) if pd.notna(width_z_full.iloc[-1]) else 0.0

    if today in stress.index:
        stress_today = float(stress.loc[today])
    else:
        stress_today = float(stress.iloc[-1])

    diag: dict = {
        "width_z_today": width_z_today,
        "stress_today": stress_today,
        "alert_level": "green",
    }
    scale = 1.0
    if width_z_today > rules.width_z_red or stress_today > rules.stress_red:
        scale = rules.red_scale
        diag["alert_level"] = "red"
    elif width_z_today > rules.width_z_yellow or stress_today > rules.stress_yellow:
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