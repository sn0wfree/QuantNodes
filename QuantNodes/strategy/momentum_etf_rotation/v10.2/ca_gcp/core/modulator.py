"""Systemic-stress modulator (Sec. 4.5, Eq. 7-8).

Two market-wide signals combined into a logistic score in [0, 1];
the interval half-width is multiplied by exp(eta * S).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_SIGMOID = lambda x: 1.0 / (1.0 + np.exp(-x))


def compute_systemic_stress(
    returns: pd.DataFrame,
    volatility: pd.DataFrame,
    threshold_sigma: float = 1.5,
) -> pd.Series:
    """S_t = sigmoid(a + b * dispersion_t + c * anomalous_frac_t).

    Centered so that S_t is small (< 0.3) on normal days and spikes
    only on genuinely stressed days. This prevents over-widening
    intervals during ordinary market activity.
    """
    cross_dispersion = returns.std(axis=1)
    anomaly_count = (returns.abs() > (threshold_sigma * volatility)).sum(axis=1)
    anomaly_frac = anomaly_count / returns.shape[1]

    a, b, c = -2.5, 1.0, 4.0
    z = a + b * cross_dispersion / (cross_dispersion.std() + 1e-8) + c * (
        anomaly_frac - anomaly_frac.mean()
    ) / (anomaly_frac.std() + 1e-8)
    s = pd.Series(_SIGMOID(z.values), index=returns.index)
    return s.fillna(0.0)


def apply_modulator(
    half_width: pd.DataFrame,
    stress: pd.Series,
    eta: float = 0.5,
) -> pd.DataFrame:
    """Multiply half-width by exp(eta * S)."""
    factor = np.exp(eta * stress.reindex(half_width.index).fillna(0.0).values)
    return half_width.multiply(factor, axis=0)