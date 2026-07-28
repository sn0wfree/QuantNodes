"""Volatility estimation (Sec. 4.3, Eq. 4).

Exponentially weighted moving average of squared returns blended with
20-day realized volatility, with a small epsilon to stabilize the ratio.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_volatility(
    returns: pd.DataFrame,
    ewma_span: int = 60,
    realized_window: int = 20,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    """Per-asset volatility estimate.

    σ_v,t = sqrt(blend(EWMA(r²), σ_realized_20d)) + epsilon
    """
    ewma_var = returns.pow(2).ewm(span=ewma_span, adjust=False).mean()
    ewma_vol = np.sqrt(ewma_var)

    realized_vol = returns.rolling(window=realized_window, min_periods=5).std()

    blend = 0.5 * ewma_vol + 0.5 * realized_vol
    sigma = np.sqrt(np.maximum(blend.pow(2), 0.0))
    sigma = sigma.bfill().fillna(0.0)
    sigma = sigma + epsilon

    return sigma