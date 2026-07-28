"""Width validators (paper Fig. 5 right)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def width_timeseries(half_width: pd.DataFrame) -> pd.Series:
    return (2.0 * half_width).mean(axis=1)


def width_volatility_correlation(
    half_width: pd.DataFrame, realized_vol: pd.Series
) -> float:
    w = width_timeseries(half_width)
    aligned = pd.concat([w, realized_vol], axis=1).dropna()
    aligned.columns = ["width", "realized_vol"]
    if len(aligned) < 2:
        return 0.0
    return float(aligned["width"].corr(aligned["realized_vol"]))


def width_stability(half_width: pd.DataFrame, window: int = 60) -> float:
    w = width_timeseries(half_width)
    rolling = w.rolling(window, min_periods=10).std() / w.rolling(window, min_periods=10).mean()
    return float(rolling.dropna().mean())