"""Coverage validators (paper Sec. 5.3, Table I)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _is_covered(actual: pd.DataFrame, lower: pd.DataFrame, upper: pd.DataFrame) -> pd.DataFrame:
    return ((actual >= lower) & (actual <= upper)).astype(float)


def compute_coverage_metrics(
    actual: pd.DataFrame,
    lower: pd.DataFrame,
    upper: pd.DataFrame,
    extreme_vol: pd.Series | None = None,
) -> dict[str, float]:
    """Marginal + per-asset + worst-decile + extreme-day coverage at 95% target."""
    covered = _is_covered(actual, lower, upper)
    per_asset = covered.mean(axis=0)

    marginal = float(covered.values.mean())
    pa_std = float(per_asset.std())
    sorted_pa = np.sort(per_asset.values)
    worst10 = float(sorted_pa[: max(1, len(sorted_pa) // 10)].mean())
    worst_min = float(per_asset.min())

    extr = float("nan")
    if extreme_vol is not None:
        mask = extreme_vol.reindex(covered.index).fillna(False).astype(bool)
        if mask.any():
            extr = float(covered.loc[mask].values.mean())

    return {
        "marginal": marginal,
        "pa_std": pa_std,
        "worst10": worst10,
        "min": worst_min,
        "extreme": extr,
    }


def width_bps(half_width: pd.DataFrame) -> float:
    """Mean interval width in basis points (decimal -> bps)."""
    width = 2.0 * half_width
    return float(width.mean().mean() * 1e4)