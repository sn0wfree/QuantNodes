"""Early warning evaluators (paper Sec. 5.6 + our addition)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_warnings(
    stress: pd.Series,
    half_width: pd.DataFrame,
    width_z_thresh: float = 2.0,
    stress_thresh: float = 0.6,
    mode: str = "and",
) -> pd.DataFrame:
    width_z = (width_timeseries(half_width) - width_timeseries(half_width).rolling(60, min_periods=10).mean()) / width_timeseries(half_width).rolling(60, min_periods=10).std().replace(0, np.nan)
    width_alert = width_z > width_z_thresh
    stress_alert = stress > stress_thresh
    if mode == "and":
        fired = (width_alert.fillna(False) & stress_alert.fillna(False)).astype(int)
    else:
        fired = (width_alert.fillna(False) | stress_alert.fillna(False)).astype(int)
    return pd.DataFrame(
        {"width_z": width_z, "stress": stress, "fired": fired}
    )


def width_timeseries(half_width: pd.DataFrame) -> pd.Series:
    return (2.0 * half_width).mean(axis=1)


def evaluate_against_events(
    fired: pd.Series,
    events: list[dict],
    horizon: int = 5,
    drawdown_thresh: float = 0.03,
    returns_for_eval: pd.Series | None = None,
) -> pd.DataFrame:
    rows = []
    for ev in events:
        ev_date = pd.Timestamp(ev["date"])
        post = returns_for_eval.loc[ev_date : ev_date + pd.Timedelta(days=horizon * 2)] if returns_for_eval is not None else None
        if post is not None and len(post) >= horizon:
            cumret = (1 + post).prod() - 1
            realized_dd = float(cumret.min())
        else:
            realized_dd = float("nan")

        prior_window = fired.loc[ev_date - pd.Timedelta(days=20) : ev_date]
        lead_days = None
        fired_idx = prior_window.index[prior_window.fillna(0) == 1]
        if len(fired_idx):
            lead_days = int((ev_date - fired_idx[-1]).days)

        rows.append(
            {
                "event": ev["name"],
                "date": ev["date"],
                "lead_days": lead_days,
                "realized_dd_5d": realized_dd,
                "warned": lead_days is not None,
            }
        )
    return pd.DataFrame(rows)