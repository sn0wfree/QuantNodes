"""Early-warning evaluation against known market stress events."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2"))

from _path import *  # noqa: F401,F403
from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import detect_warnings, evaluate_against_events  # noqa: E402

DATA_DIR = ROOT / "data" / "high_freq_macro"
OUT_DIR = ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2" / "data" / "results"


def main() -> None:
    actual_full = pd.read_parquet(DATA_DIR / "v7_6_daily_etf_returns.parquet").dropna(thresh=int(2058 * 0.7), axis=1).ffill().fillna(0.0)
    hw = pd.read_parquet(OUT_DIR / "hw_CA_GCP.parquet")
    actual = actual_full.loc[hw.index, hw.columns]
    returns_for_eval = actual.mean(axis=1)

    hw = pd.read_parquet(OUT_DIR / "hw_CA_GCP.parquet")
    lo = pd.read_parquet(OUT_DIR / "lo_CA_GCP.parquet")
    up = pd.read_parquet(OUT_DIR / "up_CA_GCP.parquet")

        from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import compute_systemic_stress, estimate_volatility

    sigma = estimate_volatility(actual)
    stress = compute_systemic_stress(actual, sigma)

    modes = ["and", "or"]
    rows = []
    fig, axes = plt.subplots(len(modes), 1, figsize=(12, 4 * len(modes)), sharex=True)
    if len(modes) == 1:
        axes = [axes]
    for ax, mode in zip(axes, modes):
        fired_df = detect_warnings(stress, hw, mode=mode)
        for col in ["width_z", "stress"]:
            ax.plot(fired_df.index, fired_df[col].values, label=col, alpha=0.7)
        for d in fired_df.index[fired_df["fired"] == 1]:
            ax.axvline(d, color="red", alpha=0.4)
        ax.set_title(f"Early warning signals (mode={mode}, red lines = fired)")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "early_warning_signals.png", dpi=120)

    events = [
        {"name": "A-share_底部_10月", "date": "2022-10-31"},
        {"name": "A-share_底部_11月初", "date": "2022-11-01"},
        {"name": "硅谷银行_3月", "date": "2023-03-13"},
        {"name": "AI主线_切换", "date": "2023-04-10"},
    ]
    fired_df = detect_warnings(stress, hw, mode="and")
    eval_df = evaluate_against_events(
        fired=fired_df["fired"],
        events=events,
        horizon=5,
        drawdown_thresh=0.03,
        returns_for_eval=returns_for_eval,
    )
    eval_df.to_csv(OUT_DIR / "early_warning_log.csv", index=False)
    print(eval_df.to_string(index=False))


if __name__ == "__main__":
    main()