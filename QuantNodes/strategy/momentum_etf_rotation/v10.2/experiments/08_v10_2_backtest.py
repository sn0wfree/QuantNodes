"""v10.2 mock backtest: CA-GCP risk filter applied to mock momentum signal."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2"))
sys.path.insert(0, str(ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation"))

from _path import *  # noqa: F401,F403
from integration import RiskFilterRules, ca_gcp_risk_filter  # noqa: E402

DATA_DIR = ROOT / "data" / "high_freq_macro"
OUT_DIR = ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2" / "data" / "results"


def compute_metrics(nav: pd.Series, name: str) -> dict:
    ret = nav.pct_change().fillna(0.0)
    ann_ret = (1 + ret).prod() ** (252 / len(ret)) - 1
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cummax = nav.cummax()
    dd = nav / cummax - 1.0
    max_dd = float(dd.min())
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else 0.0
    return {
        "strategy": name,
        "ann_return": float(ann_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_dd": max_dd,
        "calmar": float(calmar),
    }


def mock_v10_signal(history: pd.DataFrame) -> pd.Series:
    mom_60 = (1 + history.tail(60)).prod() - 1
    top5 = mom_60.nlargest(5)
    weights = pd.Series(0.0, index=history.columns)
    weights[top5.index] = 1.0 / 5.0
    return weights


def run() -> None:
    df = pd.read_parquet(DATA_DIR / "v7_6_daily_etf_returns.parquet")
    df = df.dropna(thresh=int(len(df) * 0.7), axis=1).ffill().fillna(0.0)

    hw = pd.read_parquet(OUT_DIR / "hw_CA_GCP.parquet")
    lo = pd.read_parquet(OUT_DIR / "lo_CA_GCP.parquet")
    up = pd.read_parquet(OUT_DIR / "up_CA_GCP.parquet")

    test = df.loc[hw.index, hw.columns]
    print(f"Test: {test.shape}")

    from ca_gcp.core.modulator import compute_systemic_stress
    from ca_gcp.core.volatility import estimate_volatility
    sigma = estimate_volatility(test)
    stress_series = compute_systemic_stress(test, sigma)

    nav_v10 = pd.Series(1.0, index=test.index)
    nav_v10_2 = pd.Series(1.0, index=test.index)
    diag_rows = []
    prev_v10 = 1.0
    prev_v10_2 = 1.0

    for t in test.index:
        history = df.loc[: t - pd.Timedelta(days=1)].tail(252)
        if len(history) < 60:
            continue
        w_v10 = mock_v10_signal(history)
        realized = test.loc[t]

        intervals_t = {
            "lower": lo.loc[[t]],
            "upper": up.loc[[t]],
            "half_width": hw.loc[[t]],
            "stress": stress_series.loc[[t]],
        }
        w_v10_2, diag = ca_gcp_risk_filter(w_v10, intervals_t, RiskFilterRules())

        r_v10 = float((w_v10 * realized).sum())
        r_v10_2 = float((w_v10_2 * realized).sum())
        prev_v10 *= (1 + r_v10)
        prev_v10_2 *= (1 + r_v10_2)
        nav_v10.loc[t] = prev_v10
        nav_v10_2.loc[t] = prev_v10_2
        diag_rows.append({"date": t, "realized_ret_v10": r_v10, "realized_ret_v10_2": r_v10_2, **diag})

    metrics = pd.DataFrame(
        [
            compute_metrics(nav_v10, "v10 (mock momentum)"),
            compute_metrics(nav_v10_2, "v10.2 (mock momentum + CA-GCP risk)"),
        ]
    )
    print(metrics.round(4).to_string(index=False))
    metrics.to_csv(OUT_DIR / "v10_2_comparison.csv", index=False)
    pd.DataFrame(diag_rows).to_csv(OUT_DIR / "v10_2_diagnostics.csv", index=False)
    nav_v10.to_csv(OUT_DIR / "nav_v10.csv", index=True)
    nav_v10_2.to_csv(OUT_DIR / "nav_v10_2.csv", index=True)


if __name__ == "__main__":
    run()