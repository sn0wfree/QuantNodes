"""Dual Momentum + CA-GCP Walk-Forward backtest.

Walk-Forward 5 fold校准 + 3 candidates:
  Candidate 1: dual_mom pure vs dual_mom + CA-GCP
  Candidate 2: 4 strategies comparison table
  Candidate 3: Sector CA-GCP on dual_mom

Monthly rebal (month-end) + weekly CA-GCP risk control (Monday)
Other days: hold. Every trade incurs 10bp transaction cost.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/ll/Public/QuantNodes")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation"))
sys.path.insert(0, str(ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2"))

from _path import *  # noqa: F401,F403,E402
from ca_gcp import CAGCPConfig, CAGCPipeline  # noqa: E402
from ca_gcp.validators import compute_coverage_metrics, width_bps  # noqa: E402
from integration.ca_gcp_risk_filter import RiskFilterRules  # noqa: E402
from integration.dual_momentum_ca_gcp import (  # noqa: E402
    dual_momentum_bare,
    dual_momentum_with_ca_gcp,
)

V10 = ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10"
sys.path.insert(0, str(V10))
from dual_momentum import load_all_assets_daily  # noqa: E402

COMMON = ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "common"
sys.path.insert(0, str(COMMON))
from metrics import compute_metrics  # noqa: E402

OUT_DIR = ROOT / "QuantNodes" / "strategy" / "momentum_etf_rotation" / "v10.2" / "data" / "results"

K_GRID = [2, 4, 6, 8, 12, 16, 24]
ETA_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
TAU_GRID = [20, 30, 45, 60, 90, 120, 180]
EARLY_STOP_THRESHOLD = 50

WF_FOLDS = [
    {
        "train_end": pd.Timestamp("2022-01-01"),
        "calib_end": pd.Timestamp("2022-10-01"),
        "test_end": pd.Timestamp("2023-09-01"),
    },
    {
        "train_end": pd.Timestamp("2022-08-01"),
        "calib_end": pd.Timestamp("2023-05-01"),
        "test_end": pd.Timestamp("2024-04-01"),
    },
    {
        "train_end": pd.Timestamp("2023-03-01"),
        "calib_end": pd.Timestamp("2024-01-01"),
        "test_end": pd.Timestamp("2024-12-01"),
    },
    {
        "train_end": pd.Timestamp("2023-10-01"),
        "calib_end": pd.Timestamp("2024-08-01"),
        "test_end": pd.Timestamp("2025-07-01"),
    },
    {
        "train_end": pd.Timestamp("2024-05-01"),
        "calib_end": pd.Timestamp("2025-03-01"),
        "test_end": pd.Timestamp("2026-06-30"),
    },
]


def pareto_score(m: dict, w_bps: float) -> float:
    if np.isnan(m.get("extreme", np.nan)):
        return -1e9
    return (
        10.0 * m["extreme"]
        - 5.0 * m["pa_std"]
        - 1.0 * (w_bps / 1000.0)
    )


def calibrate_fold(
    etf_returns: pd.DataFrame,
    train_end: pd.Timestamp,
    calib_end: pd.Timestamp,
) -> CAGCPConfig:
    """Grid search on calib period, fit on train, return best config.

    Uses full calib period (no internal train/calib split) for maximum
    data utilization. Validation is the last 252 calib days.
    """
    train = etf_returns.loc[:train_end].iloc[:-1]
    calib = etf_returns.loc[train_end:calib_end]

    # Use last 252 calib days as validation for grid search
    if len(calib) > 252:
        val = calib.iloc[-252:]
        calib_grid = calib.iloc[:-252]
    else:
        # Calib too short: use 70/30 split
        split = int(len(calib) * 0.7)
        calib_grid = calib.iloc[:split]
        val = calib.iloc[split:]
    actual = val

    cross_dispersion = val.std(axis=1)
    extreme_mask = cross_dispersion > cross_dispersion.quantile(0.9)

    pipe = CAGCPipeline(CAGCPConfig(k=4))
    pipe.fit(train)

    best = None
    best_score = -np.inf
    no_improve = 0
    total = len(K_GRID) * len(ETA_GRID) * len(TAU_GRID)
    t0 = time.time()

    for idx, (k, eta, tau) in enumerate(itertools.product(K_GRID, ETA_GRID, TAU_GRID), 1):
        pipe.config.k = k
        pipe.config.sensitivity_eta = eta
        pipe.config.recency_tau = tau

        out = pipe.predict_fast(calib_grid, val)
        m = compute_coverage_metrics(actual, out["lower"], out["upper"], extreme_mask)
        w_bps = width_bps(out["half_width"])
        s = pareto_score(m, w_bps)

        if s > best_score:
            best_score = s
            best = {"k": k, "eta": eta, "tau": tau, "score": s}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP_THRESHOLD and idx >= 100:
            break

    elapsed = time.time() - t0
    print(f"  Grid search: {idx}/{total} combos in {elapsed:.1f}s, best_score={best_score:.3f}")

    if best is None:
        return CAGCPConfig()

    return CAGCPConfig(
        k=int(best["k"]),
        sensitivity_eta=float(best["eta"]),
        recency_tau=float(best["tau"]),
    )


def run_fold(
    daily_prices: pd.DataFrame,
    weekly_prices: pd.DataFrame,
    etf_returns: pd.DataFrame,
    fold: dict,
    fold_idx: int,
) -> dict:
    """Run one WF fold: calibrate → backtest bare vs CA-GCP."""
    train_end = fold["train_end"]
    calib_end = fold["calib_end"]
    test_end = fold["test_end"]

    print(f"\n--- Fold {fold_idx + 1}: "
          f"Train→{train_end.date()}, Calib→{calib_end.date()}, "
          f"Test→{test_end.date()} ---")

    # 1. Calibrate
    print("  Calibrating...")
    cfg = calibrate_fold(etf_returns, train_end, calib_end)
    print(f"  Best config: k={cfg.k}, eta={cfg.sensitivity_eta}, tau={cfg.recency_tau}")

    # 2. Fit CA-GCP on train window (600 days before calib_end)
    train_slice = etf_returns.loc[:calib_end].iloc[-600:]
    pipe = CAGCPipeline(cfg)
    pipe.fit(train_slice)

    # 3. Run bare dual momentum on test period
    print("  Running bare dual momentum...")
    nav_bare, diag_bare = dual_momentum_bare(
        daily_prices, weekly_prices,
        cost_bp=10,
        test_start=calib_end,
        test_end=test_end,
    )
    m_bare = compute_metrics(nav_bare)

    # 4. Run dual momentum + CA-GCP on test period
    print("  Running dual momentum + CA-GCP...")
    nav_cagcp, diag_cagcp = dual_momentum_with_ca_gcp(
        daily_prices, weekly_prices, pipe, etf_returns,
        rules=RiskFilterRules(),
        cost_bp=10,
        test_start=calib_end,
        test_end=test_end,
    )
    m_cagcp = compute_metrics(nav_cagcp)

    # 5. Turnover & cost diagnostics
    n_years = m_bare.get("Years", 1.0)
    total_turnover_bare = diag_bare["turnover"].sum()
    total_cost_bare = diag_bare["cost"].sum()
    ann_turnover_bare = total_turnover_bare / n_years
    ann_cost_bare = total_cost_bare / n_years

    total_turnover_cagcp = diag_cagcp["turnover"].sum()
    total_cost_cagcp = diag_cagcp["cost"].sum()
    ann_turnover_cagcp = total_turnover_cagcp / n_years
    ann_cost_cagcp = total_cost_cagcp / n_years

    alerts = diag_cagcp["alert_level"].value_counts().to_dict()

    print(f"  Bare: Sharpe={m_bare.get('Sharpe', 0):.3f}, "
          f"MaxDD={m_bare.get('MaxDD', 0):.2%}, "
          f"AnnRet={m_bare.get('AnnRet', 0):.2%}")
    print(f"  CA-GCP: Sharpe={m_cagcp.get('Sharpe', 0):.3f}, "
          f"MaxDD={m_cagcp.get('MaxDD', 0):.2%}, "
          f"AnnRet={m_cagcp.get('AnnRet', 0):.2%}")
    print(f"  Bare turnover: {ann_turnover_bare:.1f}x/yr, cost: {ann_cost_bare:.2%}/yr")
    print(f"  CA-GCP turnover: {ann_turnover_cagcp:.1f}x/yr, cost: {ann_cost_cagcp:.2%}/yr")

    return {
        "fold": fold_idx + 1,
        "test_period": f"{calib_end.date()} ~ {test_end.date()}",
        "config": f"k={cfg.k},eta={cfg.sensitivity_eta},tau={cfg.recency_tau}",
        "bare": m_bare,
        "cagcp": m_cagcp,
        "alerts": alerts,
        "ann_turnover_bare": ann_turnover_bare,
        "ann_cost_bare": ann_cost_bare,
        "ann_turnover_cagcp": ann_turnover_cagcp,
        "ann_cost_cagcp": ann_cost_cagcp,
        "nav_bare": nav_bare,
        "nav_cagcp": nav_cagcp,
        "diag_bare": diag_bare,
        "diag_cagcp": diag_cagcp,
    }


def main() -> None:
    print("=" * 60)
    print("Dual Momentum + CA-GCP Walk-Forward Backtest")
    print("Monthly rebal + weekly CA-GCP risk control")
    print("=" * 60)

    # Load data
    print("\n[1] Loading data...")
    daily_prices = load_all_assets_daily()
    weekly_prices = daily_prices.resample("W-SUN").last().dropna()
    etf_returns = daily_prices.pct_change().fillna(0.0)
    print(f"  Daily: {daily_prices.shape}, "
          f"{daily_prices.index[0].date()} ~ {daily_prices.index[-1].date()}")

    # Run 4-fold WF
    print("\n[2] Walk-Forward 5 fold...")
    all_results = []
    for i, fold in enumerate(WF_FOLDS):
        result = run_fold(daily_prices, weekly_prices, etf_returns, fold, i)
        all_results.append(result)

    # Summary table — full metrics
    print("\n" + "=" * 80)
    print("SUMMARY: Dual Momentum Bare vs + CA-GCP (Monthly Rebal + Weekly Risk)")
    print("=" * 80)

    rows = []
    for r in all_results:
        b = r["bare"]
        c = r["cagcp"]
        rows.append({
            "Fold": r["fold"],
            "Period": r["test_period"],
            "Bare_AnnRet": b.get("AnnRet", 0),
            "Bare_Vol": b.get("Vol", 0),
            "Bare_MaxDD": b.get("MaxDD", 0),
            "Bare_Sharpe": b.get("Sharpe", 0),
            "Bare_Sortino": b.get("Sortino", 0),
            "Bare_Calmar": b.get("Calmar", 0),
            "Bare_WinRate": b.get("WinRate", 0),
            "Bare_AnnTurnover": r["ann_turnover_bare"],
            "Bare_AnnCost": r["ann_cost_bare"],
            "CAGCP_AnnRet": c.get("AnnRet", 0),
            "CAGCP_Vol": c.get("Vol", 0),
            "CAGCP_MaxDD": c.get("MaxDD", 0),
            "CAGCP_Sharpe": c.get("Sharpe", 0),
            "CAGCP_Sortino": c.get("Sortino", 0),
            "CAGCP_Calmar": c.get("Calmar", 0),
            "CAGCP_WinRate": c.get("WinRate", 0),
            "CAGCP_AnnTurnover": r["ann_turnover_cagcp"],
            "CAGCP_AnnCost": r["ann_cost_cagcp"],
        })

    summary = pd.DataFrame(rows)

    # Print Bare metrics
    print("\n--- Bare Dual Momentum ---")
    bare_cols = ["Fold", "Period", "Bare_AnnRet", "Bare_Vol",
                 "Bare_MaxDD", "Bare_Sharpe", "Bare_Sortino",
                 "Bare_Calmar", "Bare_WinRate",
                 "Bare_AnnTurnover", "Bare_AnnCost"]
    print(summary[bare_cols].to_string(index=False, float_format="%.4f"))

    # Print CA-GCP metrics
    print("\n--- Dual Momentum + CA-GCP ---")
    cagcp_cols = ["Fold", "Period", "CAGCP_AnnRet", "CAGCP_Vol",
                  "CAGCP_MaxDD", "CAGCP_Sharpe", "CAGCP_Sortino",
                  "CAGCP_Calmar", "CAGCP_WinRate",
                  "CAGCP_AnnTurnover", "CAGCP_AnnCost"]
    print(summary[cagcp_cols].to_string(index=False, float_format="%.4f"))

    # Print deltas
    print("\n--- Delta (CA-GCP - Bare) ---")
    delta_df = pd.DataFrame({
        "Fold": summary["Fold"],
        "Period": summary["Period"],
        "AnnRet_Δ": summary["CAGCP_AnnRet"] - summary["Bare_AnnRet"],
        "Vol_Δ": summary["CAGCP_Vol"] - summary["Bare_Vol"],
        "MaxDD_Δ": summary["CAGCP_MaxDD"] - summary["Bare_MaxDD"],
        "Sharpe_Δ": summary["CAGCP_Sharpe"] - summary["Bare_Sharpe"],
        "Sortino_Δ": summary["CAGCP_Sortino"] - summary["Bare_Sortino"],
        "Calmar_Δ": summary["CAGCP_Calmar"] - summary["Bare_Calmar"],
        "WinRate_Δ": summary["CAGCP_WinRate"] - summary["Bare_WinRate"],
        "AnnTurnover_Δ": summary["CAGCP_AnnTurnover"] - summary["Bare_AnnTurnover"],
        "AnnCost_Δ": summary["CAGCP_AnnCost"] - summary["Bare_AnnCost"],
    })
    print(delta_df.to_string(index=False, float_format="%+.4f"))

    # Aggregate averages
    print("\n--- Aggregate ---")
    metrics_list = ["AnnRet", "Vol", "MaxDD", "Sharpe", "Sortino", "Calmar",
                    "WinRate", "AnnTurnover", "AnnCost"]
    agg = pd.DataFrame({
        "Metric": metrics_list,
        "Bare_Avg": [
            summary["Bare_AnnRet"].mean(),
            summary["Bare_Vol"].mean(),
            summary["Bare_MaxDD"].mean(),
            summary["Bare_Sharpe"].mean(),
            summary["Bare_Sortino"].mean(),
            summary["Bare_Calmar"].mean(),
            summary["Bare_WinRate"].mean(),
            summary["Bare_AnnTurnover"].mean(),
            summary["Bare_AnnCost"].mean(),
        ],
        "CAGCP_Avg": [
            summary["CAGCP_AnnRet"].mean(),
            summary["CAGCP_Vol"].mean(),
            summary["CAGCP_MaxDD"].mean(),
            summary["CAGCP_Sharpe"].mean(),
            summary["CAGCP_Sortino"].mean(),
            summary["CAGCP_Calmar"].mean(),
            summary["CAGCP_WinRate"].mean(),
            summary["CAGCP_AnnTurnover"].mean(),
            summary["CAGCP_AnnCost"].mean(),
        ],
    })
    agg["Delta"] = agg["CAGCP_Avg"] - agg["Bare_Avg"]
    print(agg.to_string(index=False, float_format="%.4f"))
    print(f"\nCA-GCP helps (Sharpe): "
          f"{'YES' if agg.loc[agg['Metric'] == 'Sharpe', 'Delta'].iloc[0] > 0 else 'NO'}")

    # Save
    summary.to_csv(OUT_DIR / "dual_momentum_wf_summary.csv", index=False)
    for r in all_results:
        r["nav_bare"].to_frame("nav_bare").to_csv(
            OUT_DIR / f"dual_mom_bare_fold{r['fold']}.csv", index=True,
        )
        r["nav_cagcp"].to_frame("nav_cagcp").to_csv(
            OUT_DIR / f"dual_mom_cagcp_fold{r['fold']}.csv", index=True,
        )
        r["diag_bare"].to_csv(
            OUT_DIR / f"dual_mom_diag_bare_fold{r['fold']}.csv", index=False,
        )
        r["diag_cagcp"].to_csv(
            OUT_DIR / f"dual_mom_diag_cagcp_fold{r['fold']}.csv", index=False,
        )

    with (OUT_DIR / "dual_momentum_wf_config.json").open("w") as f:
        config_rows = [
            {"fold": r["fold"], "config": r["config"], "alerts": r["alerts"]}
            for r in all_results
        ]
        json.dump(config_rows, f, indent=2)

    print(f"\nResults saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
