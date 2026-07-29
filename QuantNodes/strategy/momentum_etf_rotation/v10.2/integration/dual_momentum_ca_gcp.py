"""Dual Momentum + CA-GCP integration.

月频调仓 + 日频保护:
  - 月度: dual_momentum_signal → target weights
  - 日频: CA-GCP monitor → panic scale-down if stress/yellow
  - 每次交易扣 10bp 交易成本
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_V102 = Path(__file__).resolve().parent.parent
if str(_V102) not in sys.path:
    sys.path.insert(0, str(_V102))

_V10 = _V102.parent / "v10"
if str(_V10) not in sys.path:
    sys.path.insert(0, str(_V10))

from ca_gcp.core import CAGCPipeline  # noqa: E402
from .ca_gcp_risk_filter import RiskFilterRules, ca_gcp_risk_filter  # noqa: E402

LOOKBACK_WEEKS = 52
BOND_CODE = "511260"


def dual_momentum_signal(
    prices: pd.DataFrame,
    lookback_weeks: int = LOOKBACK_WEEKS,
) -> pd.Series:
    """Compute dual momentum signal for a single date.

    Args:
        prices: Weekly close prices, shape (N_weeks, 4).
        lookback_weeks: Lookback window for momentum.

    Returns:
        pd.Series: index=asset codes, values=0.0 or 1.0 (exactly one 1.0).
    """
    returns_lookback = prices.pct_change(lookback_weeks).iloc[-1:]
    total_ret = returns_lookback.iloc[0]
    positive_mask = total_ret > 0

    risk_assets = [c for c in prices.columns if c != BOND_CODE]
    weights = pd.Series(0.0, index=prices.columns)

    if positive_mask[risk_assets].any():
        valid_rets = total_ret[risk_assets][positive_mask[risk_assets]]
        best = valid_rets.idxmax()
        weights[best] = 1.0
    else:
        weights[BOND_CODE] = 1.0

    return weights


def dual_momentum_with_ca_gcp(
    daily_prices: pd.DataFrame,
    weekly_prices: pd.DataFrame,
    pipe: CAGCPipeline,
    etf_returns: pd.DataFrame,
    rules: RiskFilterRules | None = None,
    cost_bp: int = 10,
    rebal_dates: pd.DatetimeIndex | None = None,
    test_start: pd.Timestamp | None = None,
    test_end: pd.Timestamp | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run dual momentum with CA-GCP daily protection.

    Monthly rebal + daily CA-GCP override:
      - Month-end: dual_momentum_signal → target weights
      - Daily: if CA-GCP stress > threshold →临时降仓
      - Every trade incurs cost_bp transaction cost

    Args:
        daily_prices: Daily close prices (T, 4) for the 4 assets.
        weekly_prices: Weekly close prices (T_w, 4) for signal.
        pipe: Fitted CAGCPipeline.
        etf_returns: ETF daily returns for risk filter.
        rules: Risk filter rules.
        cost_bp: Transaction cost in basis points.
        rebal_dates: Rebalance dates (month-end). Auto if None.
        test_start: Start of test period.
        test_end: End of test period.

    Returns:
        (nav, diagnostics_df): NAV series and daily diagnostics.
    """
    rules = rules or RiskFilterRules()

    if test_start is not None:
        daily_prices = daily_prices.loc[test_start:]
    if test_end is not None:
        daily_prices = daily_prices.loc[:test_end]

    if rebal_dates is None:
        rebal_dates = daily_prices.resample("M").last().index

    # Pre-compute CA-GCP intervals for entire test period
    test_returns = etf_returns.reindex(daily_prices.index, method="ffill").fillna(0.0)
    calib_returns = etf_returns.loc[
        daily_prices.index[0] - pd.Timedelta(days=300):
        daily_prices.index[0] - pd.Timedelta(days=1),
    ]
    intervals = pipe.predict_fast(calib_returns, test_returns)

    nav = pd.Series(1.0, index=daily_prices.index, dtype=float)
    prev_weights = pd.Series(0.0, index=daily_prices.columns, dtype=float)
    diag_rows = []
    history_hw = None
    prev_alert_level = "green"  # hysteresis state

    for i in range(1, len(daily_prices)):
        date = daily_prices.index[i]

        # Step 1: Determine target weights
        if date in rebal_dates:
            wk = weekly_prices.loc[:date]
            if len(wk) >= LOOKBACK_WEEKS:
                w_target = dual_momentum_signal(wk, LOOKBACK_WEEKS)
            else:
                n_assets = len(daily_prices.columns)
                w_target = pd.Series(0.0, index=daily_prices.columns)
                w_target[BOND_CODE] = 0.75
                for col in daily_prices.columns:
                    if col != BOND_CODE:
                        w_target[col] = 0.25 / (n_assets - 1)
        else:
            w_target = prev_weights.copy()

        # Step 2: Daily CA-GCP check (override if panic)
        if date in intervals["half_width"].index:
            idx_t = intervals["half_width"].index.get_loc(date)
            intervals_t = {
                "lower": intervals["lower"].iloc[[idx_t]],
                "upper": intervals["upper"].iloc[[idx_t]],
                "half_width": intervals["half_width"].iloc[[idx_t]],
                "stress": intervals["stress"].iloc[[idx_t]],
            }
            w_adj, diag = ca_gcp_risk_filter(
                w_target, intervals_t, rules, today=date, history=history_hw,
            )
            # Hysteresis: if prev was yellow/red, require lower stress to recover
            if prev_alert_level in ("yellow", "red") and diag["alert_level"] == "green":
                stress_now = diag.get("stress_today", 0.0)
                width_now = diag.get("width_z_today", 0.0)
                if (stress_now > rules.stress_yellow_recovery or
                        width_now > rules.width_z_yellow_recovery):
                    diag["alert_level"] = "yellow"
                    diag["applied_scale"] = rules.yellow_scale
                    w_adj = w_target * rules.yellow_scale
                    residual = (1.0 - w_adj.sum()) if w_adj.sum() < 1.0 else 0.0
                    if residual > 0:
                        w_min = w_target.abs().idxmin()
                        w_adj[w_min] += residual
            prev_alert_level = diag["alert_level"]
            if history_hw is None:
                history_hw = intervals["half_width"].iloc[: idx_t + 1].copy()
            else:
                history_hw = pd.concat([history_hw, intervals["half_width"].iloc[[idx_t]]])
        else:
            w_adj = w_target.copy()
            diag = {
                "alert_level": "green",
                "applied_scale": 1.0,
                "width_z_today": 0.0,
                "stress_today": 0.0,
            }

        # Step 3: Compute portfolio return
        day_ret = daily_prices.iloc[i] / daily_prices.iloc[i - 1] - 1
        port_ret = (w_adj * day_ret).sum()

        # Step 4: Transaction cost (every trade)
        turnover = (w_adj - prev_weights).abs().sum()
        cost = turnover * cost_bp / 10000

        # Step 5: Update NAV
        nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret - cost)

        diag_rows.append({
            "date": date,
            "w_target": w_target.to_dict(),
            "w_adj": w_adj.to_dict(),
            "turnover": turnover,
            "cost": cost,
            "port_ret": port_ret,
            "alert_level": diag["alert_level"],
            "width_z": diag.get("width_z_today", 0.0),
            "stress": diag.get("stress_today", 0.0),
        })

        prev_weights = w_adj.copy()

    diag_df = pd.DataFrame(diag_rows)
    return nav, diag_df


def dual_momentum_bare(
    daily_prices: pd.DataFrame,
    weekly_prices: pd.DataFrame,
    cost_bp: int = 10,
    rebal_dates: pd.DatetimeIndex | None = None,
    test_start: pd.Timestamp | None = None,
    test_end: pd.Timestamp | None = None,
) -> pd.Series:
    """Pure dual momentum without CA-GCP (baseline).

    Same as dual_momentum_with_ca_gcp but no CA-GCP intervention.
    """
    if test_start is not None:
        daily_prices = daily_prices.loc[test_start:]
    if test_end is not None:
        daily_prices = daily_prices.loc[:test_end]

    if rebal_dates is None:
        rebal_dates = daily_prices.resample("M").last().index

    nav = pd.Series(1.0, index=daily_prices.index, dtype=float)
    prev_weights = pd.Series(0.0, index=daily_prices.columns, dtype=float)

    for i in range(1, len(daily_prices)):
        date = daily_prices.index[i]

        if date in rebal_dates:
            wk = weekly_prices.loc[:date]
            if len(wk) >= LOOKBACK_WEEKS:
                curr_weights = dual_momentum_signal(wk, LOOKBACK_WEEKS)
            else:
                n_assets = len(daily_prices.columns)
                curr_weights = pd.Series(0.0, index=daily_prices.columns)
                curr_weights[BOND_CODE] = 0.75
                for col in daily_prices.columns:
                    if col != BOND_CODE:
                        curr_weights[col] = 0.25 / (n_assets - 1)
        else:
            curr_weights = prev_weights.copy()

        day_ret = daily_prices.iloc[i] / daily_prices.iloc[i - 1] - 1
        port_ret = (curr_weights * day_ret).sum()

        turnover = (curr_weights - prev_weights).abs().sum()
        cost = turnover * cost_bp / 10000

        nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret - cost)
        prev_weights = curr_weights

    return nav
