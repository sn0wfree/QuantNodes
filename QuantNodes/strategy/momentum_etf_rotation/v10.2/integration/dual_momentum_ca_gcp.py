"""Dual Momentum + CA-GCP integration.

月频调仓 + 周频风控 + 其他日 hold:
  - 月末: dual_momentum_signal → target weights
  - 周一: CA-GCP full check (green/yellow/red) → 调整权重
  - 其他日 (周二~周日, 非月末): hold 不动
  - 每次交易扣 10bp 交易成本
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_V102 = Path(__file__).resolve().parent.parent
if str(_V102) not in sys.path:
    sys.path.insert(0, str(_V102))

_V10 = _V102.parent / "v10"
if str(_V10) not in sys.path:
    sys.path.insert(0, str(_V10))

from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import CAGCPConfig, CAGCPipeline  # noqa: E402
from .ca_gcp_risk_filter import RiskFilterRules, ca_gcp_risk_filter  # noqa: E402

LOOKBACK_WEEKS = 52
BOND_CODE = "511260"

REPO = Path(__file__).resolve().parents[5]
PER_ETF_DIR = REPO / "data" / "real" / "per_etf"


def load_bond_etf_returns(
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load 511010/511220 daily returns from per_etf data.

    Returns:
        DataFrame (T, 2) with columns ['511010', '511220'].
    """
    codes = ["511010", "511220"]
    series_list = []
    for code in codes:
        path = PER_ETF_DIR / f"{code}.parquet"
        df = pd.read_parquet(path)
        col = "close" if "close" in df.columns else df.columns[0]
        s = df[col].dropna()
        s.name = code
        series_list.append(s)

    prices = pd.concat(series_list, axis=1).dropna()
    returns = prices.pct_change().dropna()
    if start is not None:
        returns = returns.loc[start:]
    if end is not None:
        returns = returns.loc[:end]
    return returns


def build_sector_pipelines(
    etf_returns: pd.DataFrame,
    sector_map: dict[str, str],
    target_assets: list[str],
    config: "CAGCPConfig | None" = None,
) -> dict[str, CAGCPipeline]:
    """Build per-sector CA-GCP pipelines for target assets.

    Each target asset gets its own pipeline trained on its sector's ETFs.
    k is auto-adjusted for small sectors.

    Args:
        etf_returns: Full ETF returns (T, N) including bond ETFs.
        sector_map: {etf_code: sector_name}.
        target_assets: List of 4 asset codes to build pipelines for.
        config: CAGCPConfig (applied to all sectors, k auto-adjusted).

    Returns:
        {asset_code: fitted CAGCPipeline}
    """
    from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import CAGCPConfig

    config = config or CAGCPConfig()
    target_sectors = {sector_map[a] for a in target_assets if a in sector_map}

    # Group codes by sector
    sector_codes: dict[str, list[str]] = {}
    for code, sector in sector_map.items():
        if sector in target_sectors and code in etf_returns.columns:
            sector_codes.setdefault(sector, []).append(code)

    pipelines: dict[str, CAGCPipeline] = {}
    for asset in target_assets:
        sector = sector_map.get(asset)
        if sector is None or sector not in sector_codes:
            continue
        codes = sector_codes[sector]
        sec_returns = etf_returns[codes].dropna()
        if len(sec_returns) < 100:
            continue

        # Auto-adjust k for small sectors
        sec_cfg = CAGCPConfig(
            k=min(config.k, max(1, len(codes) - 1)),
            sensitivity_eta=config.sensitivity_eta,
            recency_tau=config.recency_tau,
            sharpness_p=config.sharpness_p,
            graph_method=config.graph_method,
            ewma_span=config.ewma_span,
            realized_window=config.realized_window,
        )
        pipe = CAGCPipeline(sec_cfg)
        pipe.fit(sec_returns)
        pipelines[asset] = pipe

    return pipelines


def _days_since_month_end(date: pd.Timestamp, rebal_dates: pd.DatetimeIndex) -> int:
    """Trading days since the most recent month-end rebalance date.

    Used to flag Monday alerts that fire too close to month-end (likely
    to be unwound at the next rebalance, generating churn).
    """
    prior = rebal_dates[rebal_dates <= date]
    if len(prior) == 0:
        return -1
    last_rebal = prior[-1]
    return int((date - last_rebal).days)


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


def _build_sector_intervals_t(
    per_asset_intervals: dict[str, dict],
    weights: pd.Series,
    date: pd.Timestamp,
    history_hw: pd.DataFrame | None,
) -> dict | None:
    """Build merged intervals dict from per-asset sector pipelines.

    Slices each asset's intervals to the given date, then concatenates
    into a single dict suitable for ca_gcp_risk_filter.
    """
    lower_parts = []
    upper_parts = []
    hw_parts = []
    stress_parts = []

    for asset_code in weights.index:
        if asset_code not in per_asset_intervals:
            continue
        ai = per_asset_intervals[asset_code]
        if date not in ai["half_width"].index:
            continue
        idx = ai["half_width"].index.get_loc(date)
        lower_parts.append(ai["lower"].iloc[[idx]])
        upper_parts.append(ai["upper"].iloc[[idx]])
        hw_parts.append(ai["half_width"].iloc[[idx]])
        stress_parts.append(ai["stress"].iloc[idx])

    if not lower_parts:
        return None

    mean_stress = float(np.mean(stress_parts)) if stress_parts else 0.0
    return {
        "lower": pd.concat(lower_parts, axis=1),
        "upper": pd.concat(upper_parts, axis=1),
        "half_width": pd.concat(hw_parts, axis=1),
        "stress": pd.Series([mean_stress], index=[date]),
    }


def dual_momentum_with_ca_gcp(
    daily_prices: pd.DataFrame,
    weekly_prices: pd.DataFrame,
    pipe: CAGCPipeline | dict[str, CAGCPipeline],
    etf_returns: pd.DataFrame,
    rules: RiskFilterRules | None = None,
    cost_bp: int = 10,
    rebal_dates: pd.DatetimeIndex | None = None,
    test_start: pd.Timestamp | None = None,
    test_end: pd.Timestamp | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run dual momentum with CA-GCP weekly risk control.

    Monthly rebal + weekly CA-GCP check + hold on other days:
      - Month-end: dual_momentum_signal → target weights
      - Monday: CA-GCP full check (green/yellow/red) → adjust if needed
      - Other days (Tue-Sun, non-month-end): hold previous weights
      - Every trade incurs cost_bp transaction cost

    Supports two modes:
      - Global: pipe is a single CAGCPipeline (all assets)
      - Sector: pipe is {asset_code: CAGCPipeline} (per-asset sector pipelines)

    Args:
        daily_prices: Daily close prices (T, 4) for the 4 assets.
        weekly_prices: Weekly close prices (T_w, 4) for signal.
        pipe: Fitted CAGCPipeline or dict of per-asset pipelines.
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
    is_sector = isinstance(pipe, dict)

    if test_start is not None:
        daily_prices = daily_prices.loc[test_start:]
    if test_end is not None:
        daily_prices = daily_prices.loc[:test_end]

    # Default: monthly end rebalancing (use actual last trading day per month)
    if rebal_dates is None:
        month_groups = daily_prices.index.to_period("M")
        rebal_dates = pd.DatetimeIndex(
            daily_prices.index.to_series().groupby(month_groups).max().values
        )

    # Pre-compute CA-GCP intervals for entire test period
    test_returns = etf_returns.reindex(daily_prices.index, method="ffill").fillna(0.0)
    calib_returns = etf_returns.loc[
        daily_prices.index[0] - pd.Timedelta(days=300):
        daily_prices.index[0] - pd.Timedelta(days=1),
    ]

    if is_sector:
        # Sector mode: per-asset pipelines (use full sector returns for stress)
        per_asset_intervals: dict[str, dict] = {}
        for asset_code, asset_pipe in pipe.items():
            # Use all sector codes for prediction (needed for stress computation)
            sec_codes = asset_pipe.codes
            sec_test = test_returns[sec_codes].dropna()
            sec_calib = calib_returns[sec_codes].dropna()
            if len(sec_test) > 0 and len(sec_calib) > 0:
                full_out = asset_pipe.predict_fast(sec_calib, sec_test)
                # Extract only the target asset's columns
                per_asset_intervals[asset_code] = {
                    "lower": full_out["lower"][[asset_code]],
                    "upper": full_out["upper"][[asset_code]],
                    "half_width": full_out["half_width"][[asset_code]],
                    "stress": full_out["stress"],
                }
        intervals = None
    else:
        # Global mode: single pipeline
        intervals = pipe.predict_fast(calib_returns, test_returns)
        per_asset_intervals = None

    nav = pd.Series(1.0, index=daily_prices.index, dtype=float)
    prev_weights = pd.Series(0.0, index=daily_prices.columns, dtype=float)
    diag_rows = []
    history_hw = None

    for i in range(1, len(daily_prices)):
        date = daily_prices.index[i]
        is_month_end = date in rebal_dates
        is_monday = date.dayofweek == 0  # Monday = 0

        if is_month_end:
            # Month-end: compute signal + full CA-GCP check
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

            # Full CA-GCP check on month-end
            if is_sector:
                intervals_t = _build_sector_intervals_t(
                    per_asset_intervals, w_target, date, history_hw,
                )
            else:
                if date in intervals["half_width"].index:
                    idx_t = intervals["half_width"].index.get_loc(date)
                    intervals_t = {
                        "lower": intervals["lower"].iloc[[idx_t]],
                        "upper": intervals["upper"].iloc[[idx_t]],
                        "half_width": intervals["half_width"].iloc[[idx_t]],
                        "stress": intervals["stress"].iloc[[idx_t]],
                    }
                else:
                    intervals_t = None

            if intervals_t is not None:
                w_adj, diag = ca_gcp_risk_filter(
                    w_target, intervals_t, rules,
                    today=date, history=history_hw,
                )
                if history_hw is None:
                    history_hw = intervals_t["half_width"].copy()
                else:
                    history_hw = pd.concat([
                        history_hw, intervals_t["half_width"],
                    ])
            else:
                w_adj = w_target.copy()
                diag = {
                    "alert_level": "green",
                    "applied_scale": 1.0,
                    "width_z_today": 0.0,
                    "stress_today": 0.0,
                }

        elif is_monday:
            # Monday (non-month-end): CA-GCP check on current weights
            w_target = prev_weights.copy()
            w_adj = prev_weights.copy()
            diag = {
                "alert_level": "green",
                "applied_scale": 1.0,
                "width_z_today": 0.0,
                "stress_today": 0.0,
            }

            if is_sector:
                intervals_t = _build_sector_intervals_t(
                    per_asset_intervals, w_target, date, history_hw,
                )
            else:
                if date in intervals["half_width"].index:
                    idx_t = intervals["half_width"].index.get_loc(date)
                    intervals_t = {
                        "lower": intervals["lower"].iloc[[idx_t]],
                        "upper": intervals["upper"].iloc[[idx_t]],
                        "half_width": intervals["half_width"].iloc[[idx_t]],
                        "stress": intervals["stress"].iloc[[idx_t]],
                    }
                else:
                    intervals_t = None

            if intervals_t is not None:
                w_adj, diag = ca_gcp_risk_filter(
                    w_target, intervals_t, rules,
                    today=date, history=history_hw,
                )
                if history_hw is None:
                    history_hw = intervals_t["half_width"].copy()
                else:
                    history_hw = pd.concat([
                        history_hw, intervals_t["half_width"],
                    ])

        else:
            # Other days (Tue-Sun, non-month-end): hold
            w_target = prev_weights.copy()
            w_adj = prev_weights.copy()
            diag = {
                "alert_level": "green",
                "applied_scale": 1.0,
                "width_z_today": 0.0,
                "stress_today": 0.0,
            }

        # Compute portfolio return
        day_ret = daily_prices.iloc[i] / daily_prices.iloc[i - 1] - 1
        port_ret = (w_adj * day_ret).sum()

        # Transaction cost (every trade)
        turnover = (w_adj - prev_weights).abs().sum()
        cost = turnover * cost_bp / 10000

        # Update NAV
        nav.iloc[i] = nav.iloc[i - 1] * (1 + port_ret - cost)

        diag_rows.append({
            "date": date,
            "w_target": w_target.to_dict(),
            "w_adj": w_adj.to_dict(),
            "w_prev": prev_weights.to_dict(),
            "turnover": turnover,
            "cost": cost,
            "port_ret": port_ret,
            "alert_level": diag["alert_level"],
            "applied_scale": diag.get("applied_scale", 1.0),
            "width_z": diag.get("width_z_today", 0.0),
            "stress": diag.get("stress_today", 0.0),
            "is_month_end": is_month_end,
            "is_monday": is_monday,
            "days_since_month_end": _days_since_month_end(date, rebal_dates),
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
) -> tuple[pd.Series, pd.DataFrame]:
    """Pure dual momentum without CA-GCP (baseline).

    Monthly rebalancing (month-end). No CA-GCP intervention.

    Returns:
        (nav, diagnostics_df): NAV series and daily diagnostics with
        columns: date, turnover, cost, port_ret, alert_level.
    """
    if test_start is not None:
        daily_prices = daily_prices.loc[test_start:]
    if test_end is not None:
        daily_prices = daily_prices.loc[:test_end]

    # Default: monthly end rebalancing (use actual last trading day per month)
    if rebal_dates is None:
        month_groups = daily_prices.index.to_period("M")
        rebal_dates = pd.DatetimeIndex(
            daily_prices.index.to_series().groupby(month_groups).max().values
        )

    nav = pd.Series(1.0, index=daily_prices.index, dtype=float)
    prev_weights = pd.Series(0.0, index=daily_prices.columns, dtype=float)
    diag_rows = []

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

        diag_rows.append({
            "date": date,
            "turnover": turnover,
            "cost": cost,
            "port_ret": port_ret,
            "alert_level": "green",
            "applied_scale": 1.0,
            "is_month_end": date in rebal_dates,
            "is_monday": date.dayofweek == 0,
            "days_since_month_end": _days_since_month_end(date, rebal_dates),
        })

        prev_weights = curr_weights

    diag_df = pd.DataFrame(diag_rows)
    return nav, diag_df
