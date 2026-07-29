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
from .ca_gcp_risk_filter import RiskFilterRules, ca_gcp_risk_filter, compute_adx_close  # noqa: E402

LOOKBACK_WEEKS = 52
BOND_CODE = "511260"

REPO = Path(__file__).resolve().parents[5]
PER_ETF_DIR = REPO / "data" / "real" / "per_etf"

# Hysteresis constants
ALERT_RANK = {"green": 0, "yellow": 1, "red": 2}

# Adaptive Gate defaults
GATE_VOL_WINDOW = 60
GATE_VOL_PCT = 0.40
GATE_CORR_WINDOW = 60
GATE_CORR_DROP = 0.30
GATE_MIN_HOLD = 10
GATE_ATR_WINDOW = 14


def compute_atr_proxy(returns: pd.DataFrame, window: int = 14) -> pd.Series:
    """Simplified ATR proxy: mean absolute daily return (rolling).

    Since we only have close prices (no OHLC), we use the absolute daily
    return as a proxy for True Range. This is equivalent to the
    mean-absolute-return, which captures volatility regime changes.

    Args:
        returns: Daily percentage returns DataFrame (T, N).
        window: Rolling window for smoothing (default 14 days).

    Returns:
        Series of mean absolute return per day (averaged across assets).
    """
    return returns.abs().rolling(window).mean().mean(axis=1)


def _compute_gate_signal(
    etf_returns: pd.DataFrame,
    vol_window: int = GATE_VOL_WINDOW,
    vol_pct: float = GATE_VOL_PCT,
    corr_window: int = GATE_CORR_WINDOW,
    corr_drop: float = GATE_CORR_DROP,
    atr_window: int = GATE_ATR_WINDOW,
    gate_mode: str = "or",
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute adaptive gate signal.

    Args:
        gate_mode: "or" (vol OR corr) or "and" (vol AND corr).

    Returns:
        (gate: bool Series, vol_signal: bool Series, corr_signal: bool Series)
    """
    # Condition A: ATR-based volatility filter
    atr = compute_atr_proxy(etf_returns, window=atr_window)
    vol_pctile = atr.expanding(vol_window).rank(pct=True)
    vol_signal = vol_pctile > vol_pct

    # Condition B: Correlation breakdown detection
    # Compute rolling average pairwise correlation
    n = etf_returns.shape[1]
    rolling_corr = etf_returns.rolling(20).corr()
    # Extract average correlation per day (excluding self-correlation)
    avg_corr_list = []
    for t in etf_returns.index:
        if t in rolling_corr.index.get_level_values(0):
            corr_t = rolling_corr.loc[t]
            if isinstance(corr_t, pd.DataFrame) and corr_t.shape == (n, n):
                # Mask diagonal and compute mean of upper triangle
                mask = np.triu(np.ones((n, n), dtype=bool), k=1)
                avg_corr_list.append(float(corr_t.values[mask].mean()))
            else:
                avg_corr_list.append(np.nan)
        else:
            avg_corr_list.append(np.nan)
    avg_corr = pd.Series(avg_corr_list, index=etf_returns.index)

    # Detect breakdown: drop from ROLLING max (not expanding/all-time)
    corr_high = avg_corr.rolling(corr_window).max()
    corr_drop_pct = (corr_high - avg_corr) / corr_high.replace(0, np.nan)
    corr_signal = corr_drop_pct > corr_drop

    # Combine based on mode
    if gate_mode == "and":
        gate = vol_signal & corr_signal
    else:
        gate = vol_signal | corr_signal

    return gate, vol_signal, corr_signal


def _apply_hysteresis(
    raw_alert: str,
    raw_scale: float,
    prev_alert: str,
    consecutive_raw_above: int,
    last_upgrade_idx: int,
    current_idx: int,
    rules: RiskFilterRules,
    w_adj: pd.Series,
    w_target: pd.Series,
) -> tuple[str, pd.Series, int]:
    """Apply hysteresis state machine to raw alert level.

    Rules:
    1. Upgrade requires `confirm_threshold` consecutive raw alerts at same or higher level.
    2. Downgrade requires minimum hold period.
    3. Minimum hold period after upgrade.

    Returns:
        (filtered_alert, adjusted_weights, new_consecutive_raw_above)
    """
    raw_rank = ALERT_RANK[raw_alert]
    prev_rank = ALERT_RANK[prev_alert]

    # Track consecutive raw alerts above prev level
    if raw_rank >= prev_rank:
        new_consecutive = consecutive_raw_above + 1
    else:
        new_consecutive = 0

    # Upgrade: raw alert is higher than previous
    if raw_rank > prev_rank:
        if new_consecutive < rules.confirm_threshold:
            # Not enough confirmations yet, stay at prev level
            return prev_alert, w_adj, new_consecutive

    # Downgrade: raw alert is lower than previous
    if raw_rank < prev_rank:
        # Check minimum hold period
        if current_idx - last_upgrade_idx < rules.min_hold_days:
            return prev_alert, w_adj, new_consecutive

    return raw_alert, w_adj, new_consecutive


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
    use_gate: bool = False,
    gate_mode: str = "or",
    gate_vol_window: int = GATE_VOL_WINDOW,
    gate_vol_pct: float = GATE_VOL_PCT,
    gate_corr_window: int = GATE_CORR_WINDOW,
    gate_corr_drop: float = GATE_CORR_DROP,
    gate_min_hold: int = GATE_MIN_HOLD,
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

    Adaptive Gate (optional):
      - use_gate=True enables market-state adaptive filtering
      - gate_mode="or": CA-GCP active if vol OR corr breakdown
      - gate_mode="and": CA-GCP active if vol AND corr breakdown
      - Gate resets at each month-end rebalance

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
        use_gate: Enable adaptive gate filtering.
        gate_mode: "or" or "and" for combining vol and corr signals.
        gate_vol_window: Rolling window for volatility percentile.
        gate_vol_pct: Volatility percentile threshold (top X%).
        gate_corr_window: Rolling window for correlation breakdown.
        gate_corr_drop: Relative correlation drop threshold.
        gate_min_hold: Minimum days to hold gate state after switching.

    Returns:
        (nav, diagnostics_df): NAV series and daily diagnostics.
    """
    rules = rules or RiskFilterRules()
    is_sector = isinstance(pipe, dict)

    # Pre-compute gate signal if enabled
    gate_signal = None
    if use_gate:
        gate_signal, _, _ = _compute_gate_signal(
            etf_returns, gate_vol_window, gate_vol_pct,
            gate_corr_window, gate_corr_drop, gate_mode=gate_mode,
        )

    # Pre-compute trend signal for trend filter
    trend_signal = None
    if rules.trend_filter:
        target_returns = etf_returns[daily_prices.columns].mean(axis=1)
        rolling_ret = target_returns.rolling(rules.trend_window).sum()
        trend_signal = rolling_ret < rules.trend_threshold

    # Pre-compute ADX for trend-strength filter
    adx_series = None
    if rules.adx_filter:
        portfolio_close = daily_prices.mean(axis=1)
        adx_series = compute_adx_close(portfolio_close, period=rules.adx_period)

    # Pre-compute MA trend signal (for fusion)
    ma_trend_series = None
    if rules.fusion_enabled:
        portfolio_close = daily_prices.mean(axis=1)
        ma20 = portfolio_close.rolling(20).mean()
        ma60 = portfolio_close.rolling(60).mean()
        ma_trend_series = pd.Series(
            np.where(ma20 > ma60, 1.0, -1.0),
            index=portfolio_close.index,
        )

    # Pre-compute volatility percentile (for fusion)
    vol_pct_series = None
    if rules.fusion_enabled:
        port_returns = daily_prices.mean(axis=1).pct_change().fillna(0.0)
        vol_20 = port_returns.rolling(20).std() * np.sqrt(252)
        vol_pct_series = vol_20.expanding(60).rank(pct=True)

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

    # Hysteresis state
    prev_alert = "green"
    consecutive_alert = 0
    last_upgrade_idx = -100

    # Gate state
    gate_active = False
    gate_last_switch_idx = -100

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
                if date in intervals["half_width"].index and len(intervals["stress"]) > 0:
                    idx_t = intervals["half_width"].index.get_loc(date)
                    intervals_t = {
                        "lower": intervals["lower"].iloc[[idx_t]],
                        "upper": intervals["upper"].iloc[[idx_t]],
                        "half_width": intervals["half_width"].iloc[[idx_t]],
                        "stress": intervals["stress"].iloc[[idx_t]],
                    }
                else:
                    intervals_t = None

            # Adaptive Gate: check if CA-GCP should be active
            gate_is_active = True  # Default: active
            if use_gate and gate_signal is not None:
                # Get gate signal for today
                if date in gate_signal.index:
                    gate_is_active = bool(gate_signal.loc[date])
                else:
                    gate_is_active = False

                # Enforce minimum holding period
                if gate_is_active != gate_active:
                    if i - gate_last_switch_idx < gate_min_hold:
                        gate_is_active = gate_active  # Hold current state
                    else:
                        gate_last_switch_idx = i
                        gate_active = gate_is_active

            if intervals_t is not None and gate_is_active:
                adx_val = float(adx_series.loc[date]) if (adx_series is not None and date in adx_series.index and pd.notna(adx_series.loc[date])) else None
                ma_sign = float(ma_trend_series.loc[date]) if (ma_trend_series is not None and date in ma_trend_series.index and pd.notna(ma_trend_series.loc[date])) else None
                vol_p = float(vol_pct_series.loc[date]) if (vol_pct_series is not None and date in vol_pct_series.index and pd.notna(vol_pct_series.loc[date])) else None
                w_adj, diag = ca_gcp_risk_filter(
                    w_target, intervals_t, rules,
                    today=date, history=history_hw,
                    residual_asset=BOND_CODE,
                    trend_signal=trend_signal,
                    adx_value=adx_val,
                    ma_trend_sign=ma_sign,
                    vol_pct=vol_p,
                )
                if history_hw is None:
                    history_hw = intervals_t["half_width"].copy()
                else:
                    history_hw = pd.concat([
                        history_hw, intervals_t["half_width"],
                    ])
                # Hysteresis: apply state machine to alert level
                raw_alert = diag["alert_level"]
                filtered_alert, w_adj, consecutive_alert = _apply_hysteresis(
                    raw_alert, diag["applied_scale"], prev_alert,
                    consecutive_alert, last_upgrade_idx, i, rules, w_adj, w_target,
                )
                diag["alert_level"] = filtered_alert
                if filtered_alert != raw_alert:
                    diag["applied_scale"] = 1.0 if filtered_alert == "green" else diag["applied_scale"]
                # Update hysteresis state
                if ALERT_RANK[filtered_alert] > ALERT_RANK[prev_alert]:
                    if consecutive_alert >= rules.confirm_threshold:
                        last_upgrade_idx = i
                prev_alert = filtered_alert
            else:
                # Gate OFF or intervals not available: use bare momentum
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
                if date in intervals["half_width"].index and len(intervals["stress"]) > 0:
                    idx_t = intervals["half_width"].index.get_loc(date)
                    intervals_t = {
                        "lower": intervals["lower"].iloc[[idx_t]],
                        "upper": intervals["upper"].iloc[[idx_t]],
                        "half_width": intervals["half_width"].iloc[[idx_t]],
                        "stress": intervals["stress"].iloc[[idx_t]],
                    }
                else:
                    intervals_t = None

            # Adaptive Gate: check if CA-GCP should be active (Monday)
            gate_is_active = True
            if use_gate and gate_signal is not None:
                if date in gate_signal.index:
                    gate_is_active = bool(gate_signal.loc[date])
                else:
                    gate_is_active = False
                # Enforce minimum holding period
                if gate_is_active != gate_active:
                    if i - gate_last_switch_idx < gate_min_hold:
                        gate_is_active = gate_active
                    else:
                        gate_last_switch_idx = i
                        gate_active = gate_is_active

            if intervals_t is not None and gate_is_active:
                adx_val = float(adx_series.loc[date]) if (adx_series is not None and date in adx_series.index and pd.notna(adx_series.loc[date])) else None
                ma_sign = float(ma_trend_series.loc[date]) if (ma_trend_series is not None and date in ma_trend_series.index and pd.notna(ma_trend_series.loc[date])) else None
                vol_p = float(vol_pct_series.loc[date]) if (vol_pct_series is not None and date in vol_pct_series.index and pd.notna(vol_pct_series.loc[date])) else None
                w_adj, diag = ca_gcp_risk_filter(
                    w_target, intervals_t, rules,
                    today=date, history=history_hw,
                    residual_asset=BOND_CODE,
                    trend_signal=trend_signal,
                    adx_value=adx_val,
                    ma_trend_sign=ma_sign,
                    vol_pct=vol_p,
                )
                if history_hw is None:
                    history_hw = intervals_t["half_width"].copy()
                else:
                    history_hw = pd.concat([
                        history_hw, intervals_t["half_width"],
                    ])
                # Hysteresis: apply state machine to alert level
                raw_alert = diag["alert_level"]
                filtered_alert, w_adj, consecutive_alert = _apply_hysteresis(
                    raw_alert, diag["applied_scale"], prev_alert,
                    consecutive_alert, last_upgrade_idx, i, rules, w_adj, w_target,
                )
                diag["alert_level"] = filtered_alert
                if filtered_alert != raw_alert:
                    diag["applied_scale"] = 1.0 if filtered_alert == "green" else diag["applied_scale"]
                # Update hysteresis state
                if ALERT_RANK[filtered_alert] > ALERT_RANK[prev_alert]:
                    if consecutive_alert >= rules.confirm_threshold:
                        last_upgrade_idx = i
                prev_alert = filtered_alert

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
            # Update history_hw daily for proper rolling z-score
            if is_sector:
                intervals_t_daily = _build_sector_intervals_t(
                    per_asset_intervals, w_target, date, history_hw,
                )
            else:
                if date in intervals["half_width"].index:
                    idx_t = intervals["half_width"].index.get_loc(date)
                    intervals_t_daily = {
                        "half_width": intervals["half_width"].iloc[[idx_t]],
                    }
                else:
                    intervals_t_daily = None
            if intervals_t_daily is not None:
                if history_hw is None:
                    history_hw = intervals_t_daily["half_width"].copy()
                else:
                    history_hw = pd.concat([
                        history_hw, intervals_t_daily["half_width"],
                    ])

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
            "adx": diag.get("adx_value", 0.0),
            "adx_scale": diag.get("adx_scale", 1.0),
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
