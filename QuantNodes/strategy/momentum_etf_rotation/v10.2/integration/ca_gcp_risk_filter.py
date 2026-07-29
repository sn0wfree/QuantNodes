"""v10.2 CA-GCP risk-filter hook.

Applies CA-GCP prediction intervals as a position-sizing adjustment:
  - Width explosion (> threshold) → reduce position size
  - System pressure (γ > 0.6) → pause new entries / scale down
  - Neighbor divergence (> threshold) → reduce most volatile legs

The function is independent of v10 source; it accepts weights and returns
adjusted weights, preserving v10's relative ranking.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import sys
from pathlib import Path

_V102 = Path(__file__).resolve().parent.parent
if str(_V102) not in sys.path:
    sys.path.insert(0, str(_V102))

from QuantNodes.strategy.momentum_etf_rotation.common.ca_gcp import CAGCPConfig, CAGCPipeline  # noqa: E402


@dataclass
class RiskFilterRules:
    """Threshold rules for v10.2 risk overlay.

    Supports per-group thresholds via group_rules and asset_groups.
    If group_rules is None, uses global thresholds for all assets.

    Hysteresis fields:
        confirm_threshold: consecutive days needed to upgrade alert level.
        min_hold_days: minimum days to hold after upgrading.
        cooldown_days: minimum days between upgrades.
    """

    width_z_yellow: float = 3.0
    width_z_red: float = 4.5
    stress_yellow: float = 0.92
    stress_red: float = 0.98
    yellow_scale: float = 0.85
    red_scale: float = 0.6
    panic_scale: float = 0.3
    stress_yellow_recovery: float = 0.80
    width_z_yellow_recovery: float = 2.0
    # Hysteresis
    confirm_threshold: int = 2
    min_hold_days: int = 5
    cooldown_days: int = 3
    # Trend filter: skip alerts during uptrend
    trend_filter: bool = True
    trend_window: int = 20
    trend_threshold: float = 0.0
    # ADX trend strength filter
    adx_filter: bool = False
    adx_period: int = 14
    adx_low: float = 15.0
    adx_high: float = 40.0
    adx_min_scale: float = 0.5
    # Multi-signal fusion
    fusion_enabled: bool = False
    fusion_w_trend: float = 0.5
    fusion_w_risk: float = 0.5
    fusion_min_scale: float = 0.3
    # Per-group support
    group_rules: dict[str, "RiskFilterRules"] | None = None
    asset_groups: dict[str, list[str]] | None = None


def experimental_rules() -> RiskFilterRules:
    """More aggressive thresholds for stress-test experiments."""
    return RiskFilterRules(
        width_z_yellow=2.0,
        width_z_red=3.0,
        stress_yellow=0.6,
        stress_red=0.85,
    )


def ca_gcp_risk_filter(
    weights: pd.Series,
    intervals: dict[str, pd.DataFrame],
    rules: RiskFilterRules | None = None,
    today: pd.Timestamp | None = None,
    history: pd.DataFrame | None = None,
    residual_asset: str | None = None,
    trend_signal: pd.Series | None = None,
    adx_value: float | None = None,
    ma_trend_sign: float | None = None,
    vol_pct: float | None = None,
) -> tuple[pd.Series, dict]:
    """Apply CA-GCP risk filter to v10 target weights.

    Supports two modes:
      - Global: single threshold for all assets (group_rules=None)
      - Per-group: separate thresholds per asset group

    Args:
        weights: v10 target weights (indexed by asset code).
        intervals: dict from CAGCPipeline.predict() with keys:
            'lower', 'upper', 'half_width', 'thresholds', 'stress'.
        rules: RiskFilterRules configuration.
        today: Specific day to evaluate. If None, uses last day of intervals.
        history: Earlier half_width values for computing rolling stats.
        residual_asset: Asset code to receive residual exposure when scale < 1.0.
        trend_signal: Boolean Series (True=downtrend). If today is uptrend, skip alerts.
        adx_value: Current ADX value for trend-strength position scaling.
        ma_trend_sign: Moving average trend direction (+1=up, -1=down). For fusion.
        vol_pct: Volatility percentile [0, 1]. For fusion.

    Returns:
        (adjusted_weights, diagnostics_dict)
    """
    rules = rules or RiskFilterRules()

    # Per-group mode
    if rules.group_rules and rules.asset_groups:
        return _ca_gcp_risk_filter_grouped(
            weights, intervals, rules, today, history,
            residual_asset=residual_asset,
            trend_signal=trend_signal,
            adx_value=adx_value,
            ma_trend_sign=ma_trend_sign,
            vol_pct=vol_pct,
        )

    # Global mode (original logic)
    return _ca_gcp_risk_filter_global(
        weights, intervals, rules, today, history,
        residual_asset=residual_asset,
        trend_signal=trend_signal,
        adx_value=adx_value,
        ma_trend_sign=ma_trend_sign,
        vol_pct=vol_pct,
    )


def _compute_width_z(
    hw: pd.DataFrame,
    history: pd.DataFrame | None,
    today: pd.Timestamp,
) -> float:
    """Compute width z-score for today."""
    if history is not None:
        hw_full = pd.concat([history, hw])
        # Deduplicate by index (keep last occurrence per date)
        hw_full = hw_full[~hw_full.index.duplicated(keep='last')]
    else:
        hw_full = hw
    width_ts = (2.0 * hw_full).mean(axis=1)
    roll = width_ts.rolling(60, min_periods=10)
    width_z_full = (width_ts - roll.mean()) / roll.std().replace(0, np.nan)

    if today in width_z_full.index:
        val = width_z_full.loc[today]
    else:
        val = width_z_full.iloc[-1]
    return float(val) if pd.notna(val) else 0.0


def _ca_gcp_risk_filter_global(
    weights: pd.Series,
    intervals: dict[str, pd.DataFrame],
    rules: RiskFilterRules,
    today: pd.Timestamp | None,
    history: pd.DataFrame | None,
    residual_asset: str | None = None,
    trend_signal: pd.Series | None = None,
    adx_value: float | None = None,
    ma_trend_sign: float | None = None,
    vol_pct: float | None = None,
) -> tuple[pd.Series, dict]:
    """Global mode: single threshold for all assets."""
    hw = intervals["half_width"]
    stress = intervals["stress"]

    if today is None:
        today = hw.index[-1]

    width_z_today = _compute_width_z(hw, history, today)

    if today in stress.index:
        stress_today = float(stress.loc[today])
    else:
        stress_today = float(stress.iloc[-1])

    diag: dict = {
        "width_z_today": width_z_today,
        "stress_today": stress_today,
        "alert_level": "green",
    }
    scale = 1.0
    if width_z_today > rules.width_z_red or stress_today > rules.stress_red:
        scale = rules.red_scale
        diag["alert_level"] = "red"
    elif width_z_today > rules.width_z_yellow or stress_today > rules.stress_yellow:
        scale = rules.yellow_scale
        diag["alert_level"] = "yellow"

    # Trend filter: skip alerts during uptrend
    if rules.trend_filter and trend_signal is not None:
        if today in trend_signal.index and not trend_signal.loc[today]:
            diag["alert_level"] = "green"
            scale = 1.0

    # ADX trend-strength scaling (multiplied on top of CA-GCP scale)
    if rules.adx_filter and adx_value is not None:
        adx_scale = adx_position_scale(
            adx_value, rules.adx_low, rules.adx_high, rules.adx_min_scale,
        )
        scale = scale * adx_scale
        diag["adx_value"] = adx_value
        diag["adx_scale"] = adx_scale
    else:
        diag["adx_value"] = 0.0
        diag["adx_scale"] = 1.0

    # Multi-signal fusion (replaces rule-based scale if enabled)
    if rules.fusion_enabled:
        alert_rank = {"green": 0, "yellow": 0.5, "red": 1.0}
        cagcp_score = alert_rank.get(diag["alert_level"], 0.0)
        adx_norm = (adx_value / 100.0) if adx_value is not None else 0.5
        fused_scale, fused_alert = _apply_fusion_to_diag(
            diag, rules, ma_trend_sign, adx_norm, vol_pct, cagcp_score,
        )
        scale = fused_scale
        diag["alert_level"] = fused_alert

    diag["applied_scale"] = scale
    adjusted = weights * scale

    if scale < 1.0:
        residual = (1.0 - adjusted.sum()) if adjusted.sum() < 1.0 else 0.0
        if residual > 0:
            if residual_asset is not None and residual_asset in adjusted.index:
                adjusted[residual_asset] += residual
            else:
                weights_min = weights.abs().idxmin()
                adjusted[weights_min] += residual

    return adjusted, diag


def _apply_fusion_to_diag(
    diag: dict,
    rules: RiskFilterRules,
    ma_trend_sign: float | None,
    adx_norm: float | None,
    vol_pct: float | None,
    cagcp_score: float,
) -> tuple[float, str]:
    """Apply multi-signal fusion and update diagnostics.

    Returns (fused_scale, fused_alert_level).
    """
    if not rules.fusion_enabled:
        return diag.get("applied_scale", 1.0), diag.get("alert_level", "green")

    ma_sign = ma_trend_sign if ma_trend_sign is not None else 0.0
    adx_n = adx_norm if adx_norm is not None else 0.5
    vol_p = vol_pct if vol_pct is not None else 0.5

    fusion_score = compute_fusion_score(
        ma_sign, adx_n, vol_p, cagcp_score,
        rules.fusion_w_trend, rules.fusion_w_risk,
    )
    fused_scale = fusion_score_to_scale(fusion_score, rules.fusion_min_scale)

    if fusion_score > 0.3:
        fused_alert = "green"
    elif fusion_score > -0.2:
        fused_alert = "yellow"
    else:
        fused_alert = "red"

    diag["fusion_score"] = fusion_score
    diag["ma_trend_sign"] = ma_sign
    diag["adx_norm"] = adx_n
    diag["vol_pct"] = vol_p
    diag["cagcp_score"] = cagcp_score

    return fused_scale, fused_alert


def _ca_gcp_risk_filter_grouped(
    weights: pd.Series,
    intervals: dict[str, pd.DataFrame],
    rules: RiskFilterRules,
    today: pd.Timestamp | None,
    history: pd.DataFrame | None,
    residual_asset: str | None = None,
    trend_signal: pd.Series | None = None,
    adx_value: float | None = None,
    ma_trend_sign: float | None = None,
    vol_pct: float | None = None,
) -> tuple[pd.Series, dict]:
    """Per-group mode: separate thresholds per asset group."""
    hw = intervals["half_width"]
    stress = intervals["stress"]

    if today is None:
        today = hw.index[-1]

    if today in stress.index:
        stress_today = float(stress.loc[today])
    else:
        stress_today = float(stress.iloc[-1])

    # Compute per-group width_z
    group_alerts: dict[str, str] = {}
    group_scales: dict[str, float] = {}
    group_width_z: dict[str, float] = {}

    for group_name, group_assets in rules.asset_groups.items():
        group_hw_cols = [c for c in group_assets if c in hw.columns]
        if not group_hw_cols:
            continue

        group_hw = hw[group_hw_cols]
        group_history = None
        if history is not None:
            group_history_cols = [c for c in group_assets if c in history.columns]
            if group_history_cols:
                group_history = history[group_history_cols]

        wz = _compute_width_z(group_hw, group_history, today)
        group_width_z[group_name] = wz

        grp_rules = rules.group_rules.get(group_name, rules)
        alert = "green"
        scale = 1.0
        if wz > grp_rules.width_z_red or stress_today > grp_rules.stress_red:
            scale = grp_rules.red_scale
            alert = "red"
        elif wz > grp_rules.width_z_yellow or stress_today > grp_rules.stress_yellow:
            scale = grp_rules.yellow_scale
            alert = "yellow"

        group_alerts[group_name] = alert
        group_scales[group_name] = scale

    # Determine overall alert level (worst across groups)
    overall_alert = "green"
    overall_scale = 1.0
    if "red" in group_alerts.values():
        overall_alert = "red"
        # Use weighted average scale for red groups
        total_weight = 0.0
        weighted_scale = 0.0
        for grp, sc in group_scales.items():
            grp_assets = rules.asset_groups.get(grp, [])
            grp_w = sum(weights.get(a, 0.0) for a in grp_assets)
            if sc < 1.0:
                weighted_scale += sc * grp_w
                total_weight += grp_w
        if total_weight > 0:
            overall_scale = weighted_scale / total_weight
    elif "yellow" in group_alerts.values():
        overall_alert = "yellow"
        total_weight = 0.0
        weighted_scale = 0.0
        for grp, sc in group_scales.items():
            grp_assets = rules.asset_groups.get(grp, [])
            grp_w = sum(weights.get(a, 0.0) for a in grp_assets)
            if sc < 1.0:
                weighted_scale += sc * grp_w
                total_weight += grp_w
        if total_weight > 0:
            overall_scale = weighted_scale / total_weight

    # Trend filter: skip alerts during uptrend
    if rules.trend_filter and trend_signal is not None:
        if today in trend_signal.index and not trend_signal.loc[today]:
            overall_alert = "green"
            overall_scale = 1.0

    # ADX trend-strength scaling
    if rules.adx_filter and adx_value is not None:
        adx_scale = adx_position_scale(
            adx_value, rules.adx_low, rules.adx_high, rules.adx_min_scale,
        )
        overall_scale = overall_scale * adx_scale
    else:
        adx_scale = 1.0

    # Multi-signal fusion (replaces rule-based scale if enabled)
    if rules.fusion_enabled:
        alert_rank = {"green": 0, "yellow": 0.5, "red": 1.0}
        cagcp_score = alert_rank.get(overall_alert, 0.0)
        adx_norm = (adx_value / 100.0) if adx_value is not None else 0.5
        fused_scale, fused_alert = _apply_fusion_to_diag(
            {}, rules, ma_trend_sign, adx_norm, vol_pct, cagcp_score,
        )
        overall_scale = fused_scale
        overall_alert = fused_alert

    diag: dict = {
        "width_z_today": group_width_z,
        "stress_today": stress_today,
        "alert_level": overall_alert,
        "applied_scale": overall_scale,
        "group_alerts": group_alerts,
        "group_scales": group_scales,
        "adx_value": adx_value if adx_value is not None else 0.0,
        "adx_scale": adx_scale,
    }

    adjusted = weights * overall_scale
    if overall_scale < 1.0:
        residual = (1.0 - adjusted.sum()) if adjusted.sum() < 1.0 else 0.0
        if residual > 0:
            if residual_asset is not None and residual_asset in adjusted.index:
                adjusted[residual_asset] += residual
            else:
                weights_min = weights.abs().idxmin()
                adjusted[weights_min] += residual

    return adjusted, diag


def build_v10_2_pipeline(
    returns_history: pd.DataFrame,
    config: CAGCPConfig | None = None,
) -> CAGCPipeline:
    """Build a fresh CA-GCP pipeline from training returns."""
    pipe = CAGCPipeline(config)
    pipe.fit(returns_history)
    return pipe


# --- Threshold calibration ---

WIDTH_Z_YELLOW_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
WIDTH_Z_RED_GRID = [2.0, 2.5, 3.0, 3.5, 4.0]
STRESS_YELLOW_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
STRESS_RED_GRID = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9]


def calibrate_risk_filter(
    daily_prices: pd.DataFrame,
    weekly_prices: pd.DataFrame,
    etf_returns: pd.DataFrame,
    pipe,
    calib_start: pd.Timestamp,
    calib_end: pd.Timestamp,
    cost_bp: int = 10,
) -> RiskFilterRules:
    """Grid search for optimal risk filter thresholds on calib period.

    Pareto score measures marginal improvement over bare:
      score = (sharpe_cagcp - sharpe_bare)
            - cost_penalty
            + maxdd_bonus

    where:
      cost_penalty = 1.0 * max(0, ann_cost_cagcp - ann_cost_bare)
        (every incremental bp above bare is penalized; no dead-zone)
      maxdd_bonus  = 0.5 * max(0, maxdd_bare - maxdd_cagcp)
        (capped at +0.5 by MaxDD improvement, maxdd_bare is negative)

    Args:
        daily_prices: Daily close prices for the 4 assets.
        weekly_prices: Weekly close prices for signal.
        etf_returns: ETF daily returns.
        pipe: Fitted CAGCPipeline (global) or dict of sector pipelines.
        calib_start: Start of calibration period.
        calib_end: End of calibration period.
        cost_bp: Transaction cost in basis points.

    Returns:
        Best RiskFilterRules found.
    """
    import itertools
    import time

    from common.metrics import compute_metrics
    from .dual_momentum_ca_gcp import dual_momentum_bare, dual_momentum_with_ca_gcp

    calib_prices = daily_prices.loc[calib_start:calib_end]
    calib_weekly = weekly_prices.loc[:calib_end]

    # 1. Run bare baseline
    nav_bare, diag_bare = dual_momentum_bare(
        calib_prices, calib_weekly, cost_bp=cost_bp,
    )
    m_bare = compute_metrics(nav_bare)
    sharpe_bare = m_bare.get("Sharpe", 0.0)
    maxdd_bare = m_bare.get("MaxDD", 0.0)
    total_cost_bare = diag_bare["cost"].sum()
    n_years_bare = m_bare.get("Years", 1.0)
    ann_cost_bare = total_cost_bare / n_years_bare if n_years_bare > 0 else 0.0

    print(f"    Bare baseline: Sharpe={sharpe_bare:.3f}, MaxDD={maxdd_bare:.2%}, "
          f"AnnCost={ann_cost_bare:.2%}")

    # 2. Grid search over thresholds
    grid = list(itertools.product(
        WIDTH_Z_YELLOW_GRID, WIDTH_Z_RED_GRID,
        STRESS_YELLOW_GRID, STRESS_RED_GRID,
    ))
    total = len(grid)
    print(f"    Risk filter grid: {total} combos")

    best_rules = RiskFilterRules()
    best_score = -np.inf
    t0 = time.time()

    for idx, (wzy, wzr, sy, sr) in enumerate(grid, 1):
        if wzr <= wzy or sr <= sy:
            continue

        rules = RiskFilterRules(
            width_z_yellow=wzy, width_z_red=wzr,
            stress_yellow=sy, stress_red=sr,
        )

        nav, diag = dual_momentum_with_ca_gcp(
            calib_prices, calib_weekly, pipe, etf_returns,
            rules=rules, cost_bp=cost_bp,
        )

        if len(nav) < 20:
            continue

        m = compute_metrics(nav)
        sharpe_cagcp = m.get("Sharpe", 0.0)
        maxdd_cagcp = m.get("MaxDD", 0.0)

        total_cost_cagcp = diag["cost"].sum()
        n_years = m.get("Years", 1.0)
        ann_cost_cagcp = total_cost_cagcp / n_years if n_years > 0 else 0.0

        # Pareto score: marginal improvement over bare
        delta_sharpe = sharpe_cagcp - sharpe_bare
        cost_penalty = max(0.0, ann_cost_cagcp - ann_cost_bare)
        maxdd_bonus = max(0.0, maxdd_bare - maxdd_cagcp)  # maxdd is negative

        score = delta_sharpe - cost_penalty + 0.5 * maxdd_bonus

        if score > best_score:
            best_score = score
            best_rules = rules

        if idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"      [{idx}/{total}] best_score={best_score:.3f} "
                  f"delta_sharpe={delta_sharpe:+.3f} maxdd_bonus={maxdd_bonus:.3f} "
                  f"({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"    Risk filter calibration: {elapsed:.1f}s, "
          f"best_score={best_score:.3f}")
    print(f"    Best: wz_yellow={best_rules.width_z_yellow}, "
          f"wz_red={best_rules.width_z_red}, "
          f"stress_yellow={best_rules.stress_yellow}, "
          f"stress_red={best_rules.stress_red}")

    return best_rules


def compute_adx_close(close: pd.Series, period: int = 14) -> pd.Series:
    """Simplified ADX using only close prices (no OHLC).

    Approximations:
      - True Range ≈ |close - prev_close|
      - +DM ≈ max(close - prev_close, 0)  (positive return days)
      - -DM ≈ max(prev_close - close, 0)  (negative return days)

    This loses the high/low information but works for close-only data.
    The resulting ADX is lower than the standard one but captures the
    same trend-strength dynamics.

    Args:
        close: Daily close price series.
        period: ADX period (default 14).

    Returns:
        ADX series (same index as close, first 2*period values are NaN).
    """
    if len(close) < 2 * period:
        return pd.Series(np.nan, index=close.index)

    delta = close.diff()
    tr = delta.abs()
    plus_dm = delta.clip(lower=0.0)
    minus_dm = (-delta).clip(lower=0.0)

    alpha = 1.0 / period
    tr_smooth = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    plus_di = 100.0 * plus_dm_smooth / tr_smooth.replace(0, np.nan)
    minus_di = 100.0 * minus_dm_smooth / tr_smooth.replace(0, np.nan)

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    adx = adx.fillna(0.0).clip(upper=100.0)

    return adx


def adx_position_scale(
    adx_value: float,
    adx_low: float = 15.0,
    adx_high: float = 40.0,
    min_scale: float = 0.5,
) -> float:
    """Map ADX value to position scale factor.

    ADX < adx_low → min_scale (low trend confidence, reduce position)
    ADX > adx_high → 1.0 (strong trend, full position)
    In between → linear interpolation
    """
    if adx_value <= adx_low:
        return min_scale
    if adx_value >= adx_high:
        return 1.0
    ratio = (adx_value - adx_low) / (adx_high - adx_low)
    return min_scale + ratio * (1.0 - min_scale)


def compute_fusion_score(
    ma_trend_sign: float,
    adx_norm: float,
    vol_pct: float,
    cagcp_score: float,
    w_trend: float = 0.5,
    w_risk: float = 0.5,
) -> float:
    """Compute multi-signal fusion score.

    Args:
        ma_trend_sign: Moving average trend direction, -1 (downtrend) or +1 (uptrend).
        adx_norm: Normalized ADX in [0, 1] (ADX/100).
        vol_pct: Volatility percentile in [0, 1] (higher = more risky).
        cagcp_score: CA-GCP risk score in [0, 1] (0=green, 0.5=yellow, 1=red).
        w_trend: Weight for trend component.
        w_risk: Weight for risk component.

    Returns:
        Fusion score in [-1, +1]. Higher = more bullish.
    """
    trend_score = ma_trend_sign * adx_norm  # [-1, +1]
    risk_score = max(vol_pct, cagcp_score)  # [0, 1]

    w_total = w_trend + w_risk
    if w_total <= 0:
        return 0.0

    final_score = (w_trend * trend_score + w_risk * (-risk_score)) / w_total
    return float(np.clip(final_score, -1.0, 1.0))


def fusion_score_to_scale(score: float, min_scale: float = 0.3) -> float:
    """Map fusion score [-1, +1] to position scale [min_scale, 1.0].

    score = +1 → 1.0 (full position)
    score =  0 → (1 + min_scale) / 2 (mid position)
    score = -1 → min_scale (minimum position)
    """
    ratio = (score + 1.0) / 2.0  # [0, 1]
    return float(min_scale + ratio * (1.0 - min_scale))
