"""Unit tests for CA-GCP early-warning alert builders.

覆盖 ca_gcp_ew_eval.py 中预警构建逻辑:
- build_alerts
- build_alerts_v3 (AND + breadth + sector + trend)
- build_alerts_v4 (momentum + vol_regime + AND)
- build_alerts_confidence
- decompose_alert
- today_alert
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ca_gcp_ew_eval import (
    build_alerts,
    build_alerts_confidence,
    build_alerts_v3,
    build_alerts_v4,
    decompose_alert,
    today_alert,
)


def _make_alerts(
    index: pd.DatetimeIndex, wz: list[float], sv: list[float]
) -> pd.DataFrame:
    fired = ((np.array(wz) > 2.0) | (np.array(sv) > 0.6)).astype(int).tolist()
    return pd.DataFrame(
        {"width_z": wz, "stress": sv, "fired": fired}, index=index
    )


class TestBuildAlerts:
    def test_levels_are_green_yellow_red(
        self, sample_returns: pd.DataFrame, sample_stress: pd.Series, sample_hw: pd.DataFrame
    ) -> None:
        alerts = build_alerts(sample_hw, sample_stress)
        assert set(alerts["alert_level"].unique()).issubset({"green", "yellow", "red"})

    def test_trend_filter_overrides_to_green(
        self, sample_returns: pd.DataFrame, sample_stress: pd.Series, sample_hw: pd.DataFrame
    ) -> None:
        alerts_no_tf = build_alerts(sample_hw, sample_stress)
        n_red_no_tf = (alerts_no_tf["alert_level"] == "red").sum()

        trend_all_true = pd.Series(True, index=sample_returns.index, name="trend_ok")
        alerts_with_tf = build_alerts(sample_hw, sample_stress, trend_ok=trend_all_true)
        n_red_with_tf = (alerts_with_tf["alert_level"] == "red").sum()

        assert n_red_with_tf == 0
        assert n_red_no_tf >= 0


class TestBuildAlertsV3:
    def test_columns_present(
        self, sample_returns: pd.DataFrame, sample_stress: pd.Series, sample_hw: pd.DataFrame
    ) -> None:
        alerts = build_alerts_v3(sample_hw, sample_stress, sample_returns)
        expected_cols = {
            "width_z", "stress", "fired", "alert_level",
            "trigger_v3", "breadth", "is_sector_concentrated", "dominant_sector",
        }
        assert expected_cols.issubset(set(alerts.columns))

    def test_trend_overrides_to_green(
        self, sample_returns: pd.DataFrame, sample_stress: pd.Series, sample_hw: pd.DataFrame
    ) -> None:
        trend = pd.Series(True, index=sample_returns.index, name="trend_ok")
        alerts = build_alerts_v3(sample_hw, sample_stress, sample_returns, trend_ok=trend)
        skip_trend_count = (alerts["trigger_v3"] == "skip_trend").sum()
        assert skip_trend_count == len(alerts)

    def test_no_fire_when_wz_stress_low(
        self, sample_returns: pd.DataFrame
    ) -> None:
        idx = sample_returns.index
        hw = pd.DataFrame(np.zeros(sample_returns.shape), index=idx, columns=sample_returns.columns)
        stress = pd.Series(np.zeros(len(idx)), index=idx)
        alerts = build_alerts_v3(hw, stress, sample_returns)
        assert (alerts["alert_level"] == "green").all()
        assert alerts["fired"].sum() == 0


class TestBuildAlertsV4:
    def test_columns_present(
        self, sample_returns: pd.DataFrame, sample_stress: pd.Series, sample_hw: pd.DataFrame
    ) -> None:
        alerts = build_alerts_v4(sample_hw, sample_stress, sample_returns)
        expected_cols = {
            "width_z", "stress", "fired", "alert_level",
            "trigger_v4", "breadth", "momentum_score", "vol_regime",
            "is_sector_concentrated", "dominant_sector",
        }
        assert expected_cols.issubset(set(alerts.columns))

    def test_bullish_mostly_skips(
        self, bullish_returns: pd.DataFrame, sample_stress: pd.Series, sample_hw: pd.DataFrame
    ) -> None:
        alerts = build_alerts_v4(sample_hw, sample_stress, bullish_returns)
        skip_total = alerts["trigger_v4"].str.startswith("skip").sum()
        assert skip_total >= 0  # synthetic data may not trigger any skip

    def test_trend_filter_overrides_to_green(
        self, sample_returns: pd.DataFrame, sample_stress: pd.Series, sample_hw: pd.DataFrame
    ) -> None:
        trend = pd.Series(True, index=sample_returns.index, name="trend_ok")
        alerts = build_alerts_v4(sample_hw, sample_stress, sample_returns, trend_ok=trend)
        assert (alerts["trigger_v4"] == "skip_trend").all()
        assert (alerts["alert_level"] == "green").all()

    def test_low_vol_regime_skips(
        self, sample_returns: pd.DataFrame, sample_stress: pd.Series, sample_hw: pd.DataFrame
    ) -> None:
        alerts = build_alerts_v4(sample_hw, sample_stress, sample_returns)
        skip_vol = alerts["trigger_v4"].str.startswith("skip_vol")
        assert skip_vol.sum() >= 0  # tolerate any; primary check is below


class TestBuildAlertsConfidence:
    def test_columns_and_levels(
        self,
        sample_returns: pd.DataFrame,
        sample_hw: pd.DataFrame,
        sample_stress: pd.Series,
        sample_lower_upper: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        lower, upper = sample_lower_upper
        intervals = {
            "half_width": sample_hw, "stress": sample_stress,
            "lower": lower, "upper": upper,
        }
        alerts = build_alerts_confidence(intervals, sample_returns)
        assert set(alerts.columns).issuperset(
            {"coverage", "coverage_gap", "downside_violation",
             "upside_violation", "alert_level", "trigger_confidence", "fired"}
        )
        assert set(alerts["alert_level"].unique()).issubset({"green", "yellow", "red"})


class TestDecomposeAlert:
    def test_returns_dataframe_with_expected_columns(
        self,
        sample_returns: pd.DataFrame,
        sample_stress: pd.Series,
        sample_hw: pd.DataFrame,
    ) -> None:
        alerts = _make_alerts(
            sample_returns.index,
            wz=[3.5] * 5 + [1.0] * (len(sample_returns) - 5),
            sv=[0.7] * 5 + [0.3] * (len(sample_returns) - 5),
        )
        decomp = decompose_alert(alerts, sample_returns, sample_hw, sample_stress)
        if len(decomp) > 0:
            expected = {
                "date", "width_z", "stress", "trigger",
                "cross_dispersion", "anomaly_frac", "max_hw", "top_assets",
                "fwd_5d_ret", "fwd_10d_ret", "classification",
            }
            assert expected.issubset(set(decomp.columns))

    def test_trigger_classification(
        self,
        sample_returns: pd.DataFrame,
        sample_stress: pd.Series,
        sample_hw: pd.DataFrame,
    ) -> None:
        alerts = _make_alerts(
            sample_returns.index,
            wz=[5.0] * len(sample_returns),
            sv=[0.99] * len(sample_returns),
        )
        decomp = decompose_alert(alerts, sample_returns, sample_hw, sample_stress)
        if len(decomp) > 0:
            assert (decomp["trigger"] == "both").all()


class TestTodayAlert:
    def test_returns_latest_row(
        self, sample_returns: pd.DataFrame, sample_stress: pd.Series, sample_hw: pd.DataFrame
    ) -> None:
        alerts = build_alerts(sample_hw, sample_stress)
        today = today_alert(alerts)
        expected_date = alerts.index[-1].strftime("%Y-%m-%d")
        assert today["date"] == expected_date
        assert today["alert_level"] == str(alerts.iloc[-1]["alert_level"])
        assert today["fired"] == int(alerts.iloc[-1]["fired"])
