"""Unit tests for CA-GCP early-warning evaluation metrics.

覆盖 ca_gcp_ew_eval.py 中评估与回测函数:
- compute_forward_stats
- evaluate_event_hits
- evaluate_precision_recall
- backtest_overlay
- distribution_stats
- compute_coverage_signals
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ca_gcp_ew_eval import (
    backtest_overlay,
    compute_coverage_signals,
    compute_forward_stats,
    distribution_stats,
    evaluate_event_hits,
    evaluate_precision_recall,
)


def _alerts_with_fired(index: pd.DatetimeIndex, n_fired: int) -> pd.DataFrame:
    wz = np.zeros(len(index))
    sv = np.zeros(len(index))
    fired = np.zeros(len(index), dtype=int)
    fired[:n_fired] = 1
    return pd.DataFrame(
        {"width_z": wz, "stress": sv, "fired": fired, "alert_level": "green"},
        index=index,
    )


class TestComputeForwardStats:
    def test_returns_expected_horizon_keys(
        self, sample_returns: pd.DataFrame
    ) -> None:
        alerts = _alerts_with_fired(sample_returns.index[:50], n_fired=5)
        fwd = compute_forward_stats(alerts, sample_returns)
        for h in (5, 10, 20):
            assert f"n_{h}d" in fwd
            assert f"avg_{h}d" in fwd
            assert f"med_{h}d" in fwd
            assert f"neg_rate_{h}d" in fwd
            assert f"min_{h}d" in fwd

    def test_no_fire_returns_empty(
        self, sample_returns: pd.DataFrame
    ) -> None:
        alerts = _alerts_with_fired(sample_returns.index, n_fired=0)
        fwd = compute_forward_stats(alerts, sample_returns)
        assert fwd == {}


class TestEvaluateEventHits:
    def test_columns_present(self, sample_returns: pd.DataFrame) -> None:
        alerts = _alerts_with_fired(sample_returns.index[:100], n_fired=3)
        df = evaluate_event_hits(alerts, sample_returns)
        assert "in_test_window" in df.columns
        assert "lead_days" in df.columns
        assert "warned_30d" in df.columns
        assert "n_warnings_20d_before" in df.columns
        assert "realized_dd_10d" in df.columns

    def test_in_window_event_has_lead_or_none(
        self, sample_returns: pd.DataFrame
    ) -> None:
        alerts = _alerts_with_fired(sample_returns.index, n_fired=5)
        df = evaluate_event_hits(alerts, sample_returns)
        in_window = df[df["in_test_window"]]
        for _, row in in_window.iterrows():
            assert pd.isna(row["lead_days"]) or row["lead_days"] >= 0

    def test_out_of_window_marked(self, sample_returns: pd.DataFrame) -> None:
        alerts = _alerts_with_fired(sample_returns.index, n_fired=3)
        df = evaluate_event_hits(alerts, sample_returns)
        out = df[~df["in_test_window"]]
        if len(out) > 0:
            assert out["warned_30d"].sum() == 0
            assert out["lead_days"].isna().all()
            assert out["realized_dd_10d"].isna().all()


class TestEvaluatePrecisionRecall:
    def test_keys_and_f1_in_range(
        self, sample_returns: pd.DataFrame
    ) -> None:
        alerts = _alerts_with_fired(sample_returns.index[:100], n_fired=5)
        alerts["alert_level"] = "yellow"
        pr = evaluate_precision_recall(alerts, sample_returns, label="test")
        expected_keys = {
            "label", "n_fired", "TP", "FP", "FN", "neutral",
            "n_events_in_window", "precision", "recall", "f1", "hit_rate",
        }
        assert expected_keys.issubset(set(pr.keys()))
        assert 0.0 <= pr["f1"] <= 1.0
        assert 0.0 <= pr["precision"] <= 1.0
        assert 0.0 <= pr["recall"] <= 1.0

    def test_no_fire_zero_tp_fp(
        self, sample_returns: pd.DataFrame
    ) -> None:
        alerts = _alerts_with_fired(sample_returns.index, n_fired=0)
        pr = evaluate_precision_recall(alerts, sample_returns, label="zero")
        assert pr["TP"] == 0
        assert pr["FP"] == 0
        assert pr["n_fired"] == 0


class TestBacktestOverlay:
    def test_returns_required_keys(
        self, sample_returns: pd.DataFrame
    ) -> None:
        n = len(sample_returns)
        alerts = pd.DataFrame(
            {
                "alert_level": (["green"] * (n - 5) + ["red"] * 5),
                "fired": [0] * (n - 5) + [1] * 5,
            },
            index=sample_returns.index,
        )
        result = backtest_overlay(alerts, sample_returns)
        expected = {
            "baseline_cagr", "baseline_vol", "baseline_sharpe", "baseline_maxdd", "baseline_calmar",
            "overlay_cagr", "overlay_vol", "overlay_sharpe", "overlay_maxdd", "overlay_calmar",
            "diff_final",
        }
        assert expected.issubset(set(result.keys()))

    def test_red_overlay_reduces_drawdown_or_neutral(
        self, sample_returns: pd.DataFrame
    ) -> None:
        n = len(sample_returns)
        alerts_all_red = pd.DataFrame(
            {"alert_level": ["red"] * n, "fired": [1] * n},
            index=sample_returns.index,
        )
        alerts_all_green = pd.DataFrame(
            {"alert_level": ["green"] * n, "fired": [0] * n},
            index=sample_returns.index,
        )
        red_result = backtest_overlay(alerts_all_red, sample_returns)
        green_result = backtest_overlay(alerts_all_green, sample_returns)
        assert red_result["overlay_maxdd"] >= green_result["overlay_maxdd"]


class TestDistributionStats:
    def test_pct_sums_to_one(self, sample_returns: pd.DataFrame) -> None:
        n = len(sample_returns)
        levels = (["green"] * (n // 2) + ["yellow"] * (n // 4) + ["red"] * (n - n // 2 - n // 4))
        alerts = pd.DataFrame({"alert_level": levels}, index=sample_returns.index)
        dist = distribution_stats(alerts)
        pct_sum = dist["green_pct"] + dist["yellow_pct"] + dist["red_pct"]
        assert abs(pct_sum - 1.0) < 1e-9

    def test_counts_match(self, sample_returns: pd.DataFrame) -> None:
        n = len(sample_returns)
        levels = (["green"] * (n - 5) + ["yellow"] * 3 + ["red"] * 2)
        alerts = pd.DataFrame({"alert_level": levels}, index=sample_returns.index)
        dist = distribution_stats(alerts)
        assert dist["green_count"] + dist["yellow_count"] + dist["red_count"] == dist["n_days"]


class TestComputeCoverageSignals:
    def test_returns_required_columns(
        self,
        sample_returns: pd.DataFrame,
        sample_hw: pd.DataFrame,
        sample_lower_upper: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        lower, upper = sample_lower_upper
        intervals = {"lower": lower, "upper": upper}
        df = compute_coverage_signals(intervals, sample_returns)
        expected = {"coverage", "coverage_gap", "downside_violation", "upside_violation"}
        assert expected.issubset(set(df.columns))

    def test_coverage_in_unit_interval(
        self,
        sample_returns: pd.DataFrame,
        sample_lower_upper: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        lower, upper = sample_lower_upper
        df = compute_coverage_signals({"lower": lower, "upper": upper}, sample_returns)
        cov = df["coverage"].dropna()
        assert ((cov >= 0) & (cov <= 1)).all()

    def test_missing_keys_raises(
        self, sample_returns: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError):
            compute_coverage_signals({}, sample_returns)
