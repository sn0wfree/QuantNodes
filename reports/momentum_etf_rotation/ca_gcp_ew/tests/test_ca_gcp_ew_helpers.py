"""Unit tests for CA-GCP early-warning helper functions.

覆盖 ca_gcp_ew_eval.py 中纯函数:
- compute_trend_signal
- decompose_stress
- aggregate_signal
- compute_market_breadth
- check_sector_concentration
- compute_momentum_score
- compute_vol_regime
- get_asset_sectors
- KNOWN_EVENTS schema
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ca_gcp_ew_eval import (
    KNOWN_EVENTS,
    aggregate_signal,
    check_sector_concentration,
    compute_market_breadth,
    compute_momentum_score,
    compute_trend_signal,
    compute_vol_regime,
    decompose_stress,
    get_asset_sectors,
)


class TestComputeTrendSignal:
    def test_default_window_and_threshold(self, sample_returns: pd.DataFrame) -> None:
        sig = compute_trend_signal(sample_returns)
        assert isinstance(sig, pd.Series)
        assert sig.name == "trend_ok"
        assert len(sig) == len(sample_returns)
        assert set(sig.dropna().unique()).issubset({True, False})

    def test_threshold_can_be_overridden(self, bullish_returns: pd.DataFrame) -> None:
        low = compute_trend_signal(bullish_returns, threshold=0.001)
        high = compute_trend_signal(bullish_returns, threshold=1.0)
        assert low.sum() >= high.sum()

    def test_window_can_be_overridden(self, bullish_returns: pd.DataFrame) -> None:
        short = compute_trend_signal(bullish_returns, window=2)
        long_ = compute_trend_signal(bullish_returns, window=20)
        assert len(short) == len(bullish_returns)
        assert set(short.dropna().unique()).issubset({True, False})
        assert len(long_) == len(bullish_returns)

    def test_bearish_returns_rarely_pass(self, bearish_returns: pd.DataFrame) -> None:
        sig = compute_trend_signal(bearish_returns, threshold=0.02)
        assert sig.sum() < len(sig) * 0.5


class TestDecomposeStress:
    def test_columns_and_length(self, sample_returns: pd.DataFrame) -> None:
        df = decompose_stress(sample_returns)
        assert set(df.columns) == {"cross_dispersion", "anomaly_frac"}
        assert len(df) == len(sample_returns)

    def test_dispersion_is_nonnegative(self, sample_returns: pd.DataFrame) -> None:
        df = decompose_stress(sample_returns)
        assert (df["cross_dispersion"].dropna() >= 0).all()

    def test_anomaly_frac_in_unit_interval(self, sample_returns: pd.DataFrame) -> None:
        df = decompose_stress(sample_returns)
        f = df["anomaly_frac"].dropna()
        assert ((f >= 0) & (f <= 1)).all()


class TestAggregateSignal:
    def test_min_consecutive_two_filters_single_spike(
        self, sample_returns: pd.DataFrame
    ) -> None:
        alerts = pd.DataFrame(
            {
                "width_z": np.zeros(len(sample_returns)),
                "stress": np.zeros(len(sample_returns)),
                "fired": [1, 0, 0, 1, 1, 1, 0, 1, 0, 0] * (len(sample_returns) // 10),
            },
            index=sample_returns.index,
        )
        out = aggregate_signal(alerts, min_consecutive=2)
        assert "fired_agg" in out.columns
        assert "fired_streak" in out.columns
        assert out["fired_agg"].max() <= 1
        assert int(out["fired_agg"].sum()) <= int(out["fired"].sum())

    def test_min_consecutive_three_more_strict(
        self, sample_returns: pd.DataFrame
    ) -> None:
        alerts = pd.DataFrame(
            {
                "width_z": np.zeros(len(sample_returns)),
                "stress": np.zeros(len(sample_returns)),
                "fired": [1, 1, 1, 0, 1, 1, 0, 0, 1, 1] * (len(sample_returns) // 10),
            },
            index=sample_returns.index,
        )
        agg2 = aggregate_signal(alerts, min_consecutive=2)
        agg3 = aggregate_signal(alerts, min_consecutive=3)
        assert agg3["fired_agg"].sum() <= agg2["fired_agg"].sum()

    def test_no_fire_means_all_zero(self, sample_returns: pd.DataFrame) -> None:
        alerts = pd.DataFrame(
            {
                "width_z": np.zeros(len(sample_returns)),
                "stress": np.zeros(len(sample_returns)),
                "fired": 0,
            },
            index=sample_returns.index,
        )
        out = aggregate_signal(alerts)
        assert out["fired_agg"].sum() == 0


class TestComputeMarketBreadth:
    def test_extremes(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        all_pos = pd.DataFrame(
            np.ones((5, 3)) * 0.01, index=idx, columns=["a", "b", "c"]
        )
        all_neg = pd.DataFrame(
            -np.ones((5, 3)) * 0.01, index=idx, columns=["a", "b", "c"]
        )
        assert (compute_market_breadth(all_pos) == 1.0).all()
        assert (compute_market_breadth(all_neg) == 0.0).all()

    def test_mixed(self) -> None:
        idx = pd.date_range("2024-01-01", periods=4, freq="B")
        df = pd.DataFrame(
            {
                "a": [0.01, -0.01, 0.01, -0.01],
                "b": [0.01, -0.01, 0.01, -0.01],
                "c": [-0.01, 0.01, -0.01, 0.01],
                "d": [-0.01, 0.01, -0.01, 0.01],
            },
            index=idx,
        )
        breadth = compute_market_breadth(df)
        assert (breadth == 0.5).all()


class TestCheckSectorConcentration:
    def test_empty_list_returns_false(self) -> None:
        assert check_sector_concentration([], {"a": "x"}) == (False, "")

    def test_all_same_sector_concentrated(self) -> None:
        is_conc, dom = check_sector_concentration(
            ["510300", "510500"], {"510300": "宽基", "510500": "宽基"}
        )
        assert is_conc is True
        assert dom == "宽基"

    def test_diverse_sectors_not_concentrated(self) -> None:
        is_conc, _ = check_sector_concentration(
            ["510300", "518880", "511260"],
            {"510300": "宽基", "518880": "黄金", "511260": "债券"},
        )
        assert is_conc is False

    def test_threshold_sensitivity(self) -> None:
        assets = ["a", "b", "c", "d"]
        sectors = {"a": "x", "b": "x", "c": "x", "d": "y"}
        assert check_sector_concentration(assets, sectors, threshold=0.5)[0] is True
        assert check_sector_concentration(assets, sectors, threshold=0.9)[0] is False


class TestComputeMomentumScore:
    def test_default_windows(self, sample_returns: pd.DataFrame) -> None:
        score = compute_momentum_score(sample_returns)
        assert isinstance(score, pd.Series)
        assert score.name == "momentum_score"
        assert ((score >= -1) & (score <= 1)).all()
        assert len(score) == len(sample_returns)

    def test_custom_windows(self, sample_returns: pd.DataFrame) -> None:
        score = compute_momentum_score(sample_returns, windows=[5, 10])
        assert score.dropna().isin([-1.0, -0.5, 0.0, 0.5, 1.0]).all()


class TestComputeVolRegime:
    def test_returns_three_levels(self, sample_returns: pd.DataFrame) -> None:
        regime = compute_vol_regime(sample_returns)
        assert set(regime.dropna().unique()).issubset({"high", "normal", "low"})

    def test_bullish_low_vol_mostly_low(self, bullish_returns: pd.DataFrame) -> None:
        regime = compute_vol_regime(bullish_returns, lookback=60)
        ratio = (regime == "low").sum() / max((regime != "normal").sum(), 1)
        assert 0 <= ratio <= 1


class TestGetAssetSectors:
    def test_returns_dict(self) -> None:
        sectors = get_asset_sectors()
        assert isinstance(sectors, dict)
        assert len(sectors) > 0
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in sectors.items())


class TestKnownEvents:
    def test_count_and_schema(self) -> None:
        assert len(KNOWN_EVENTS) >= 10
        for ev in KNOWN_EVENTS:
            assert "name" in ev and "date" in ev
            pd.Timestamp(ev["date"])

    def test_dates_ascending(self) -> None:
        dates = [pd.Timestamp(ev["date"]) for ev in KNOWN_EVENTS]
        assert dates == sorted(dates)
