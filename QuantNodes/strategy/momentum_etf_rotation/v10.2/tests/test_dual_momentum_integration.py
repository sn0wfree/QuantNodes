"""Tests for dual_momentum + CA-GCP integration."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_V102 = Path(__file__).resolve().parent.parent
if str(_V102) not in sys.path:
    sys.path.insert(0, str(_V102))

from ca_gcp.core import CAGCPConfig, CAGCPipeline  # noqa: E402
from integration.ca_gcp_risk_filter import RiskFilterRules  # noqa: E402
from integration.dual_momentum_ca_gcp import (  # noqa: E402
    dual_momentum_bare,
    dual_momentum_signal,
    dual_momentum_with_ca_gcp,
)


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-02", periods=500, freq="B")
    prices = pd.DataFrame({
        "510300": rng.normal(0, 0.01, 500).cumsum() + 4,
        "513100": rng.normal(0, 0.01, 500).cumsum() + 4,
        "518880": rng.normal(0, 0.01, 500).cumsum() + 4,
        "511260": rng.normal(0, 0.003, 500).cumsum() + 3,
    }, index=dates)
    weekly = prices.resample("W-SUN").last().dropna()
    returns = prices.pct_change().fillna(0.0)
    return prices, weekly, returns


@pytest.fixture
def fitted_pipe(sample_data):
    _, _, returns = sample_data
    cfg = CAGCPConfig(k=2, sensitivity_eta=0.5, recency_tau=20.0)
    pipe = CAGCPipeline(cfg)
    pipe.fit(returns.iloc[:400])
    return pipe


class TestDualMomentumSignal:
    def test_returns_series_of_length_4(self, sample_data):
        _, weekly, _ = sample_data
        sig = dual_momentum_signal(weekly)
        assert isinstance(sig, pd.Series)
        assert len(sig) == 4

    def test_exactly_one_asset_gets_1_0(self, sample_data):
        _, weekly, _ = sample_data
        sig = dual_momentum_signal(weekly)
        assert sig.sum() == pytest.approx(1.0)
        assert set(sig.unique()).issubset({0.0, 1.0})

    def test_bond_fallback_when_all_negative(self, sample_data):
        prices, weekly, _ = sample_data
        # Force all 52-week returns negative by creating a downtrend
        rng = np.random.default_rng(99)
        crashed = pd.DataFrame({
            col: np.linspace(prices[col].iloc[0], prices[col].iloc[0] * 0.5, len(prices))
            + rng.normal(0, 0.01, len(prices))
            for col in prices.columns
        }, index=prices.index)
        wk_crash = crashed.resample("W-SUN").last().dropna()
        sig = dual_momentum_signal(wk_crash)
        assert sig["511260"] == 1.0


class TestDualMomentumBare:
    def test_bare_returns_tuple(self, sample_data):
        prices, weekly, _ = sample_data
        result = dual_momentum_bare(prices, weekly, test_start=prices.index[200])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_bare_nav_shape(self, sample_data):
        prices, weekly, _ = sample_data
        nav, _ = dual_momentum_bare(prices, weekly, test_start=prices.index[200])
        assert len(nav) == 300

    def test_bare_nav_starts_at_1(self, sample_data):
        prices, weekly, _ = sample_data
        nav, _ = dual_momentum_bare(prices, weekly, test_start=prices.index[200])
        assert nav.iloc[0] == pytest.approx(1.0)

    def test_bare_nav_positive(self, sample_data):
        prices, weekly, _ = sample_data
        nav, _ = dual_momentum_bare(prices, weekly, test_start=prices.index[200])
        assert (nav > 0).all()

    def test_bare_diag_columns(self, sample_data):
        prices, weekly, _ = sample_data
        _, diag = dual_momentum_bare(prices, weekly, test_start=prices.index[200])
        expected_cols = {"date", "turnover", "cost", "port_ret", "alert_level"}
        assert expected_cols.issubset(set(diag.columns))

    def test_bare_turnover_non_negative(self, sample_data):
        prices, weekly, _ = sample_data
        _, diag = dual_momentum_bare(prices, weekly, test_start=prices.index[200])
        assert (diag["turnover"] >= 0).all()

    def test_bare_cost_non_negative(self, sample_data):
        prices, weekly, _ = sample_data
        _, diag = dual_momentum_bare(prices, weekly, test_start=prices.index[200])
        assert (diag["cost"] >= 0).all()

    def test_bare_alert_level_all_green(self, sample_data):
        prices, weekly, _ = sample_data
        _, diag = dual_momentum_bare(prices, weekly, test_start=prices.index[200])
        assert (diag["alert_level"] == "green").all()


class TestDualMomentumWithCaGCP:
    def test_cagcp_nav_shape(self, sample_data, fitted_pipe):
        prices, weekly, returns = sample_data
        nav, diag = dual_momentum_with_ca_gcp(
            prices, weekly, fitted_pipe, returns,
            test_start=prices.index[300],
        )
        assert len(nav) == 200
        assert len(diag) == 199  # first day has no trade

    def test_cagcp_nav_starts_at_1(self, sample_data, fitted_pipe):
        prices, weekly, returns = sample_data
        nav, _ = dual_momentum_with_ca_gcp(
            prices, weekly, fitted_pipe, returns,
            test_start=prices.index[300],
        )
        assert nav.iloc[0] == pytest.approx(1.0)

    def test_cagcp_diag_columns(self, sample_data, fitted_pipe):
        prices, weekly, returns = sample_data
        _, diag = dual_momentum_with_ca_gcp(
            prices, weekly, fitted_pipe, returns,
            test_start=prices.index[300],
        )
        expected_cols = {"date", "alert_level", "turnover", "cost", "port_ret"}
        assert expected_cols.issubset(set(diag.columns))

    def test_cagcp_alert_levels_valid(self, sample_data, fitted_pipe):
        prices, weekly, returns = sample_data
        _, diag = dual_momentum_with_ca_gcp(
            prices, weekly, fitted_pipe, returns,
            test_start=prices.index[300],
        )
        valid_levels = {"green", "yellow", "red"}
        assert set(diag["alert_level"].unique()).issubset(valid_levels)

    def test_cagcp_transaction_costs_non_negative(self, sample_data, fitted_pipe):
        prices, weekly, returns = sample_data
        _, diag = dual_momentum_with_ca_gcp(
            prices, weekly, fitted_pipe, returns,
            test_start=prices.index[300],
        )
        assert (diag["cost"] >= 0).all()

    def test_cagcp_with_aggressive_rules(self, sample_data, fitted_pipe):
        prices, weekly, returns = sample_data
        rules = RiskFilterRules(
            width_z_yellow=1.0,
            width_z_red=2.0,
            stress_yellow=0.5,
            stress_red=0.7,
        )
        _, diag = dual_momentum_with_ca_gcp(
            prices, weekly, fitted_pipe, returns,
            rules=rules,
            test_start=prices.index[300],
        )
        # More aggressive rules should trigger more alerts
        n_alerts = (diag["alert_level"] != "green").sum()
        assert n_alerts > 0


class TestRiskFilterHysteresis:
    def test_recovery_requires_lower_stress(self):
        from integration.ca_gcp_risk_filter import RiskFilterRules
        rules = RiskFilterRules()
        assert rules.stress_yellow_recovery < rules.stress_yellow
        assert rules.width_z_yellow_recovery < rules.width_z_yellow
