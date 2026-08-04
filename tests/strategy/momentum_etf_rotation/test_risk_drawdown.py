"""Tests for drawdown controller — Grossman-Zhou continuous drawdown control."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.risk.drawdown import (
    DrawdownConfig, DrawdownState,
    drawdown_multiplier, drawdown_multiplier_curve,
    apply_drawdown_control,
    turtle_drawdown_beta, turtle_drawdown_multiplier,
    drawdown_calibration_table,
)


class TestDrawdownConfig:
    def test_defaults(self) -> None:
        cfg = DrawdownConfig()
        assert cfg.max_tolerance == 0.25
        assert cfg.min_multiplier == 0.0
        assert cfg.enabled is True

    def test_custom_values(self) -> None:
        cfg = DrawdownConfig(max_tolerance=0.15, min_multiplier=0.1, enabled=False)
        assert cfg.max_tolerance == 0.15
        assert cfg.min_multiplier == 0.1
        assert cfg.enabled is False


class TestDrawdownMultiplier:
    def test_at_peak_returns_1(self) -> None:
        assert drawdown_multiplier(1.0, 1.0) == 1.0

    def test_at_max_tolerance_returns_0(self) -> None:
        m = drawdown_multiplier(0.75, 1.0, max_tolerance=0.25)
        assert m == 0.0

    def test_halfway_returns_05(self) -> None:
        m = drawdown_multiplier(0.875, 1.0, max_tolerance=0.25)
        assert m == 0.5

    def test_below_tolerance_returns_0(self) -> None:
        m = drawdown_multiplier(0.5, 1.0, max_tolerance=0.25)
        assert m == 0.0

    def test_negative_equity_returns_1(self) -> None:
        assert drawdown_multiplier(-1.0, 1.0) == 1.0

    def test_zero_peak_returns_1(self) -> None:
        assert drawdown_multiplier(1.0, 0.0) == 1.0

    def test_zero_tolerance_returns_0(self) -> None:
        assert drawdown_multiplier(0.9, 1.0, max_tolerance=0.0) == 0.0

    def test_small_dd_returns_near_1(self) -> None:
        m = drawdown_multiplier(0.99, 1.0, max_tolerance=0.25)
        assert m > 0.95


class TestDrawdownMultiplierCurve:
    def test_returns_array(self) -> None:
        eq = np.linspace(0.5, 1.0, 10)
        curve = drawdown_multiplier_curve(eq, 1.0)
        assert len(curve) == 10
        assert (curve >= 0).all()
        assert (curve <= 1).all()

    def test_monotonic(self) -> None:
        eq = np.linspace(0.5, 1.0, 20)
        curve = drawdown_multiplier_curve(eq, 1.0)
        assert (np.diff(curve) >= 0).all()


class TestDrawdownState:
    def test_update_peak(self) -> None:
        state = DrawdownState()
        state.update(1.1)
        assert state.peak == 1.1

    def test_update_dd(self) -> None:
        state = DrawdownState()
        state.update(0.9)
        assert state.current_dd == pytest.approx(0.1)

    def test_update_multiplier(self) -> None:
        state = DrawdownState()
        m = state.update(0.75, max_tolerance=0.25)
        assert m == 0.0
        assert state.current_multiplier == 0.0

    def test_new_peak_resets_dd(self) -> None:
        state = DrawdownState()
        state.update(0.9)
        state.update(1.2)
        assert state.current_dd == 0.0
        assert state.current_multiplier == 1.0


class TestApplyDrawdownControl:
    def test_returns_same_keys(self) -> None:
        weights = {"A": 0.6, "B": 0.4}
        result = apply_drawdown_control(weights, 0.95, 1.0)
        assert set(result.keys()) == {"A", "B"}

    def test_scales_proportionally(self) -> None:
        weights = {"A": 0.6, "B": 0.4}
        result = apply_drawdown_control(weights, 0.875, 1.0, DrawdownConfig(max_tolerance=0.25))
        assert result["A"] == 0.3
        assert result["B"] == 0.2

    def test_disabled_returns_original(self) -> None:
        weights = {"A": 0.6, "B": 0.4}
        result = apply_drawdown_control(weights, 0.5, 1.0, DrawdownConfig(enabled=False))
        assert result == weights

    def test_min_multiplier_clamp(self) -> None:
        weights = {"A": 0.6}
        result = apply_drawdown_control(weights, 0.5, 1.0, DrawdownConfig(max_tolerance=0.25, min_multiplier=0.2))
        assert result["A"] == 0.12

    def test_default_config(self) -> None:
        weights = {"A": 1.0}
        result = apply_drawdown_control(weights, 0.8, 1.0)
        assert result["A"] == pytest.approx(0.2)


class TestTurtle:
    def test_beta_positive(self) -> None:
        beta = turtle_drawdown_beta(0.25)
        assert beta > 0

    def test_beta_decreases_with_tolerance(self) -> None:
        b1 = turtle_drawdown_beta(0.40)
        b2 = turtle_drawdown_beta(0.15)
        assert b1 > b2, f"beta(0.40)={b1} should be > beta(0.15)={b2}"

    def test_multiplier_at_peak(self) -> None:
        assert turtle_drawdown_multiplier(1.0, 1.0) == 1.0

    def test_multiplier_after_one_step(self) -> None:
        m = turtle_drawdown_multiplier(0.85, 1.0, max_tolerance=0.25)
        # One 10% step: multiplier = 1 - beta = 1 - (1/(1-0.25)-1) = 1 - 0.333 = 0.667
        assert m < 1.0

    def test_multiplier_below_tolerance(self) -> None:
        m = turtle_drawdown_multiplier(0.5, 1.0, max_tolerance=0.25)
        # 50% DD = 5 steps × 10%, multiplier = (1-beta)^5
        beta = turtle_drawdown_beta(0.25)
        expected = (1.0 - beta) ** 5
        assert m == pytest.approx(expected)


class TestCalibrationTable:
    def test_returns_dataframe(self) -> None:
        df = drawdown_calibration_table()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_has_expected_columns(self) -> None:
        df = drawdown_calibration_table()
        for col in ("max_tolerance", "max_dd_pct", "turtle_beta_per_10pct", "floor"):
            assert col in df.columns, f"missing {col}"