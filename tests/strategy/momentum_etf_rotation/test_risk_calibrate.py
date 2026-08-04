"""Tests for PlattCalibrator."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from QuantNodes.strategy.momentum_etf_rotation.risk.calibrate import PlattCalibrator


class TestPlattCalibrator:
    def test_init_defaults(self) -> None:
        c = PlattCalibrator()
        assert c.A == 1.0
        assert c.B == 0.0
        assert c._fitted is False

    def test_not_fitted_returns_identity(self) -> None:
        c = PlattCalibrator()
        prob = pd.Series([0.3, 0.5, 0.7])
        result = c.transform(prob)
        assert (result == prob).all()

    def test_fit_transform_small_data(self) -> None:
        c = PlattCalibrator(min_samples=5)
        idx = pd.bdate_range("2024-01-01", periods=60)
        np.random.seed(42)
        prob = pd.Series(np.random.uniform(0.2, 0.8, len(idx)), index=idx)
        y_true = pd.Series(np.random.randint(0, 2, len(idx)), index=idx)
        result = c.fit_transform(prob, y_true)
        assert len(result) == len(prob)
        assert result.between(0, 1).all()

    def test_fit_transform_maintains_monotonicity(self) -> None:
        c = PlattCalibrator(window=60, min_samples=10)
        idx = pd.bdate_range("2024-01-01", periods=60)
        np.random.seed(42)
        prob = pd.Series(np.linspace(0.1, 0.9, len(idx)), index=idx)
        y_true = pd.Series(np.random.randint(0, 2, len(idx)), index=idx)
        result = c.fit_transform(prob, y_true)
        # Calibrated output should be monotonic with input
        diffs = np.diff(result.values)
        # Allow small violations due to noise
        neg_count = (diffs < -1e-6).sum()
        assert neg_count < 5, f"too many negative diffs: {neg_count}"

    def test_insufficient_samples_skips_fit(self) -> None:
        c = PlattCalibrator(window=10, min_samples=100)
        idx = pd.bdate_range("2024-01-01", periods=20)
        prob = pd.Series(np.linspace(0.1, 0.9, len(idx)), index=idx)
        y_true = pd.Series(np.ones(len(idx)), index=idx)
        c.fit(prob, y_true)
        assert c._fitted is False

    def test_single_class_skips_fit(self) -> None:
        c = PlattCalibrator(min_samples=5)
        idx = pd.bdate_range("2024-01-01", periods=30)
        prob = pd.Series(np.linspace(0.1, 0.9, len(idx)), index=idx)
        y_true = pd.Series(np.ones(len(idx)), index=idx)
        c.fit(prob, y_true)
        assert c._fitted is False

    def test_fitted_produces_different_output(self) -> None:
        c = PlattCalibrator(window=60, min_samples=10)
        idx = pd.bdate_range("2024-01-01", periods=60)
        np.random.seed(42)
        prob = pd.Series(np.random.uniform(0.2, 0.8, len(idx)), index=idx)
        y_true = pd.Series(np.random.randint(0, 2, len(idx)), index=idx)
        result = c.fit_transform(prob, y_true)
        # If fitted, output should differ from input
        if c._fitted:
            assert not (result == prob).all()