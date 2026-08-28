"""tests/test_gold_ensemble.py — 集成策略单元测试."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from QuantNodes.strategy.commodity_fundamental.gold.gold_ensemble import (
    compute_adaptive_ensemble,
    compute_layered_ensemble_signal,
)


class TestAdaptiveEnsemble(unittest.TestCase):
    """测试自适应集成."""

    def test_output_columns(self):
        signal = compute_adaptive_ensemble()
        self.assertIn("close", signal.columns)
        self.assertIn("position", signal.columns)

    def test_position_range(self):
        signal = compute_adaptive_ensemble()
        self.assertTrue((signal["position"] >= 0).all())
        self.assertTrue((signal["position"] <= 1.2).all())

    def test_fixed_weight(self):
        signal = compute_adaptive_ensemble(weight_method="fixed")
        self.assertGreater(len(signal), 0)

    def test_adaptive_weight(self):
        signal = compute_adaptive_ensemble(weight_method="adaptive")
        self.assertGreater(len(signal), 0)

    def test_invalid_weight_method(self):
        with self.assertRaises(ValueError):
            compute_adaptive_ensemble(weight_method="invalid")

    def test_without_risk_overlay(self):
        signal = compute_adaptive_ensemble(use_risk_overlay=False)
        self.assertGreater(len(signal), 0)


class TestLayeredEnsemble(unittest.TestCase):
    """测试分层集成."""

    def test_output_columns(self):
        signal = compute_layered_ensemble_signal()
        self.assertIn("close", signal.columns)
        self.assertIn("open", signal.columns)
        self.assertIn("position", signal.columns)

    def test_position_range(self):
        signal = compute_layered_ensemble_signal()
        self.assertTrue((signal["position"] >= 0).all())
        self.assertTrue((signal["position"] <= 1.2).all())

    def test_data_length(self):
        signal = compute_layered_ensemble_signal()
        self.assertGreater(len(signal), 1000)

    def test_without_risk_overlay(self):
        signal = compute_layered_ensemble_signal(use_risk_overlay=False)
        self.assertGreater(len(signal), 0)

    def test_with_breakout(self):
        signal = compute_layered_ensemble_signal(breakout_weight=0.2)
        self.assertGreater(len(signal), 0)


if __name__ == "__main__":
    unittest.main()
