"""tests/test_gold_signals.py — 6个单因子信号单元测试."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from QuantNodes.strategy.commodity_fundamental.gold.gold_real_rate import (
    compute_real_rate_signal, backtest_real_rate,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_usd import (
    compute_usd_signal, backtest_usd,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_risk import (
    compute_risk_signal, backtest_risk,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_momentum import (
    compute_momentum_signal, backtest_momentum,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_position_flow import (
    compute_position_flow_signal, backtest_position_flow,
)
from QuantNodes.strategy.commodity_fundamental.gold.gold_breakout import (
    compute_breakout_signal, backtest_breakout,
)


class TestRealRateSignal(unittest.TestCase):
    """测试实际利率信号."""

    def test_output_columns(self):
        signal = compute_real_rate_signal()
        for col in ["close", "tips", "rir", "tips_ma", "signal", "position"]:
            self.assertIn(col, signal.columns)

    def test_position_range(self):
        signal = compute_real_rate_signal()
        self.assertTrue((signal["position"] >= 0).all())
        self.assertTrue((signal["position"] <= 1).all())

    def test_variant_tips_ma(self):
        signal = compute_real_rate_signal(variant="tips_ma")
        self.assertGreater(len(signal), 0)

    def test_variant_tips_level(self):
        signal = compute_real_rate_signal(variant="tips_level")
        self.assertGreater(len(signal), 0)

    def test_variant_rir_negative(self):
        signal = compute_real_rate_signal(variant="rir_negative")
        self.assertGreater(len(signal), 0)

    def test_variant_combined(self):
        signal = compute_real_rate_signal(variant="combined")
        self.assertGreater(len(signal), 0)

    def test_invalid_variant(self):
        with self.assertRaises(ValueError):
            compute_real_rate_signal(variant="invalid")

    def test_backtest_output(self):
        signal = compute_real_rate_signal()
        result = backtest_real_rate(signal)
        self.assertIn("sharpe", result)
        self.assertIn("max_dd", result)


class TestUSDSignal(unittest.TestCase):
    """测试美元趋势信号."""

    def test_output_columns(self):
        signal = compute_usd_signal()
        for col in ["close", "dxy", "signal", "position"]:
            self.assertIn(col, signal.columns)

    def test_position_range(self):
        signal = compute_usd_signal()
        self.assertTrue((signal["position"] >= 0).all())
        self.assertTrue((signal["position"] <= 1).all())

    def test_variant_dxy_ema(self):
        signal = compute_usd_signal(variant="dxy_ema")
        self.assertGreater(len(signal), 0)

    def test_variant_dxy_momentum(self):
        signal = compute_usd_signal(variant="dxy_momentum")
        self.assertGreater(len(signal), 0)

    def test_variant_dxy_zscore(self):
        signal = compute_usd_signal(variant="dxy_zscore")
        self.assertGreater(len(signal), 0)

    def test_invalid_variant(self):
        with self.assertRaises(ValueError):
            compute_usd_signal(variant="invalid")

    def test_backtest_output(self):
        signal = compute_usd_signal()
        result = backtest_usd(signal)
        self.assertIn("sharpe", result)


class TestRiskSignal(unittest.TestCase):
    """测试避险情绪信号."""

    def test_output_columns(self):
        signal = compute_risk_signal()
        for col in ["close", "vix", "gprd", "signal", "position"]:
            self.assertIn(col, signal.columns)

    def test_position_range(self):
        signal = compute_risk_signal()
        self.assertTrue((signal["position"] >= 0).all())
        self.assertTrue((signal["position"] <= 1).all())

    def test_variant_vix_pct(self):
        signal = compute_risk_signal(variant="vix_pct")
        self.assertGreater(len(signal), 0)

    def test_variant_gprd_pct(self):
        signal = compute_risk_signal(variant="gprd_pct")
        self.assertGreater(len(signal), 0)

    def test_variant_combined(self):
        signal = compute_risk_signal(variant="combined")
        self.assertGreater(len(signal), 0)

    def test_variant_weighted(self):
        signal = compute_risk_signal(variant="weighted")
        self.assertGreater(len(signal), 0)

    def test_invalid_variant(self):
        with self.assertRaises(ValueError):
            compute_risk_signal(variant="invalid")

    def test_backtest_output(self):
        signal = compute_risk_signal()
        result = backtest_risk(signal)
        self.assertIn("sharpe", result)


class TestMomentumSignal(unittest.TestCase):
    """测试动量信号."""

    def test_output_columns(self):
        signal = compute_momentum_signal()
        for col in ["close", "mom_long", "mom_short", "signal", "position"]:
            self.assertIn(col, signal.columns)

    def test_position_range(self):
        signal = compute_momentum_signal()
        self.assertTrue((signal["position"] >= 0).all())
        self.assertTrue((signal["position"] <= 1).all())

    def test_variant_long_only(self):
        signal = compute_momentum_signal(variant="long_only")
        self.assertGreater(len(signal), 0)

    def test_variant_dual_window(self):
        signal = compute_momentum_signal(variant="dual_window")
        self.assertGreater(len(signal), 0)

    def test_invalid_variant(self):
        with self.assertRaises(ValueError):
            compute_momentum_signal(variant="invalid")

    def test_backtest_output(self):
        signal = compute_momentum_signal()
        result = backtest_momentum(signal)
        self.assertIn("sharpe", result)


class TestPositionFlowSignal(unittest.TestCase):
    """测试仓位流信号."""

    def test_output_columns(self):
        signal = compute_position_flow_signal()
        for col in ["close", "cb_reserves", "etf_tonnes", "signal", "position"]:
            self.assertIn(col, signal.columns)

    def test_position_range(self):
        signal = compute_position_flow_signal()
        pos = signal["position"].dropna()
        self.assertTrue((pos >= 0).all())
        self.assertTrue((pos <= 1).all())

    def test_variant_cb_only(self):
        signal = compute_position_flow_signal(variant="cb_only")
        self.assertGreater(len(signal), 0)

    def test_variant_etf_only(self):
        signal = compute_position_flow_signal(variant="etf_only")
        self.assertGreater(len(signal), 0)

    def test_variant_combined(self):
        signal = compute_position_flow_signal(variant="combined")
        self.assertGreater(len(signal), 0)

    def test_invalid_variant(self):
        with self.assertRaises(ValueError):
            compute_position_flow_signal(variant="invalid")

    def test_backtest_output(self):
        signal = compute_position_flow_signal()
        result = backtest_position_flow(signal)
        self.assertIn("sharpe", result)


class TestBreakoutSignal(unittest.TestCase):
    """测试突破信号."""

    def test_output_columns(self):
        signal = compute_breakout_signal()
        for col in ["close", "position"]:
            self.assertIn(col, signal.columns)

    def test_position_range(self):
        signal = compute_breakout_signal()
        self.assertTrue((signal["position"] >= 0).all())
        self.assertTrue((signal["position"] <= 1).all())

    def test_data_length(self):
        signal = compute_breakout_signal()
        self.assertGreater(len(signal), 100)

    def test_backtest_output(self):
        signal = compute_breakout_signal()
        result = backtest_breakout(signal)
        self.assertIn("sharpe", result)


if __name__ == "__main__":
    unittest.main()
