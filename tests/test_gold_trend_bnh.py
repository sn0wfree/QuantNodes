"""tests/test_gold_trend_bnh.py — 趋势B&H策略单元测试."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from QuantNodes.strategy.commodity_fundamental.gold.gold_trend_bnh import (
    load_gold_price_d,
    load_gold_price_d_with_open,
    compute_trend_bnh_signal,
    backtest_trend_bnh,
    compute_hysteresis_trend_d,
    compute_risk_overlay,
    compute_macro_regime_signal,
)


class TestLoadGoldPrice(unittest.TestCase):
    """测试金价加载函数."""

    def test_load_gold_price_d(self):
        close = load_gold_price_d()
        self.assertIsInstance(close, pd.Series)
        self.assertGreater(len(close), 1000)
        self.assertTrue(close.index.is_monotonic_increasing)

    def test_load_gold_price_d_with_open(self):
        df = load_gold_price_d_with_open()
        self.assertIn("open", df.columns)
        self.assertIn("close", df.columns)
        self.assertGreater(len(df), 1000)


class TestTrendBnhSignal(unittest.TestCase):
    """测试趋势B&H信号."""

    def test_output_columns(self):
        signal = compute_trend_bnh_signal()
        self.assertIn("close", signal.columns)
        self.assertIn("position", signal.columns)

    def test_position_range(self):
        signal = compute_trend_bnh_signal()
        self.assertTrue((signal["position"] >= 0).all())
        self.assertTrue((signal["position"] <= 1.2).all())

    def test_weekly_freq(self):
        signal = compute_trend_bnh_signal(freq="weekly")
        self.assertGreater(len(signal), 100)

    def test_daily_freq(self):
        signal = compute_trend_bnh_signal(freq="daily")
        self.assertGreater(len(signal), 500)

    def test_trailing_stop_effect(self):
        """止损越紧, 空仓越多."""
        signal_loose = compute_trend_bnh_signal(trailing_stop=0.20)
        signal_tight = compute_trend_bnh_signal(trailing_stop=0.05)
        pos_loose = signal_loose["position"].mean()
        pos_tight = signal_tight["position"].mean()
        self.assertGreater(pos_loose, pos_tight)

    def test_backtest_output(self):
        signal = compute_trend_bnh_signal()
        result = backtest_trend_bnh(signal)
        self.assertIn("sharpe", result)
        self.assertIn("max_dd", result)
        self.assertIn("turnover", result)

    def test_backtest_positive_sharpe(self):
        """默认参数应有正Sharpe."""
        signal = compute_trend_bnh_signal()
        result = backtest_trend_bnh(signal)
        self.assertGreater(result["sharpe"], 0)


class TestHysteresisTrend(unittest.TestCase):
    """测试滞回带趋势信号."""

    def test_output_type(self):
        pos = compute_hysteresis_trend_d()
        self.assertIsInstance(pos, pd.Series)

    def test_position_range(self):
        pos = compute_hysteresis_trend_d()
        self.assertTrue((pos >= 0).all())
        self.assertTrue((pos <= 1).all())

    def test_data_length(self):
        pos = compute_hysteresis_trend_d()
        self.assertGreater(len(pos), 1000)


class TestRiskOverlay(unittest.TestCase):
    """测试风控覆盖层."""

    def test_output_type(self):
        close = load_gold_price_d()
        position = pd.Series(1.0, index=close.index)
        result = compute_risk_overlay(close=close, position=position)
        self.assertIsInstance(result, pd.Series)

    def test_output_length(self):
        close = load_gold_price_d()
        position = pd.Series(1.0, index=close.index)
        result = compute_risk_overlay(close=close, position=position)
        self.assertEqual(len(result), len(close))

    def test_position_range(self):
        close = load_gold_price_d()
        position = pd.Series(1.0, index=close.index)
        result = compute_risk_overlay(close=close, position=position)
        self.assertTrue((result >= 0).all())
        self.assertTrue((result <= 1.2).all())


class TestMacroRegimeSignal(unittest.TestCase):
    """测试宏观机制信号."""

    def test_output_columns(self):
        signal = compute_macro_regime_signal()
        self.assertIn("close", signal.columns)
        self.assertIn("position", signal.columns)

    def test_position_range(self):
        signal = compute_macro_regime_signal()
        self.assertTrue((signal["position"] >= 0).all())
        self.assertTrue((signal["position"] <= 1.2).all())

    def test_data_length(self):
        signal = compute_macro_regime_signal()
        self.assertGreater(len(signal), 1000)


if __name__ == "__main__":
    unittest.main()
