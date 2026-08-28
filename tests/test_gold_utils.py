"""tests/test_gold_utils.py — gold_utils.py 公共函数单元测试."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from QuantNodes.strategy.commodity_fundamental.gold.gold_utils import (
    DB_PATH,
    load_gold_price_d,
    load_gold_price_w,
    load_indicator_d,
    load_indicator_w,
    backtest_weekly,
    get_weekly_returns,
)


class TestDBPath(unittest.TestCase):
    """测试DB_PATH常量."""

    def test_is_path(self):
        self.assertIsInstance(DB_PATH, Path)

    def test_ends_with_duckdb(self):
        self.assertTrue(str(DB_PATH).endswith(".duckdb"))


class TestBacktestWeekly(unittest.TestCase):
    """测试通用周频回测函数."""

    def _make_signal(self, n=52, ret_seq=None):
        """构造测试信号."""
        dates = pd.date_range("2020-01-01", periods=n, freq="W-FRI")
        if ret_seq is None:
            close = pd.Series(np.cumprod(1 + np.random.randn(n) * 0.02), index=dates)
        else:
            close = pd.Series(np.cumprod(1 + np.array(ret_seq)), index=dates)
        position = pd.Series(1.0, index=dates)
        return pd.DataFrame({"close": close, "position": position})

    def test_basic_output_keys(self):
        """输出应包含所有必要key."""
        signal = self._make_signal()
        result = backtest_weekly(signal)
        required_keys = [
            "n_weeks", "ann_ret", "ann_vol", "sharpe", "max_dd",
            "win_rate", "win_rate_active", "avg_pos", "pct_active", "nav",
        ]
        for key in required_keys:
            self.assertIn(key, result, f"缺少key: {key}")

    def test_empty_signal(self):
        """空信号应返回零值."""
        signal = pd.DataFrame({
            "close": pd.Series(dtype=float),
            "position": pd.Series(dtype=float),
        })
        result = backtest_weekly(signal)
        self.assertEqual(result["n_weeks"], 0)
        self.assertEqual(result["ann_ret"], 0)

    def test_positive_return(self):
        """持续上涨应有正收益."""
        signal = self._make_signal(ret_seq=[0.01] * 52)
        result = backtest_weekly(signal)
        self.assertGreater(result["ann_ret"], 0)

    def test_negative_return(self):
        """持续下跌应有负收益."""
        signal = self._make_signal(ret_seq=[-0.01] * 52)
        result = backtest_weekly(signal)
        self.assertLess(result["ann_ret"], 0)

    def test_zero_position(self):
        """空仓应有零收益."""
        signal = self._make_signal()
        signal["position"] = 0.0
        result = backtest_weekly(signal)
        self.assertAlmostEqual(result["ann_ret"], 0, places=5)

    def test_win_rate_range(self):
        """胜率应在[0,1]范围内."""
        signal = self._make_signal()
        result = backtest_weekly(signal)
        self.assertGreaterEqual(result["win_rate"], 0)
        self.assertLessEqual(result["win_rate"], 1)

    def test_nav_series(self):
        """nav应为Series且长度等于n_weeks."""
        signal = self._make_signal(52)
        result = backtest_weekly(signal)
        self.assertIsInstance(result["nav"], pd.Series)
        self.assertEqual(len(result["nav"]), result["n_weeks"])


class TestGetWeeklyReturns(unittest.TestCase):
    """测试周频收益获取函数."""

    def _make_signal(self):
        dates = pd.date_range("2018-01-01", periods=200, freq="W-FRI")
        close = pd.Series(np.cumprod(1 + np.random.randn(200) * 0.02), index=dates)
        position = pd.Series(1.0, index=dates)
        return pd.DataFrame({"close": close, "position": position})

    def test_date_filter(self):
        """收益应只包含指定日期范围."""
        signal = self._make_signal()
        result = get_weekly_returns(signal, start="2019-01-01", end="2020-12-31")
        self.assertTrue(all(result.index >= "2019-01-01"))
        self.assertTrue(all(result.index <= "2020-12-31"))

    def test_output_type(self):
        """输出应为Series."""
        signal = self._make_signal()
        result = get_weekly_returns(signal)
        self.assertIsInstance(result, pd.Series)


if __name__ == "__main__":
    unittest.main()
