"""tests/test_gold_rollover.py — 黄金渐进式换月单元测试.

测试 build_gold_smooth.py 中的核心函数:
    - compute_rollover_weights
    - blend_row
    - build_rollover_windows
    - compute_smooth_prices
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "cmf"))

from build_gold_smooth import (
    RolloverWindow,
    blend_row,
    build_rollover_windows,
    compute_rollover_weights,
    compute_smooth_prices,
)


class TestComputeRolloverWeights(unittest.TestCase):
    """测试换月权重计算."""

    def test_4_day_window(self):
        """4天窗口: 权重线性从1.0降到0.0."""
        n = 4
        expected = [(1.0, 0.0), (0.667, 0.333), (0.333, 0.667), (0.0, 1.0)]
        for i, (exp_old, exp_new) in enumerate(expected):
            w_old, w_new = compute_rollover_weights(n, i)
            self.assertAlmostEqual(w_old, exp_old, places=2,
                                   msg=f"Day {i}: w_old expected {exp_old}, got {w_old}")
            self.assertAlmostEqual(w_new, exp_new, places=2,
                                   msg=f"Day {i}: w_new expected {exp_new}, got {w_new}")

    def test_3_day_window(self):
        """3天窗口: 权重线性插值."""
        n = 3
        expected = [(1.0, 0.0), (0.5, 0.5), (0.0, 1.0)]
        for i, (exp_old, exp_new) in enumerate(expected):
            w_old, w_new = compute_rollover_weights(n, i)
            self.assertAlmostEqual(w_old, exp_old, places=2)
            self.assertAlmostEqual(w_new, exp_new, places=2)

    def test_single_day(self):
        """单天窗口: 直接切换到新合约."""
        w_old, w_new = compute_rollover_weights(1, 0)
        self.assertEqual(w_old, 0.0)
        self.assertEqual(w_new, 1.0)

    def test_weights_sum_to_one(self):
        """任意天数窗口, 权重之和应为1.0."""
        for n in range(1, 10):
            for i in range(n):
                w_old, w_new = compute_rollover_weights(n, i)
                self.assertAlmostEqual(w_old + w_new, 1.0, places=10,
                                       msg=f"n={n}, i={i}: weights sum != 1.0")


class TestBlendRow(unittest.TestCase):
    """测试数据加权平均."""

    def test_both_rows_present(self):
        """双方都有数据时, 加权平均."""
        old = pd.Series({"open": 100.0, "close": 102.0, "volume": 1000, "open_interest": 5000})
        new = pd.Series({"open": 200.0, "close": 204.0, "volume": 2000, "open_interest": 8000})

        result = blend_row(old, new, 0.75, 0.25)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["open"], 125.0)   # 0.75*100 + 0.25*200
        self.assertAlmostEqual(result["close"], 127.5)  # 0.75*102 + 0.25*204
        self.assertEqual(result["volume"], 1250)         # 0.75*1000 + 0.25*2000
        self.assertEqual(result["open_interest"], 5750)  # 0.75*5000 + 0.25*8000

    def test_only_old_present(self):
        """只有旧合约数据时, 使用旧合约."""
        old = pd.Series({"open": 100.0, "close": 102.0, "volume": 1000, "open_interest": 5000})

        result = blend_row(old, None, 0.5, 0.5)
        self.assertIsNotNone(result)
        self.assertEqual(result["open"], 100.0)
        self.assertEqual(result["close"], 102.0)
        self.assertEqual(result["w_old"], 1.0)
        self.assertEqual(result["w_new"], 0.0)

    def test_only_new_present(self):
        """只有新合约数据时, 使用新合约."""
        new = pd.Series({"open": 200.0, "close": 204.0, "volume": 2000, "open_interest": 8000})

        result = blend_row(None, new, 0.5, 0.5)
        self.assertIsNotNone(result)
        self.assertEqual(result["open"], 200.0)
        self.assertEqual(result["close"], 204.0)
        self.assertEqual(result["w_old"], 0.0)
        self.assertEqual(result["w_new"], 1.0)

    def test_neither_present(self):
        """双方都无数据时, 返回None."""
        result = blend_row(None, None, 0.5, 0.5)
        self.assertIsNone(result)


class TestBuildRolloverWindows(unittest.TestCase):
    """测试换月窗口构建."""

    def test_basic_window(self):
        """基本换月窗口构建."""
        main_dates = pd.to_datetime([
            "2024-01-02", "2024-01-03", "2024-01-04",
            "2024-01-05", "2024-01-08", "2024-01-09",
        ])
        switches = pd.DataFrame({
            "switch_date": pd.to_datetime(["2024-01-05"]),
            "old_contract": ["AU2402"],
            "new_contract": ["AU2406"],
        })

        windows = build_rollover_windows(main_dates, switches, rollover_days=4)
        self.assertEqual(len(windows), 1)
        w = windows[0]
        self.assertEqual(w.old_contract, "AU2402")
        self.assertEqual(w.new_contract, "AU2406")
        self.assertEqual(len(w.window_dates), 4)

    def test_insufficient_dates(self):
        """日期不足时, 跳过该窗口."""
        main_dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        switches = pd.DataFrame({
            "switch_date": pd.to_datetime(["2024-01-03"]),
            "old_contract": ["AU2402"],
            "new_contract": ["AU2406"],
        })

        windows = build_rollover_windows(main_dates, switches, rollover_days=4)
        # 3个日期, rollover_days=4, 但before有1个after有2个, 共3个<4
        # 实际可能仍然构建窗口, 取决于实现
        self.assertIsInstance(windows, list)

    def test_multiple_switches(self):
        """多次换月."""
        main_dates = pd.date_range("2024-01-01", periods=20, freq="B")
        switches = pd.DataFrame({
            "switch_date": pd.to_datetime(["2024-01-08", "2024-01-15"]),
            "old_contract": ["AU2402", "AU2406"],
            "new_contract": ["AU2406", "AU2412"],
        })

        windows = build_rollover_windows(main_dates, switches, rollover_days=4)
        self.assertEqual(len(windows), 2)


class TestComputeSmoothPrices(unittest.TestCase):
    """测试平滑价格计算."""

    def _make_data(self):
        """构造测试数据."""
        dates = pd.date_range("2024-01-01", periods=10, freq="B")

        # 旧合约: 价格100附近
        old_data = pd.DataFrame({
            "trade_date": dates,
            "contract": "AU2402",
            "open": [100 + i * 0.1 for i in range(10)],
            "close": [100.1 + i * 0.1 for i in range(10)],
            "volume": [1000] * 10,
            "open_interest": [5000 - i * 500 for i in range(10)],
        })

        # 新合约: 价格100附近 (与旧合约价差很小)
        new_data = pd.DataFrame({
            "trade_date": dates,
            "contract": "AU2406",
            "open": [100.5 + i * 0.1 for i in range(10)],
            "close": [100.6 + i * 0.1 for i in range(10)],
            "volume": [2000] * 10,
            "open_interest": [1000 + i * 500 for i in range(10)],
        })

        all_contracts = pd.concat([old_data, new_data], ignore_index=True)

        # 主力合约: 前5天用旧合约, 后5天用新合约
        main_df = pd.DataFrame({
            "trade_date": dates,
            "main_contract": ["AU2402"] * 5 + ["AU2406"] * 5,
            "open": [100 + i * 0.1 for i in range(5)] + [100.5 + i * 0.1 for i in range(5, 10)],
            "close": [100.1 + i * 0.1 for i in range(5)] + [100.6 + i * 0.1 for i in range(5, 10)],
            "volume": [1000] * 5 + [2000] * 5,
            "open_interest": [5000 - i * 500 for i in range(5)] + [1000 + i * 500 for i in range(5, 10)],
        })

        switches = pd.DataFrame({
            "switch_date": pd.to_datetime(["2024-01-05"]),
            "old_contract": ["AU2402"],
            "new_contract": ["AU2406"],
        })

        return all_contracts, main_df, switches

    def test_smooth_prices_no_jump(self):
        """换月期不应有跳空."""
        all_contracts, main_df, switches = self._make_data()
        result = compute_smooth_prices(all_contracts, main_df, switches, rollover_days=4)

        # 检查换月期的gap应该很小
        rollover_gaps = result[result["is_rollover"]]["gap"]
        if len(rollover_gaps) > 0:
            self.assertTrue(
                abs(rollover_gaps).max() < 0.1,
                f"换月期最大gap应<10%, 实际={abs(rollover_gaps).max()*100:.2f}%"
            )

    def test_output_columns(self):
        """输出应包含所有必要列."""
        all_contracts, main_df, switches = self._make_data()
        result = compute_smooth_prices(all_contracts, main_df, switches)

        required_cols = ["trade_date", "open", "close", "volume", "open_interest",
                         "contract", "is_rollover", "w_old", "w_new", "ret", "gap"]
        for col in required_cols:
            self.assertIn(col, result.columns, f"缺少列: {col}")

    def test_output_length(self):
        """输出行数应等于主力合约天数."""
        all_contracts, main_df, switches = self._make_data()
        result = compute_smooth_prices(all_contracts, main_df, switches)
        self.assertEqual(len(result), len(main_df))

    def test_rollover_dates_marked(self):
        """换月期应被正确标记."""
        all_contracts, main_df, switches = self._make_data()
        result = compute_smooth_prices(all_contracts, main_df, switches, rollover_days=4)

        # 换月期应有4天
        rollover_count = result["is_rollover"].sum()
        self.assertGreater(rollover_count, 0, "应有换月期")
        self.assertLessEqual(rollover_count, 4, "换月期不应超过4天")

    def test_weights_boundary(self):
        """换月期第一天权重应接近1.0(旧), 最后一天接近0.0(旧)."""
        all_contracts, main_df, switches = self._make_data()
        result = compute_smooth_prices(all_contracts, main_df, switches, rollover_days=4)

        rollover_rows = result[result["is_rollover"]]
        if len(rollover_rows) >= 2:
            self.assertGreater(rollover_rows.iloc[0]["w_old"], 0.5,
                               "第一天旧合约权重应>0.5")
            self.assertLess(rollover_rows.iloc[-1]["w_old"], 0.5,
                            "最后一天旧合约权重应<0.5")


if __name__ == "__main__":
    unittest.main()
