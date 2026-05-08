# -*- coding: utf-8 -*-
"""ConfigStrategyNode 单元测试

覆盖 backtest/config_strategy.py 中的 ConfigStrategyNode。
"""
import unittest
import numpy as np
import pandas as pd

from QuantNodes.backtest.config_strategy import ConfigStrategyNode
from QuantNodes.backtest.strategy_node import OrdersResult


def make_signal_data(n_days=20, codes=None, seed=42):
    """创建带 signal 列的测试数据"""
    np.random.seed(seed)
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    if codes is None:
        codes = ['AAPL']
    records = []
    for code in codes:
        for i, d in enumerate(dates):
            signal = 0
            if i > 5 and i % 7 == 0:
                signal = 1
            elif i > 10 and i % 11 == 0:
                signal = -1
            records.append({
                'date': str(d.date()),
                'Code': code,
                'Close': 100 + i * 0.5 + np.random.randn() * 0.1,
                'signal': signal,
            })
    return pd.DataFrame(records)


class TestConfigStrategyNode(unittest.TestCase):
    """ConfigStrategyNode 测试"""

    def test_creation_default(self):
        node = ConfigStrategyNode()
        self.assertEqual(node.name, 'ConfigStrategy')
        self.assertEqual(node._signal_col, 'signal')

    def test_creation_custom_signal_col(self):
        node = ConfigStrategyNode(signal_col='my_signal')
        self.assertEqual(node._signal_col, 'my_signal')

    def test_empty_dataframe(self):
        node = ConfigStrategyNode()
        df = pd.DataFrame({'date': [], 'Code': [], 'Close': [], 'signal': []})
        result = node.execute(df)
        self.assertIsInstance(result, OrdersResult)
        self.assertEqual(len(result.signals), 0)

    def test_no_signals(self):
        node = ConfigStrategyNode()
        df = make_signal_data(n_days=10)
        df['signal'] = 0
        result = node.execute(df)
        self.assertEqual(len(result.signals), 0)

    def test_buy_signals(self):
        node = ConfigStrategyNode()
        df = make_signal_data(n_days=20)
        result = node.execute(df)
        buy_signals = [s for s in result.signals if s.signal_type == 'buy']
        self.assertGreater(len(buy_signals), 0)
        for sig in buy_signals:
            self.assertEqual(sig.strength, 1.0)

    def test_sell_signals(self):
        node = ConfigStrategyNode()
        df = make_signal_data(n_days=20)
        result = node.execute(df)
        sell_signals = [s for s in result.signals if s.signal_type == 'sell']
        self.assertGreater(len(sell_signals), 0)
        for sig in sell_signals:
            self.assertEqual(sig.strength, 1.0)

    def test_signal_columns_case_insensitive(self):
        node = ConfigStrategyNode()
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'code': ['AAPL', 'AAPL'],
            'close': [100.0, 101.0],
            'signal': [1, -1],
        })
        result = node.execute(df)
        self.assertEqual(len(result.signals), 2)

    def test_multi_stock_signals(self):
        node = ConfigStrategyNode()
        df = make_signal_data(n_days=20, codes=['AAPL', 'GOOG'])
        result = node.execute(df)
        codes = set(s.code for s in result.signals)
        self.assertIn('AAPL', codes)
        self.assertIn('GOOG', codes)

    def test_signal_strength_from_value(self):
        node = ConfigStrategyNode()
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Code': ['AAPL', 'AAPL', 'AAPL'],
            'Close': [100.0, 101.0, 102.0],
            'signal': [2, 0, -3],
        })
        result = node.execute(df)
        signals = sorted(result.signals, key=lambda s: s.date)
        self.assertEqual(signals[0].strength, 2.0)
        self.assertEqual(signals[1].strength, 3.0)

    def test_orders_from_signals(self):
        node = ConfigStrategyNode()
        df = make_signal_data(n_days=20)
        result = node.execute(df)
        self.assertIsInstance(result.orders, list)
        self.assertEqual(len(result.orders), len(result.signals))

    def test_order_size_from_signal(self):
        node = ConfigStrategyNode()
        df = pd.DataFrame({
            'date': ['2024-01-01'],
            'Code': ['AAPL'],
            'Close': [100.0],
            'signal': [1],
        })
        result = node.execute(df)
        self.assertEqual(len(result.orders), 1)
        self.assertEqual(result.orders[0].size, 1.0)

    def test_order_negative_size_for_sell(self):
        node = ConfigStrategyNode()
        df = pd.DataFrame({
            'date': ['2024-01-01'],
            'Code': ['AAPL'],
            'Close': [100.0],
            'signal': [-1],
        })
        result = node.execute(df)
        self.assertEqual(len(result.orders), 1)
        self.assertEqual(result.orders[0].size, -1.0)


if __name__ == '__main__':
    unittest.main()
