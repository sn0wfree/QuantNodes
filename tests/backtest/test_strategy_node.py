# -*- coding: utf-8 -*-
"""StrategyNode 单元测试"""
import unittest
import numpy as np
import pandas as pd
from QuantNodes.backtest.strategy_node import Order, Signal, OrdersResult, StrategyNode, MAStrategyNode, MomentumStrategyNode


def make_quote_data(n_days=50, codes=None, start_price=100, seed=42):
    np.random.seed(seed)
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    if codes is None:
        codes = ['AAPL']
    records = []
    for code in codes:
        for i, d in enumerate(dates):
            close = start_price + i * 0.5 + np.random.randn() * 0.5
            records.append({
                'date': str(d.date()), 'Code': code,
                'Open': close - 0.5, 'High': close + 2.0, 'Low': close - 2.0,
                'Close': close, 'Volume': 1_000_000,
            })
    return pd.DataFrame(records)


class MockStrategyNode(StrategyNode):
    def _generate_signals(self, input_data, **kwargs):
        if input_data is None or input_data.empty:
            return []
        return [Signal(code='AAPL', signal_type='buy', strength=1.0, price=100.0, date='2024-01-01')]


class TestOrder(unittest.TestCase):
    def test_required_fields(self):
        order = Order(code='AAPL', size=100)
        self.assertEqual(order.code, 'AAPL')
        self.assertEqual(order.size, 100)

    def test_all_fields(self):
        order = Order(code='AAPL', size=100, limit_price=150.0, stop_price=145.0,
                     sl_price=140.0, tp_price=160.0, order_id='o1', create_date='2024-01-01')
        self.assertEqual(order.limit_price, 150.0)
        self.assertEqual(order.stop_price, 145.0)

    def test_defaults(self):
        order = Order(code='GOOG', size=-50)
        self.assertIsNone(order.limit_price)
        self.assertIsNone(order.order_id)


class TestSignal(unittest.TestCase):
    def test_required_fields(self):
        signal = Signal(code='AAPL', signal_type='buy')
        self.assertEqual(signal.code, 'AAPL')
        self.assertEqual(signal.signal_type, 'buy')

    def test_all_fields(self):
        signal = Signal(code='AAPL', signal_type='sell', strength=1.5, price=150.0, date='2024-01-01')
        self.assertEqual(signal.strength, 1.5)
        self.assertEqual(signal.price, 150.0)

    def test_default_strength(self):
        signal = Signal(code='AAPL', signal_type='buy')
        self.assertEqual(signal.strength, 1.0)


class TestOrdersResult(unittest.TestCase):
    def test_empty(self):
        r = OrdersResult()
        self.assertEqual(r.orders, [])
        self.assertEqual(r.signals, [])

    def test_add_order(self):
        r = OrdersResult()
        r.add_order(Order(code='AAPL', size=100))
        self.assertEqual(len(r.orders), 1)

    def test_add_signal(self):
        r = OrdersResult()
        r.add_signal(Signal(code='AAPL', signal_type='buy'))
        self.assertEqual(len(r.signals), 1)

    def test_to_dataframe_empty(self):
        r = OrdersResult()
        df = r.to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)

    def test_to_dataframe_with_orders(self):
        r = OrdersResult()
        r.add_order(Order(code='AAPL', size=100, order_id='o1'))
        r.add_order(Order(code='GOOG', size=-50, order_id='o2'))
        df = r.to_dataframe()
        self.assertEqual(len(df), 2)
        self.assertIn('code', df.columns)
        self.assertIn('size', df.columns)


class TestStrategyNodeInit(unittest.TestCase):
    def test_default_name(self):
        node = MockStrategyNode()
        self.assertEqual(node.name, 'MockStrategyNode')

    def test_custom_name(self):
        node = MockStrategyNode(name='MyStrategy')
        self.assertEqual(node.name, 'MyStrategy')


class TestStrategyNodeExecute(unittest.TestCase):
    def setUp(self):
        self.df = make_quote_data(n_days=30)

    def test_execute_empty(self):
        node = MAStrategyNode(config={'short_window': 5, 'long_window': 20})
        result = node.execute(pd.DataFrame())
        self.assertIsInstance(result, OrdersResult)
        self.assertEqual(len(result.signals), 0)

    def test_execute_with_data(self):
        node = MAStrategyNode(config={'short_window': 5, 'long_window': 20})
        result = node.execute(self.df)
        self.assertIsInstance(result, OrdersResult)
        self.assertIsInstance(result.signals, list)


class TestMAStrategyNode(unittest.TestCase):
    def setUp(self):
        self.df = make_quote_data(n_days=50, start_price=100)

    def test_creation(self):
        s = MAStrategyNode(config={'short_window': 5, 'long_window': 20})
        self.assertEqual(s._short_window, 5)
        self.assertEqual(s._long_window, 20)

    def test_defaults(self):
        s = MAStrategyNode()
        self.assertEqual(s._short_window, 5)
        self.assertEqual(s._long_window, 20)

    def test_produces_signals(self):
        s = MAStrategyNode(config={'short_window': 5, 'long_window': 10})
        result = s.execute(self.df)
        self.assertIsInstance(result.signals, list)
        for sig in result.signals:
            self.assertIn(sig.signal_type, ('buy', 'sell'))

    def test_empty_dataframe(self):
        s = MAStrategyNode(config={'short_window': 5, 'long_window': 10})
        result = s.execute(pd.DataFrame())
        self.assertEqual(len(result.signals), 0)

    def test_multi_stock(self):
        data = make_quote_data(n_days=50, codes=['AAPL', 'GOOG'])
        s = MAStrategyNode(config={'short_window': 5, 'long_window': 10})
        result = s.execute(data)
        codes = set(sig.code for sig in result.signals)
        self.assertIn('AAPL', codes)

    def test_buy_and_sell_signals(self):
        s = MAStrategyNode(config={'short_window': 5, 'long_window': 10})
        result = s.execute(self.df)
        types = set(s.signal_type for s in result.signals)
        self.assertIn('buy', types)


class TestMomentumStrategyNode(unittest.TestCase):
    def setUp(self):
        self.df = make_quote_data(n_days=50, start_price=100)

    def test_creation(self):
        s = MomentumStrategyNode(config={'lookback': 10, 'threshold': 0.05})
        self.assertEqual(s._lookback, 10)
        self.assertEqual(s._threshold, 0.05)

    def test_defaults(self):
        s = MomentumStrategyNode()
        self.assertEqual(s._lookback, 20)
        self.assertEqual(s._threshold, 0.05)

    def test_produces_signals(self):
        s = MomentumStrategyNode(config={'lookback': 5, 'threshold': 0.02})
        result = s.execute(self.df)
        self.assertIsInstance(result.signals, list)

    def test_empty_dataframe(self):
        s = MomentumStrategyNode(config={'lookback': 5, 'threshold': 0.02})
        result = s.execute(pd.DataFrame())
        self.assertEqual(len(result.signals), 0)

    def test_multi_stock(self):
        data = make_quote_data(n_days=50, codes=['AAPL', 'GOOG'])
        s = MomentumStrategyNode(config={'lookback': 5, 'threshold': 0.02})
        result = s.execute(data)
        codes = set(sig.code for sig in result.signals)
        self.assertIn('AAPL', codes)


class TestStrategyNodeGetters(unittest.TestCase):
    def setUp(self):
        self.df = make_quote_data(n_days=30)

    def test_get_signals_before(self):
        node = MAStrategyNode()
        self.assertEqual(node.get_signals(), [])

    def test_get_orders_before(self):
        node = MAStrategyNode()
        self.assertEqual(node.get_orders(), [])


class TestStrategyNodeCreateOrders(unittest.TestCase):
    def test_create_orders_from_signals(self):
        node = MockStrategyNode()
        signals = [
            Signal(code='AAPL', signal_type='buy', strength=1.0, price=100.0, date='2024-01-01'),
            Signal(code='GOOG', signal_type='sell', strength=0.5, price=200.0, date='2024-01-02'),
        ]
        orders = node._create_orders(signals)
        self.assertEqual(len(orders), 2)
        self.assertEqual(orders[0].code, 'AAPL')
        self.assertEqual(orders[0].size, 1.0)
        self.assertEqual(orders[1].size, -0.5)


class TestStrategyNodeValidateInput(unittest.TestCase):
    def test_validate_none_passes(self):
        node = MAStrategyNode()
        node._validate_input(None)

    def test_validate_dataframe_passes(self):
        node = MAStrategyNode()
        node._validate_input(pd.DataFrame({'a': [1]}))

    def test_validate_invalid_raises(self):
        node = MAStrategyNode()
        with self.assertRaises(ValueError):
            node._validate_input('invalid')


if __name__ == '__main__':
    unittest.main()
