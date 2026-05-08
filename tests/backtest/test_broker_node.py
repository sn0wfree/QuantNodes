# -*- coding: utf-8 -*-
"""BrokerNode 单元测试"""
import unittest
import numpy as np
import pandas as pd
from QuantNodes.backtest.broker_node import (
    Trade, TradeResult, SimulatedBrokerNode, ExecutionBrokerNode,
)
from QuantNodes.backtest.strategy_node import Order, OrdersResult


def make_quote_data(dates=None, codes=None, start_price=100, n_days=10, seed=42):
    np.random.seed(seed)
    if dates is None:
        dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    if codes is None:
        codes = ['AAPL']
    records = []
    for code in codes:
        for i, d in enumerate(dates):
            records.append({
                'date': str(d.date()), 'Code': code,
                'Open': start_price + i * 0.5 + np.random.randn() * 0.1,
                'High': start_price + i * 0.5 + 5 + np.random.randn() * 0.1,
                'Low': start_price + i * 0.5 - 5 + np.random.randn() * 0.1,
                'Close': start_price + i * 0.5 + np.random.randn() * 0.1,
                'Volume': 1_000_000,
            })
    return pd.DataFrame(records)


class TestTrade(unittest.TestCase):
    def test_required_fields(self):
        t = Trade(order_id='o1', code='AAPL', side='buy', size=100.0,
                  price=100.0, adjusted_price=100.1, fee=10.01, dt='2024-01-01')
        self.assertEqual(t.code, 'AAPL')
        self.assertEqual(t.side, 'buy')

    def test_default_status(self):
        t = Trade(order_id='o1', code='AAPL', side='buy', size=100.0,
                  price=100.0, adjusted_price=100.1, fee=10.01, dt='2024-01-01')
        self.assertEqual(t.status, 'completed')


class TestTradeResult(unittest.TestCase):
    def test_defaults(self):
        r = TradeResult()
        self.assertEqual(r.trades, [])
        self.assertEqual(r.cash, 0.0)

    def test_to_dataframe_empty(self):
        r = TradeResult()
        df = r.to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)

    def test_to_dataframe_with_trades(self):
        r = TradeResult()
        r.trades.append(Trade(order_id='o1', code='AAPL', side='buy', size=100.0,
                              price=100.0, adjusted_price=100.1, fee=10.01, dt='2024-01-01'))
        r.trades.append(Trade(order_id='o2', code='GOOG', side='sell', size=50.0,
                              price=200.0, adjusted_price=199.8, fee=9.99, dt='2024-01-10'))
        df = r.to_dataframe()
        self.assertEqual(len(df), 2)
        self.assertIn('code', df.columns)


class TestBrokerNodeInit(unittest.TestCase):
    def test_default_config(self):
        b = SimulatedBrokerNode()
        self.assertEqual(b._cash, 100000)
        self.assertEqual(b._commission, 0.001)
        self.assertEqual(b._margin, 0.1)
        self.assertEqual(b._leverage, 10.0)

    def test_custom_config(self):
        b = SimulatedBrokerNode(config={
            'cash': 500000, 'commission': 0.002, 'margin': 0.2,
            'trade_on_close': True, 'hedging': True,
        })
        self.assertEqual(b._cash, 500000)
        self.assertEqual(b._margin, 0.2)
        self.assertEqual(b._leverage, 5.0)
        self.assertTrue(b._trade_on_close)


class TestBrokerNodeGetters(unittest.TestCase):
    def test_get_positions_empty(self):
        b = SimulatedBrokerNode()
        self.assertEqual(b.get_positions(), {})

    def test_get_positions_returns_copy(self):
        b = SimulatedBrokerNode()
        b._positions = {'AAPL': 100}
        pos = b.get_positions()
        pos['AAPL'] = 999
        self.assertEqual(b.get_positions()['AAPL'], 100)

    def test_get_cash(self):
        b = SimulatedBrokerNode(config={'cash': 500000})
        self.assertEqual(b.get_cash(), 500000)


class TestBrokerNodeReset(unittest.TestCase):
    def test_reset_restores_cash(self):
        b = SimulatedBrokerNode(config={'cash': 500000})
        b._cash = 400000
        b.reset()
        self.assertEqual(b._cash, 500000)

    def test_reset_clears_positions(self):
        b = SimulatedBrokerNode()
        b._positions = {'AAPL': 100}
        b.reset()
        self.assertEqual(b._positions, {})

    def test_reset_clears_result(self):
        b = SimulatedBrokerNode()
        b._result = TradeResult()
        b.reset()
        self.assertIsNone(b._result)


class TestSimulatedBrokerNodeExecute(unittest.TestCase):
    def setUp(self):
        self.quote_data = make_quote_data(n_days=10)

    def test_execute_empty_orders(self):
        b = SimulatedBrokerNode(config={'cash': 100000})
        result = b.execute((OrdersResult(), self.quote_data))
        self.assertEqual(result.cash, 100000)
        self.assertEqual(len(result.trades), 0)

    def test_execute_empty_quote_data(self):
        b = SimulatedBrokerNode(config={'cash': 100000})
        orders = OrdersResult()
        orders.add_order(Order(code='AAPL', size=100, create_date='2024-01-01'))
        result = b.execute((orders, pd.DataFrame()))
        self.assertEqual(len(result.trades), 0)

    def test_single_buy(self):
        b = SimulatedBrokerNode(config={'cash': 100000, 'commission': 0.001})
        orders = OrdersResult()
        orders.add_order(Order(code='AAPL', size=100, create_date='2024-01-01'))
        result = b.execute((orders, self.quote_data))
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].side, 'buy')

    def test_single_sell(self):
        b = SimulatedBrokerNode(config={'cash': 100000, 'commission': 0.001})
        b._positions = {'AAPL': 100}
        orders = OrdersResult()
        orders.add_order(Order(code='AAPL', size=-50, create_date='2024-01-01'))
        result = b.execute((orders, self.quote_data))
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].side, 'sell')

    def test_insufficient_cash_rejects(self):
        b = SimulatedBrokerNode(config={'cash': 100, 'commission': 0.001})
        orders = OrdersResult()
        orders.add_order(Order(code='AAPL', size=1000, create_date='2024-01-01'))
        result = b.execute((orders, self.quote_data))
        self.assertEqual(len(result.trades), 0)

    def test_sell_without_position(self):
        b = SimulatedBrokerNode(config={'cash': 100000, 'commission': 0.001})
        orders = OrdersResult()
        orders.add_order(Order(code='AAPL', size=-50, create_date='2024-01-01'))
        result = b.execute((orders, self.quote_data))
        self.assertEqual(len(result.trades), 1)


class TestSimulatedBrokerNodeAdjustedPrice(unittest.TestCase):
    def test_adjusted_price_buy(self):
        b = SimulatedBrokerNode(config={'commission': 0.001})
        adj = b._adjusted_price(100.0, 100)
        self.assertAlmostEqual(adj, 100.1, places=2)

    def test_adjusted_price_sell(self):
        b = SimulatedBrokerNode(config={'commission': 0.001})
        adj = b._adjusted_price(100.0, -100)
        self.assertAlmostEqual(adj, 99.9, places=2)


class TestSimulatedBrokerNodePositionTracking(unittest.TestCase):
    def test_position_accumulates(self):
        b = SimulatedBrokerNode(config={'cash': 1000000, 'commission': 0.001})
        q = make_quote_data(n_days=20)
        o1 = OrdersResult()
        o1.add_order(Order(code='AAPL', size=100, create_date='2024-01-05'))
        b.execute((o1, q))
        self.assertEqual(b._positions.get('AAPL'), 100)
        o2 = OrdersResult()
        o2.add_order(Order(code='AAPL', size=50, create_date='2024-01-15'))
        b.execute((o2, q))
        self.assertEqual(b._positions.get('AAPL'), 150)

    def test_position_reduces_on_sell(self):
        b = SimulatedBrokerNode(config={'cash': 1000000, 'commission': 0.001})
        b._positions = {'AAPL': 200}
        q = make_quote_data(n_days=20)
        orders = OrdersResult()
        orders.add_order(Order(code='AAPL', size=-80, create_date='2024-01-15'))
        b.execute((orders, q))
        self.assertEqual(b._positions.get('AAPL'), 120)


class TestExecutionBrokerNodeExecute(unittest.TestCase):
    def setUp(self):
        self.quote_data = make_quote_data(n_days=10)

    def test_creation(self):
        b = ExecutionBrokerNode(config={'cash': 100000, 'slippage': 0.001, 'commission': 0.001})
        self.assertEqual(b._slippage, 0.001)

    def test_default_slippage(self):
        b = ExecutionBrokerNode()
        self.assertEqual(b._slippage, 0.0005)

    def test_empty_orders(self):
        b = ExecutionBrokerNode(config={'cash': 100000})
        result = b.execute((OrdersResult(), self.quote_data))
        self.assertEqual(result.cash, 100000)

    def test_single_buy(self):
        b = ExecutionBrokerNode(config={'cash': 100000, 'slippage': 0.001, 'commission': 0.001})
        orders = OrdersResult()
        orders.add_order(Order(code='AAPL', size=100, create_date='2024-01-01'))
        result = b.execute((orders, self.quote_data))
        self.assertEqual(len(result.trades), 1)

    def test_multi_day_orders(self):
        b = ExecutionBrokerNode(config={'cash': 100000, 'slippage': 0.001, 'commission': 0.001})
        orders = OrdersResult()
        orders.add_order(Order(code='AAPL', size=100, create_date='2024-01-01'))
        orders.add_order(Order(code='AAPL', size=50, create_date='2024-01-03'))
        result = b.execute((orders, self.quote_data))
        self.assertEqual(len(result.trades), 2)

    def test_multi_stock(self):
        q = make_quote_data(codes=['AAPL', 'GOOG'], n_days=10)
        b = ExecutionBrokerNode(config={'cash': 1000000, 'slippage': 0.001, 'commission': 0.001})
        orders = OrdersResult()
        orders.add_order(Order(code='AAPL', size=100, create_date='2024-01-01'))
        orders.add_order(Order(code='GOOG', size=50, create_date='2024-01-01'))
        result = b.execute((orders, q))
        self.assertEqual(len(result.trades), 2)

    def test_sell_short(self):
        b = ExecutionBrokerNode(config={'cash': 100000, 'slippage': 0.001, 'commission': 0.001})
        orders = OrdersResult()
        orders.add_order(Order(code='AAPL', size=-100, create_date='2024-01-01'))
        result = b.execute((orders, self.quote_data))
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].side, 'sell')

    def test_trade_on_close(self):
        b = ExecutionBrokerNode(config={
            'cash': 100000, 'slippage': 0.001, 'commission': 0.001, 'trade_on_close': True,
        })
        orders = OrdersResult()
        orders.add_order(Order(code='AAPL', size=100, create_date='2024-01-01'))
        result = b.execute((orders, self.quote_data))
        self.assertEqual(len(result.trades), 1)


class TestBrokerNodeValidateInput(unittest.TestCase):
    def test_validate_none_passes(self):
        b = SimulatedBrokerNode()
        b._validate_input(None)

    def test_validate_orders_result_passes(self):
        b = SimulatedBrokerNode()
        b._validate_input(OrdersResult())

    def test_validate_tuple_passes(self):
        b = SimulatedBrokerNode()
        b._validate_input((OrdersResult(), pd.DataFrame({'a': [1]})))

    def test_validate_invalid_raises(self):
        b = SimulatedBrokerNode()
        with self.assertRaises(ValueError):
            b._validate_input('invalid')


if __name__ == '__main__':
    unittest.main()
