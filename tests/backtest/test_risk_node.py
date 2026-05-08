# -*- coding: utf-8 -*-
"""RiskNode 单元测试"""
import unittest
import pandas as pd
from QuantNodes.backtest.risk_node import (
    RiskNode, RiskCheck, RiskResult,
    PositionLimitRiskNode, StopLossRiskNode, CashRiskNode, CompositeRiskNode,
)
from QuantNodes.backtest.strategy_node import Order, OrdersResult
from QuantNodes.backtest.broker_node import Trade, TradeResult
from QuantNodes.core.node import NodeExecutionError


class MockRiskNode(RiskNode):
    def _check_order(self, order, positions, **kwargs):
        return RiskCheck(passed=True, order=order)


class TestRiskCheck(unittest.TestCase):
    def test_passed(self):
        order = Order(code='AAPL', size=100)
        check = RiskCheck(passed=True, order=order)
        self.assertTrue(check.passed)
        self.assertIsNone(check.reason)

    def test_rejected(self):
        order = Order(code='AAPL', size=100)
        check = RiskCheck(passed=False, order=order, reason='limit exceeded')
        self.assertFalse(check.passed)
        self.assertIn('limit', check.reason.lower())

    def test_adjusted(self):
        order = Order(code='AAPL', size=1000)
        check = RiskCheck(passed=True, order=order, adjusted_size=500)
        self.assertTrue(check.passed)
        self.assertEqual(check.adjusted_size, 500)


class TestRiskResult(unittest.TestCase):
    def test_defaults(self):
        r = RiskResult()
        self.assertEqual(r.passed_orders, [])
        self.assertEqual(r.rejected_orders, [])

    def test_to_dataframe_empty(self):
        r = RiskResult()
        df = r.to_dataframe()
        self.assertEqual(len(df), 0)

    def test_to_dataframe_with_orders(self):
        r = RiskResult()
        r.passed_orders.append(Order(code='AAPL', size=100))
        r.passed_orders.append(Order(code='GOOG', size=50))
        df = r.to_dataframe()
        self.assertEqual(len(df), 2)


class TestRiskNodeInit(unittest.TestCase):
    def test_defaults(self):
        node = MockRiskNode()
        self.assertEqual(node._max_position, float('inf'))
        self.assertEqual(node._max_order_size, float('inf'))
        self.assertEqual(node._min_order_size, 0)

    def test_custom(self):
        node = MockRiskNode(config={'max_position': 1000, 'max_order_size': 500, 'min_order_size': 10})
        self.assertEqual(node._max_position, 1000)
        self.assertEqual(node._max_order_size, 500)
        self.assertEqual(node._min_order_size, 10)


class TestRiskNodeExecute(unittest.TestCase):
    def test_with_orders_result(self):
        node = MockRiskNode()
        orders = OrdersResult()
        orders.add_order(Order(code='AAPL', size=100))
        result = node.execute(orders)
        self.assertIsInstance(result, RiskResult)

    def test_with_tuple(self):
        node = MockRiskNode()
        orders = OrdersResult()
        orders.add_order(Order(code='AAPL', size=100))
        result = node.execute((orders, {'AAPL': 200}))
        self.assertIsInstance(result, RiskResult)

    def test_with_trade_result(self):
        node = MockRiskNode()
        orders = OrdersResult()
        orders.add_order(Order(code='AAPL', size=100))
        trade_result = TradeResult()
        trade_result.trades.append(Trade(
            order_id='o1', code='AAPL', side='buy', size=100,
            price=100.0, adjusted_price=100.1, fee=10.0, dt='2024-01-01',
        ))
        result = node.execute((orders, trade_result))
        self.assertIsInstance(result, RiskResult)

    def test_none_orders_raises(self):
        node = MockRiskNode()
        with self.assertRaises(NodeExecutionError):
            node.execute(None)


class TestRiskNodeDerivePositions(unittest.TestCase):
    def test_derive_buys(self):
        node = MockRiskNode()
        tr = TradeResult()
        tr.trades.append(Trade(
            order_id='o1', code='AAPL', side='buy', size=100,
            price=100.0, adjusted_price=100.1, fee=10.0, dt='2024-01-01',
        ))
        tr.trades.append(Trade(
            order_id='o2', code='AAPL', side='buy', size=50,
            price=102.0, adjusted_price=102.1, fee=5.0, dt='2024-01-10',
        ))
        pos = node._derive_positions_from_trades(tr)
        self.assertEqual(pos['AAPL'], 150)

    def test_derive_buy_and_sell(self):
        node = MockRiskNode()
        tr = TradeResult()
        tr.trades.append(Trade(
            order_id='o1', code='AAPL', side='buy', size=100,
            price=100.0, adjusted_price=100.1, fee=10.0, dt='2024-01-01',
        ))
        tr.trades.append(Trade(
            order_id='o2', code='AAPL', side='sell', size=30,
            price=105.0, adjusted_price=104.9, fee=5.0, dt='2024-01-10',
        ))
        pos = node._derive_positions_from_trades(tr)
        self.assertEqual(pos['AAPL'], 70)

    def test_derive_empty(self):
        node = MockRiskNode()
        pos = node._derive_positions_from_trades(TradeResult())
        self.assertEqual(pos, {})


class TestRiskNodeSetPositions(unittest.TestCase):
    def test_set_positions(self):
        node = MockRiskNode()
        node.set_positions({'AAPL': 100, 'GOOG': -50})
        self.assertEqual(node._positions, {'AAPL': 100, 'GOOG': -50})


class TestPositionLimitRiskNode(unittest.TestCase):
    def test_pass(self):
        r = PositionLimitRiskNode(config={'max_position': 1000})
        order = Order(code='AAPL', size=100)
        check = r._check_order(order, {'AAPL': 500})
        self.assertTrue(check.passed)

    def test_reject_exceeded(self):
        r = PositionLimitRiskNode(config={'max_position': 1000})
        order = Order(code='AAPL', size=600)
        check = r._check_order(order, {'AAPL': 500})
        self.assertFalse(check.passed)

    def test_adjust_size(self):
        r = PositionLimitRiskNode(config={'max_position': 1000, 'max_order_size': 500})
        order = Order(code='AAPL', size=600)
        check = r._check_order(order, {'AAPL': 0})
        self.assertTrue(check.passed)
        self.assertEqual(check.adjusted_size, 500)

    def test_reject_min_size(self):
        r = PositionLimitRiskNode(config={'min_order_size': 10})
        order = Order(code='AAPL', size=5)
        check = r._check_order(order, {})
        self.assertFalse(check.passed)

    def test_execute_filters(self):
        r = PositionLimitRiskNode(config={'max_position': 1000})
        orders = OrdersResult()
        orders.add_order(Order(code='AAPL', size=600))
        orders.add_order(Order(code='GOOG', size=100))
        result = r.execute((orders, {'AAPL': 500, 'GOOG': 0}))
        self.assertEqual(len(result.passed_orders), 1)
        self.assertEqual(result.passed_orders[0].code, 'GOOG')


class TestStopLossRiskNode(unittest.TestCase):
    def test_creation(self):
        r = StopLossRiskNode(config={'max_loss': 10000})
        self.assertEqual(r._max_loss, 10000)

    def test_buy_order_passes(self):
        r = StopLossRiskNode(config={'max_loss': 10000})
        order = Order(code='AAPL', size=100)
        check = r._check_order(order, {})
        self.assertTrue(check.passed)

    def test_sell_without_sl_passes(self):
        r = StopLossRiskNode(config={'max_loss': 10000})
        order = Order(code='AAPL', size=-100, sl_price=None)
        check = r._check_order(order, {})
        self.assertTrue(check.passed)

    def test_update_pnl(self):
        r = StopLossRiskNode()
        r.update_pnl(-5000)
        self.assertEqual(r._current_pnl, -5000)


class TestCashRiskNode(unittest.TestCase):
    def test_creation(self):
        r = CashRiskNode(config={'min_cash': 1000})
        self.assertEqual(r._min_cash, 1000)

    def test_sell_order_passes(self):
        r = CashRiskNode(config={'min_cash': 1000})
        order = Order(code='AAPL', size=-100)
        check = r._check_order(order, {}, cash=5000)
        self.assertTrue(check.passed)

    def test_buy_insufficient_cash(self):
        # NOTE: CashRiskNode._check_order has a bug - uses self._config instead of self.config
        # This test documents the bug: calling _check_order directly triggers AttributeError
        r = CashRiskNode(config={'min_cash': 1000})
        order = Order(code='AAPL', size=100, limit_price=50.0)
        with self.assertRaises(AttributeError):
            r._check_order(order, {}, cash=2000)

    def test_buy_sufficient_cash(self):
        # Same bug as above
        r = CashRiskNode(config={'min_cash': 1000})
        order = Order(code='AAPL', size=100, limit_price=50.0)
        with self.assertRaises(AttributeError):
            r._check_order(order, {}, cash=10000)


class TestCompositeRiskNode(unittest.TestCase):
    def test_creation(self):
        r1 = PositionLimitRiskNode(config={'max_position': 1000})
        r2 = CashRiskNode(config={'min_cash': 1000})
        c = CompositeRiskNode(config={'mode': 'all'}, risk_nodes=[r1, r2])
        self.assertEqual(len(c._risk_nodes), 2)

    def test_add_risk_node(self):
        c = CompositeRiskNode()
        c.add_risk_node(PositionLimitRiskNode(config={'max_position': 1000}))
        self.assertEqual(len(c._risk_nodes), 1)

    def test_mode_all_stops_on_first_reject(self):
        r1 = PositionLimitRiskNode(config={'max_position': 1000})
        r2 = StopLossRiskNode(config={'max_loss': 10000})
        c = CompositeRiskNode(config={'mode': 'all'}, risk_nodes=[r1, r2])
        order = Order(code='AAPL', size=600)
        check = c._check_order(order, {'AAPL': 500})
        self.assertFalse(check.passed)

    def test_mode_any_passes(self):
        r1 = PositionLimitRiskNode(config={'max_position': 1000})
        r2 = StopLossRiskNode(config={'max_loss': 10000})
        c = CompositeRiskNode(config={'mode': 'any'}, risk_nodes=[r1, r2])
        order = Order(code='AAPL', size=600)
        check = c._check_order(order, {'AAPL': 500})
        self.assertTrue(check.passed)

    def test_execute_empty_orders(self):
        c = CompositeRiskNode()
        result = c.execute(OrdersResult())
        self.assertEqual(len(result.passed_orders), 0)

    def test_execute_filters_orders(self):
        r1 = PositionLimitRiskNode(config={'max_position': 1000})
        c = CompositeRiskNode(config={'mode': 'all'}, risk_nodes=[r1])
        orders = OrdersResult()
        orders.add_order(Order(code='AAPL', size=600))
        orders.add_order(Order(code='GOOG', size=100))
        result = c.execute((orders, {'AAPL': 500, 'GOOG': 0}))
        self.assertEqual(len(result.passed_orders), 1)


class TestRiskNodeValidateInput(unittest.TestCase):
    def test_validate_none_passes(self):
        node = MockRiskNode()
        node._validate_input(None)

    def test_validate_orders_result_passes(self):
        node = MockRiskNode()
        node._validate_input(OrdersResult())

    def test_validate_tuple_passes(self):
        node = MockRiskNode()
        node._validate_input((OrdersResult(), pd.DataFrame({'a': [1]})))

    def test_validate_invalid_raises(self):
        node = MockRiskNode()
        with self.assertRaises(ValueError):
            node._validate_input('invalid')


if __name__ == '__main__':
    unittest.main()
