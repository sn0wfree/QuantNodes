# coding=utf-8
"""
BacktestNode 单元测试

测试回测节点体系的各个组件。
"""
import unittest
from datetime import datetime, timedelta
import uuid

import numpy as np
import pandas as pd

from QuantNodes.backtest.backtest_node import BacktestNode, BacktestResult, BacktestPipeline
from QuantNodes.backtest.strategy_node import (
    StrategyNode, Order, Signal, OrdersResult,
    MAStrategyNode, MomentumStrategyNode
)
from QuantNodes.backtest.broker_node import (
    BrokerNode, Trade, TradeResult,
    SimulatedBrokerNode, ExecutionBrokerNode
)
from QuantNodes.backtest.risk_node import (
    RiskNode, RiskCheck, RiskResult,
    PositionLimitRiskNode, StopLossRiskNode, CashRiskNode, CompositeRiskNode
)
from QuantNodes.core.node import NodeState


class MockBacktestNode(BacktestNode):
    """用于测试的模拟回测节点"""

    def _run_backtest(self, quote_data, signals, **kwargs):
        result = BacktestResult()
        result.final_cash = self._cash
        result.statistics = {
            'total_trades': 0,
            'total_return': 0.0,
        }
        return result


class TestBacktestNode(unittest.TestCase):
    """BacktestNode 基类测试"""

    def test_backtest_node_creation(self):
        """测试 BacktestNode 创建"""
        config = {
            'cash': 100000,
            'commission': 0.001,
            'margin': 0.1,
        }
        node = MockBacktestNode(name="TestBacktest", config=config)
        self.assertEqual(node.name, "TestBacktest")
        self.assertEqual(node._cash, 100000)
        self.assertEqual(node._commission, 0.001)

    def test_backtest_node_execute(self):
        """测试 BacktestNode 执行"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        quote_data = pd.DataFrame({
            'date': [str(d.date()) for d in dates],
            'Code': ['AAPL'] * 10,
            'Open': [100 + i for i in range(10)],
            'High': [105 + i for i in range(10)],
            'Low': [95 + i for i in range(10)],
            'Close': [102 + i for i in range(10)],
            'Volume': [1000000] * 10,
        })
        node = MockBacktestNode(config={'cash': 100000})
        result = node.execute(quote_data)
        self.assertIsInstance(result, BacktestResult)
        self.assertEqual(result.final_cash, 100000)

    def test_backtest_result_properties(self):
        """测试 BacktestResult 属性"""
        result = BacktestResult()
        result.final_cash = 95000
        result.total_return = 0.05
        self.assertEqual(result.final_cash, 95000)
        self.assertEqual(result.total_return, 0.05)


class TestStrategyNode(unittest.TestCase):
    """StrategyNode 测试"""

    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        self.quote_data = pd.DataFrame({
            'date': dates,
            'Code': ['AAPL'] * 50,
            'Open': 100 + np.random.randn(50).cumsum(),
            'High': 105 + np.random.randn(50).cumsum(),
            'Low': 95 + np.random.randn(50).cumsum(),
            'Close': 100 + np.random.randn(50).cumsum(),
            'Volume': np.random.randint(1000000, 10000000, 50),
        })

    def test_ma_strategy_creation(self):
        """测试 MA 策略创建"""
        strategy = MAStrategyNode(config={
            'short_window': 5,
            'long_window': 20
        })
        self.assertEqual(strategy._short_window, 5)
        self.assertEqual(strategy._long_window, 20)

    def test_ma_strategy_execute(self):
        """测试 MA 策略执行"""
        strategy = MAStrategyNode(config={
            'short_window': 5,
            'long_window': 10
        })
        result = strategy.execute(self.quote_data)
        self.assertIsInstance(result, OrdersResult)
        self.assertIsInstance(result.signals, list)
        self.assertIsInstance(result.orders, list)

    def test_momentum_strategy_creation(self):
        """测试动量策略创建"""
        strategy = MomentumStrategyNode(config={
            'lookback': 10,
            'threshold': 0.02
        })
        self.assertEqual(strategy._lookback, 10)
        self.assertEqual(strategy._threshold, 0.02)

    def test_momentum_strategy_execute(self):
        """测试动量策略执行"""
        strategy = MomentumStrategyNode(config={
            'lookback': 10,
            'threshold': 0.05
        })
        result = strategy.execute(self.quote_data)
        self.assertIsInstance(result, OrdersResult)

    def test_order_creation(self):
        """测试订单创建"""
        order = Order(
            code='AAPL',
            size=100,
            limit_price=150.0,
            create_date='2024-01-01'
        )
        self.assertEqual(order.code, 'AAPL')
        self.assertEqual(order.size, 100)

    def test_signal_creation(self):
        """测试信号创建"""
        signal = Signal(
            code='AAPL',
            signal_type='buy',
            strength=1.0,
            price=150.0,
            date='2024-01-01'
        )
        self.assertEqual(signal.code, 'AAPL')
        self.assertEqual(signal.signal_type, 'buy')

    def test_orders_result_to_dataframe(self):
        """测试 OrdersResult 转换 DataFrame"""
        orders_result = OrdersResult()
        orders_result.add_order(Order(code='AAPL', size=100))
        orders_result.add_order(Order(code='GOOG', size=50))
        df = orders_result.to_dataframe()
        self.assertEqual(len(df), 2)
        self.assertIn('code', df.columns)
        self.assertIn('size', df.columns)


class TestBrokerNode(unittest.TestCase):
    """BrokerNode 测试"""

    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        self.quote_data = pd.DataFrame({
            'date': [str(d.date()) for d in dates],
            'Code': ['AAPL'] * 10,
            'Open': [100 + i for i in range(10)],
            'High': [105 + i for i in range(10)],
            'Low': [95 + i for i in range(10)],
            'Close': [102 + i for i in range(10)],
            'Volume': [1000000] * 10,
        })

    def test_simulated_broker_creation(self):
        """测试模拟经纪商创建"""
        broker = SimulatedBrokerNode(config={
            'cash': 100000,
            'commission': 0.001,
            'margin': 0.1,
        })
        self.assertEqual(broker._cash, 100000)
        self.assertEqual(broker._commission, 0.001)

    def test_simulated_broker_execute(self):
        """测试模拟经纪商执行"""
        broker = SimulatedBrokerNode(config={'cash': 100000})

        orders_result = OrdersResult()
        orders_result.add_order(Order(
            code='AAPL',
            size=100,
            limit_price=100.0,
            create_date='2024-01-01'
        ))

        result = broker.execute((orders_result, self.quote_data))
        self.assertIsInstance(result, TradeResult)
        self.assertIsInstance(result.trades, list)

    def test_execution_broker_creation(self):
        """测试执行经纪商创建"""
        broker = ExecutionBrokerNode(config={
            'cash': 100000,
            'slippage': 0.001,
        })
        self.assertEqual(broker._slippage, 0.001)

    def test_execution_broker_execute(self):
        """测试执行经纪商执行"""
        broker = ExecutionBrokerNode(config={
            'cash': 100000,
            'slippage': 0.001,
        })

        orders_result = OrdersResult()
        orders_result.add_order(Order(
            code='AAPL',
            size=100,
            create_date='2024-01-01'
        ))

        result = broker.execute((orders_result, self.quote_data))
        self.assertIsInstance(result, TradeResult)

    def test_trade_creation(self):
        """测试 Trade 创建"""
        trade = Trade(
            order_id='test_order',
            code='AAPL',
            side='buy',
            size=100,
            price=100.0,
            adjusted_price=100.1,
            fee=10.0,
            dt='2024-01-01',
        )
        self.assertEqual(trade.code, 'AAPL')
        self.assertEqual(trade.side, 'buy')
        self.assertEqual(trade.size, 100)

    def test_trade_result_to_dataframe(self):
        """测试 TradeResult 转换 DataFrame"""
        result = TradeResult()
        result.trades.append(Trade(
            order_id='order1',
            code='AAPL',
            side='buy',
            size=100,
            price=100.0,
            adjusted_price=100.1,
            fee=10.0,
            dt='2024-01-01',
        ))
        df = result.to_dataframe()
        self.assertEqual(len(df), 1)
        self.assertIn('code', df.columns)


class TestRiskNode(unittest.TestCase):
    """RiskNode 测试"""

    def test_position_limit_risk_creation(self):
        """测试仓位限制风控创建"""
        risk = PositionLimitRiskNode(config={
            'max_position': 1000,
            'max_order_size': 100,
        })
        self.assertEqual(risk._max_position, 1000)
        self.assertEqual(risk._max_order_size, 100)

    def test_position_limit_risk_pass(self):
        """测试仓位限制通过"""
        risk = PositionLimitRiskNode(config={'max_position': 1000})
        order = Order(code='AAPL', size=100)
        positions = {'AAPL': 500}
        check = risk._check_order(order, positions)
        self.assertTrue(check.passed)

    def test_position_limit_risk_reject(self):
        """测试仓位限制拒绝"""
        risk = PositionLimitRiskNode(config={'max_position': 1000})
        order = Order(code='AAPL', size=600)
        positions = {'AAPL': 500}
        check = risk._check_order(order, positions)
        self.assertFalse(check.passed)

    def test_position_limit_risk_adjust(self):
        """测试仓位限制调整"""
        risk = PositionLimitRiskNode(config={'max_position': 1000, 'max_order_size': 500})
        order = Order(code='AAPL', size=600)
        positions = {'AAPL': 0}
        check = risk._check_order(order, positions)
        self.assertTrue(check.passed)
        self.assertIsNotNone(check.adjusted_size)
        self.assertEqual(check.adjusted_size, 500)

    def test_stop_loss_risk_creation(self):
        """测试止损风控创建"""
        risk = StopLossRiskNode(config={'max_loss': 10000})
        self.assertEqual(risk._max_loss, 10000)

    def test_cash_risk_creation(self):
        """测试现金风控创建"""
        risk = CashRiskNode(config={'min_cash': 1000})
        self.assertEqual(risk._min_cash, 1000)

    def test_composite_risk_creation(self):
        """测试复合风控创建"""
        risk1 = PositionLimitRiskNode(config={'max_position': 1000})
        risk2 = CashRiskNode(config={'min_cash': 1000})
        composite = CompositeRiskNode(
            config={'mode': 'all'},
            risk_nodes=[risk1, risk2]
        )
        self.assertEqual(len(composite._risk_nodes), 2)

    def test_risk_check_creation(self):
        """测试 RiskCheck 创建"""
        order = Order(code='AAPL', size=100)
        check = RiskCheck(passed=True, order=order)
        self.assertTrue(check.passed)
        self.assertEqual(check.order.code, 'AAPL')

    def test_risk_result_to_dataframe(self):
        """测试 RiskResult 转换 DataFrame"""
        result = RiskResult()
        result.passed_orders.append(Order(code='AAPL', size=100))
        result.passed_orders.append(Order(code='GOOG', size=50))
        df = result.to_dataframe()
        self.assertEqual(len(df), 2)

    def test_orders_result_empty(self):
        """测试空 OrdersResult"""
        result = OrdersResult()
        df = result.to_dataframe()
        self.assertEqual(len(df), 0)


class TestBacktestPipeline(unittest.TestCase):
    """BacktestPipeline 测试"""

    def test_pipeline_creation(self):
        """测试管道创建"""
        node1 = MockBacktestNode(name="Backtest1")
        node2 = MockBacktestNode(name="Backtest2")
        pipeline = BacktestPipeline([node1, node2])
        self.assertEqual(len(pipeline.nodes), 2)

    def test_pipeline_execute(self):
        """测试管道执行"""
        node1 = MockBacktestNode(name="Backtest1", config={'cash': 100000})
        node2 = MockBacktestNode(name="Backtest2", config={'cash': 100000})
        pipeline = BacktestPipeline([node1, node2])
        self.assertEqual(len(pipeline.nodes), 2)

    def test_pipeline_rshift(self):
        """测试管道 >> 运算符"""
        node1 = MockBacktestNode(name="Backtest1")
        node2 = MockBacktestNode(name="Backtest2")
        node3 = MockBacktestNode(name="Backtest3")
        pipeline = node1 >> node2 >> node3
        self.assertEqual(len(pipeline.nodes), 3)


class TestNodeIntegration(unittest.TestCase):
    """节点集成测试"""

    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        self.quote_data = pd.DataFrame({
            'date': [str(d.date()) for d in dates],
            'Code': ['AAPL'] * 30,
            'Open': [100 + i * 0.5 for i in range(30)],
            'High': [105 + i * 0.5 for i in range(30)],
            'Low': [95 + i * 0.5 for i in range(30)],
            'Close': [100 + i * 0.5 for i in range(30)],
            'Volume': [1000000] * 30,
        })

    def test_strategy_broker_integration(self):
        """测试策略-经纪商集成"""
        strategy = MAStrategyNode(config={'short_window': 5, 'long_window': 10})
        broker = SimulatedBrokerNode(config={'cash': 100000, 'commission': 0.001})

        orders_result = strategy.execute(self.quote_data)
        trade_result = broker.execute((orders_result, self.quote_data))

        self.assertIsInstance(trade_result, TradeResult)
        self.assertGreaterEqual(trade_result.cash, 0)

    def test_full_backtest_flow(self):
        """测试完整回测流程"""
        strategy = MAStrategyNode(config={'short_window': 5, 'long_window': 10})
        broker = SimulatedBrokerNode(config={'cash': 100000, 'commission': 0.001})
        risk = PositionLimitRiskNode(config={'max_position': 10000, 'max_order_size': 1000})

        orders_result = strategy.execute(self.quote_data)

        risk_result = risk.execute((orders_result, {}))

        filtered_orders = OrdersResult()
        filtered_orders.orders = risk_result.passed_orders
        filtered_orders.signals = orders_result.signals
        trade_result = broker.execute((filtered_orders, self.quote_data))

        self.assertIsInstance(trade_result, TradeResult)
        self.assertGreaterEqual(trade_result.cash, 0)


if __name__ == '__main__':
    unittest.main()