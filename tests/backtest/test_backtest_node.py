# -*- coding: utf-8 -*-
"""BacktestNode 单元测试"""
import unittest
import pandas as pd
from QuantNodes.backtest.backtest_node import BacktestNode, BacktestResult, BacktestPipeline
from QuantNodes.core.node import NodeState, NodeExecutionError


class MockBacktestNode(BacktestNode):
    def _run_backtest(self, quote_data, signals, **kwargs):
        result = BacktestResult()
        result.final_cash = self._cash
        result.statistics = {'total_trades': 0, 'total_return': 0.0}
        return result


class MockBacktestNodeWithData(BacktestNode):
    def _run_backtest(self, quote_data, signals, **kwargs):
        result = BacktestResult()
        result.final_cash = self._cash * 1.05
        result.total_return = 0.05
        result.sharpe_ratio = 1.2
        result.max_drawdown = -0.08
        result.win_rate = 0.55
        result.statistics = {
            'total_trades': 10, 'total_return': 0.05,
            'sharpe_ratio': 1.2, 'max_drawdown': -0.08, 'win_rate': 0.55,
        }
        return result


def make_df():
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    return pd.DataFrame({
        'date': [str(d.date()) for d in dates],
        'Code': ['AAPL'] * 10,
        'Open': [100 + i for i in range(10)],
        'Close': [102 + i for i in range(10)],
        'Volume': [1_000_000] * 10,
    })


class TestBacktestResult(unittest.TestCase):
    def test_defaults(self):
        r = BacktestResult()
        self.assertIsInstance(r.positions, pd.DataFrame)
        self.assertIsInstance(r.trades, pd.DataFrame)
        self.assertIsInstance(r.statistics, dict)
        self.assertEqual(r.final_cash, 0.0)

    def test_with_data(self):
        r = BacktestResult(final_cash=95000.0, total_return=0.05)
        self.assertEqual(r.final_cash, 95000.0)
        self.assertEqual(r.total_return, 0.05)


class TestBacktestNodeInit(unittest.TestCase):
    def test_default_config(self):
        node = MockBacktestNode()
        self.assertEqual(node.name, 'MockBacktestNode')
        self.assertEqual(node._cash, 100000)
        self.assertEqual(node._commission, 0.001)

    def test_custom_config(self):
        node = MockBacktestNode(name='Custom', config={'cash': 500000, 'commission': 0.002})
        self.assertEqual(node.name, 'Custom')
        self.assertEqual(node._cash, 500000)
        self.assertEqual(node._commission, 0.002)

    def test_state_idle(self):
        node = MockBacktestNode()
        self.assertEqual(node.state, NodeState.IDLE)


class TestBacktestNodeExecute(unittest.TestCase):
    def setUp(self):
        self.df = make_df()

    def test_execute_dataframe(self):
        node = MockBacktestNode(config={'cash': 100000})
        result = node.execute(self.df)
        self.assertIsInstance(result, BacktestResult)

    def test_execute_tuple(self):
        node = MockBacktestNode(config={'cash': 100000})
        result = node.execute((self.df, pd.DataFrame({'signal': [1]*10})))
        self.assertIsInstance(result, BacktestResult)

    def test_execute_state_success(self):
        node = MockBacktestNode()
        node.execute(self.df)
        self.assertEqual(node.state, NodeState.SUCCESS)

    def test_execute_invalid_raises(self):
        node = MockBacktestNode()
        with self.assertRaises(NodeExecutionError):
            node.execute([1, 2, 3])

    def test_execute_missing_quote_data_raises(self):
        node = MockBacktestNode()
        with self.assertRaises(NodeExecutionError):
            node.execute(None)


class TestBacktestNodeGetters(unittest.TestCase):
    def setUp(self):
        self.df = make_df()

    def test_get_statistics_before(self):
        node = MockBacktestNode()
        self.assertEqual(node.get_statistics(), {})

    def test_get_statistics_after(self):
        node = MockBacktestNodeWithData()
        node.execute(self.df)
        stats = node.get_statistics()
        self.assertIn('total_trades', stats)

    def test_get_equity_curve_before(self):
        node = MockBacktestNode()
        self.assertTrue(node.get_equity_curve().empty)


class TestBacktestNodeReset(unittest.TestCase):
    def setUp(self):
        self.df = make_df()

    def test_reset_clears_result(self):
        node = MockBacktestNodeWithData()
        node.execute(self.df)
        self.assertIsNotNone(node._result)
        node.reset()
        self.assertIsNone(node._result)

    def test_reset_state_idle(self):
        node = MockBacktestNode()
        node.execute(self.df)
        node.reset()
        self.assertEqual(node.state, NodeState.IDLE)


class TestBacktestPipeline(unittest.TestCase):
    def setUp(self):
        self.df = make_df()

    def test_empty_pipeline(self):
        p = BacktestPipeline([])
        self.assertEqual(len(p.nodes), 0)

    def test_single_node(self):
        node = MockBacktestNode(name='N1')
        p = BacktestPipeline([node])
        self.assertEqual(len(p.nodes), 1)

    def test_multiple_nodes(self):
        node1 = MockBacktestNode(name='N1')
        node2 = MockBacktestNode(name='N2')
        p = BacktestPipeline([node1, node2])
        self.assertEqual(len(p.nodes), 2)

    def test_pipeline_execute(self):
        node1 = MockBacktestNode(config={'cash': 100000})
        p = BacktestPipeline([node1])
        results = p.execute(self.df)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], BacktestResult)

    def test_pipeline_rshift_single(self):
        node1 = MockBacktestNode(name='N1')
        node2 = MockBacktestNode(name='N2')
        p = node1 >> node2
        self.assertEqual(len(p.nodes), 2)

    def test_pipeline_rshift_chain(self):
        node1 = MockBacktestNode(name='N1')
        node2 = MockBacktestNode(name='N2')
        node3 = MockBacktestNode(name='N3')
        p = node1 >> node2 >> node3
        self.assertEqual(len(p.nodes), 3)


if __name__ == '__main__':
    unittest.main()
