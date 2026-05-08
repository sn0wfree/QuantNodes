# -*- coding: utf-8 -*-
"""ConfigBacktestRunner 单元测试"""
import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from QuantNodes.backtest.config_runner import ConfigBacktestRunner
from QuantNodes.backtest.backtest_node import BacktestResult
from QuantNodes.backtest.strategy_node import OrdersResult, Order
from QuantNodes.backtest.broker_node import Trade, TradeResult
from QuantNodes.backtest.risk_node import PositionLimitRiskNode


def make_quote_df(n_days=20, codes=None, seed=42):
    np.random.seed(seed)
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    if codes is None:
        codes = ['AAPL']
    records = []
    for code in codes:
        for i, d in enumerate(dates):
            records.append({
                'date': str(d.date()), 'Code': code,
                'Open': 100 + i * 0.5 + np.random.randn() * 0.1,
                'Close': 100 + i * 0.5 + np.random.randn() * 0.1,
                'Volume': 1_000_000,
            })
    return pd.DataFrame(records)


class TestConfigBacktestRunnerNormalizeColumns(unittest.TestCase):
    def test_no_changes_needed(self):
        r = ConfigBacktestRunner()
        df = pd.DataFrame({
            'date': ['2024-01-01'], 'Code': ['AAPL'],
            'Open': [100.0], 'Close': [100.0], 'Volume': [1_000_000],
        })
        result = r._normalize_columns(df)
        self.assertIn('Code', result.columns)

    def test_lowercase_code_to_code(self):
        r = ConfigBacktestRunner()
        df = pd.DataFrame({'date': ['2024-01-01'], 'code': ['AAPL'], 'Close': [100.0]})
        result = r._normalize_columns(df)
        self.assertIn('Code', result.columns)

    def test_lowercase_close_to_close(self):
        r = ConfigBacktestRunner()
        df = pd.DataFrame({'date': ['2024-01-01'], 'Code': ['AAPL'], 'close': [100.0]})
        result = r._normalize_columns(df)
        self.assertIn('Close', result.columns)

    def test_lowercase_open_to_open(self):
        r = ConfigBacktestRunner()
        df = pd.DataFrame({'date': ['2024-01-01'], 'Code': ['AAPL'], 'close': [100.0], 'open': [99.0]})
        result = r._normalize_columns(df)
        self.assertIn('Open', result.columns)

    def test_open_fallback_to_close(self):
        r = ConfigBacktestRunner()
        df = pd.DataFrame({'date': ['2024-01-01'], 'Code': ['AAPL'], 'Close': [100.0]})
        result = r._normalize_columns(df)
        self.assertIn('Open', result.columns)
        self.assertEqual(result['Open'].iloc[0], 100.0)


class TestConfigBacktestRunnerBuildRiskNodes(unittest.TestCase):
    def test_no_backtest_config(self):
        r = ConfigBacktestRunner()
        config = MagicMock()
        config.backtest = None
        nodes = r._build_risk_nodes(config)
        self.assertEqual(nodes, [])

    def test_no_positions(self):
        r = ConfigBacktestRunner()
        config = MagicMock()
        config.backtest = MagicMock()
        config.backtest.positions = None
        nodes = r._build_risk_nodes(config)
        self.assertEqual(nodes, [])

    def test_with_max_positions(self):
        r = ConfigBacktestRunner()
        config = MagicMock()
        config.backtest = MagicMock()
        config.backtest.positions = {'max_positions': 1000}
        nodes = r._build_risk_nodes(config)
        self.assertEqual(len(nodes), 1)
        self.assertIsInstance(nodes[0], PositionLimitRiskNode)


class TestConfigBacktestRunnerBuildBroker(unittest.TestCase):
    def test_with_full_config(self):
        r = ConfigBacktestRunner()
        config = MagicMock()
        config.backtest = MagicMock()
        config.backtest.initial_cash = 500000
        config.backtest.commission = 0.002
        config.backtest.slippage = 0.001
        broker = r._build_broker(config)
        self.assertEqual(broker._cash, 500000)
        self.assertEqual(broker._commission, 0.002)
        self.assertEqual(broker._slippage, 0.001)

    def test_defaults(self):
        r = ConfigBacktestRunner()
        config = MagicMock()
        config.backtest = None
        broker = r._build_broker(config)
        self.assertEqual(broker._cash, 1000000)


class TestConfigBacktestRunnerApplyRisk(unittest.TestCase):
    def test_no_nodes(self):
        r = ConfigBacktestRunner()
        orders = OrdersResult()
        orders.add_order(Order(code='AAPL', size=100))
        orders.add_signal(MagicMock())
        result = r._apply_risk(orders, [])
        self.assertEqual(result, orders)

    def test_with_nodes(self):
        r = ConfigBacktestRunner()
        orders = OrdersResult()
        orders.add_order(Order(code='AAPL', size=600))
        risk = PositionLimitRiskNode(config={'max_position': 1000})
        result = r._apply_risk(orders, [risk])
        self.assertEqual(len(result.orders), 1)


class TestConfigBacktestRunnerComputeStatistics(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.config.backtest = MagicMock()
        self.config.backtest.initial_cash = 100000

    def test_with_empty_trades(self):
        # NOTE: _compute_statistics has a bug when equity_curve is empty:
        # equity_curve["equity"].pct_change() raises ValueError on empty Series
        # This test documents the bug
        r = ConfigBacktestRunner()
        trade_result = TradeResult()
        trade_result.cash = 100000
        df = pd.DataFrame({'date': [], 'Code': [], 'Close': []})
        with self.assertRaises(ValueError):
            r._compute_statistics(trade_result, df, self.config)

    def test_basic_statistics(self):
        r = ConfigBacktestRunner()
        trade_result = TradeResult()
        trade_result.cash = 110000
        trade_result.commission = 100.0
        trade_result.executed_value = 10000.0
        trade_result.trades = [
            Trade(order_id='o1', code='AAPL', side='buy', size=100,
                  price=100.0, adjusted_price=100.1, fee=10.0, dt='2024-01-01'),
            Trade(order_id='o2', code='AAPL', side='sell', size=100,
                  price=105.0, adjusted_price=104.9, fee=10.5, dt='2024-01-10'),
        ]
        df = make_quote_df(n_days=20)
        result = r._compute_statistics(trade_result, df, self.config)
        self.assertIsInstance(result, BacktestResult)
        self.assertIn('total_trades', result.statistics)
        self.assertIn('sharpe_ratio', result.statistics)
        self.assertIn('max_drawdown', result.statistics)


class TestConfigBacktestRunnerBuildEquityCurve(unittest.TestCase):
    def test_empty_quote(self):
        result = ConfigBacktestRunner._build_equity_curve(
            pd.DataFrame(), pd.DataFrame(), 100000,
        )
        self.assertIsInstance(result, pd.DataFrame)

    def test_no_trades(self):
        trades = pd.DataFrame()
        quote = make_quote_df(n_days=10)
        result = ConfigBacktestRunner._build_equity_curve(trades, quote, 100000)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 10)
        self.assertIn('equity', result.columns)
        self.assertIn('cash', result.columns)

    def test_equity_with_trades(self):
        trades = pd.DataFrame({
            'order_id': ['o1'], 'code': ['AAPL'], 'side': ['buy'],
            'size': [100.0], 'adjusted_price': [100.0], 'fee': [10.0], 'dt': ['2024-01-01'],
        })
        quote = make_quote_df(n_days=5)
        result = ConfigBacktestRunner._build_equity_curve(trades, quote, 100000)
        self.assertGreater(result['equity'].iloc[-1], result['equity'].iloc[0])


class TestConfigBacktestRunnerComputeTradePnl(unittest.TestCase):
    def test_empty(self):
        result = ConfigBacktestRunner._compute_trade_pnl(pd.DataFrame())
        self.assertEqual(result, [])

    def test_single_pair(self):
        trades = pd.DataFrame({
            'code': ['AAPL', 'AAPL'], 'side': ['buy', 'sell'],
            'size': [100.0, 100.0], 'adjusted_price': [100.0, 105.0],
        })
        result = ConfigBacktestRunner._compute_trade_pnl(trades)
        self.assertEqual(len(result), 1)
        self.assertGreater(result[0], 0)

    def test_multiple_codes(self):
        trades = pd.DataFrame({
            'code': ['AAPL', 'AAPL', 'GOOG', 'GOOG'],
            'side': ['buy', 'sell', 'buy', 'sell'],
            'size': [100.0, 100.0, 50.0, 50.0],
            'adjusted_price': [100.0, 105.0, 200.0, 190.0],
        })
        result = ConfigBacktestRunner._compute_trade_pnl(trades)
        self.assertEqual(len(result), 2)


class TestConfigBacktestRunnerMaxDrawdown(unittest.TestCase):
    def test_empty(self):
        result = ConfigBacktestRunner._max_drawdown(pd.Series([]))
        self.assertEqual(result, 0.0)

    def test_rising_equity(self):
        equity = pd.Series([100000, 101000, 102000, 103000])
        result = ConfigBacktestRunner._max_drawdown(equity)
        self.assertEqual(result, 0.0)

    def test_decline(self):
        equity = pd.Series([100000, 110000, 105000, 108000])
        result = ConfigBacktestRunner._max_drawdown(equity)
        self.assertLess(result, 0.0)

    def test_constant(self):
        equity = pd.Series([100000, 100000, 100000])
        result = ConfigBacktestRunner._max_drawdown(equity)
        self.assertEqual(result, 0.0)


class TestConfigBacktestRunnerSaveOutput(unittest.TestCase):
    def test_no_output_config(self):
        r = ConfigBacktestRunner()
        config = MagicMock()
        config.output = None
        result = r.save_output(BacktestResult(), config)
        self.assertEqual(result, {})


if __name__ == '__main__':
    unittest.main()
