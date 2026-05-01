# coding=utf-8
"""
ExecutionBrokerNode + ConfigStrategyNode 优化后单元测试

覆盖:
- ExecutionBrokerNode 向量化执行正确性
- ExecutionBrokerNode 空订单
- ExecutionBrokerNode 价格查找 fallback
- ConfigStrategyNode 优化后信号生成
- ConfigStrategyNode 空信号
"""

import pytest
import pandas as pd
import numpy as np

from QuantNodes.backtest.broker_node import ExecutionBrokerNode, TradeResult
from QuantNodes.backtest.config_strategy import ConfigStrategyNode
from QuantNodes.backtest.strategy_node import Order, OrdersResult, Signal


class TestExecutionBrokerVectorized:
    """ExecutionBrokerNode 向量化执行测试"""

    def _make_quote(self, codes, dates, prices):
        """生成行情数据"""
        rows = []
        for code in codes:
            for d, p in zip(dates, prices):
                rows.append({"Code": code, "date": d, "Close": p, "Open": p * 0.99})
        return pd.DataFrame(rows)

    def _make_orders(self, orders_list):
        """生成订单列表"""
        result = OrdersResult()
        for code, size, date in orders_list:
            result.orders.append(Order(code=code, size=size, create_date=date))
        return result

    def test_basic_buy_sell(self):
        quote = self._make_quote(
            ["A"], ["2023-01-01", "2023-01-02"], [10.0, 11.0]
        )
        orders = self._make_orders([
            ("A", 100, "2023-01-01"),
            ("A", -100, "2023-01-02"),
        ])

        broker = ExecutionBrokerNode(config={
            "cash": 100000, "commission": 0.001, "slippage": 0.001,
        })
        result = broker.execute((orders, quote))

        assert len(result.trades) == 2
        assert result.trades[0].side == "buy"
        assert result.trades[1].side == "sell"

    def test_empty_orders(self):
        quote = self._make_quote(["A"], ["2023-01-01"], [10.0])
        orders = OrdersResult()

        broker = ExecutionBrokerNode(config={"cash": 100000})
        result = broker.execute((orders, quote))
        assert len(result.trades) == 0

    def test_empty_quote(self):
        orders = self._make_orders([("A", 100, "2023-01-01")])
        broker = ExecutionBrokerNode(config={"cash": 100000})
        result = broker.execute((orders, pd.DataFrame()))
        assert len(result.trades) == 0

    def test_price_fallback(self):
        """订单日期在行情中不存在时 fallback 到首条记录"""
        quote = self._make_quote(["A"], ["2023-01-01"], [10.0])
        orders = self._make_orders([("A", 100, "2023-01-05")])  # 日期不存在

        broker = ExecutionBrokerNode(config={
            "cash": 100000, "commission": 0.001, "slippage": 0.001,
        })
        result = broker.execute((orders, quote))
        assert len(result.trades) == 1
        assert result.trades[0].price == 9.9  # fallback to first row Open price

    def test_multiple_codes(self):
        quote = self._make_quote(
            ["A", "B"], ["2023-01-01", "2023-01-02"], [10.0, 20.0]
        )
        orders = self._make_orders([
            ("A", 100, "2023-01-01"),
            ("B", 50, "2023-01-01"),
            ("A", -100, "2023-01-02"),
        ])

        broker = ExecutionBrokerNode(config={
            "cash": 100000, "commission": 0.001, "slippage": 0.001,
        })
        result = broker.execute((orders, quote))
        assert len(result.trades) == 3

    def test_cash_insufficient(self):
        """现金不足时买入不执行"""
        quote = self._make_quote(["A"], ["2023-01-01"], [100000.0])
        orders = self._make_orders([("A", 100, "2023-01-01")])  # cost > cash

        broker = ExecutionBrokerNode(config={
            "cash": 100, "commission": 0.001, "slippage": 0.001,
        })
        result = broker.execute((orders, quote))
        assert len(result.trades) == 0

    def test_sell_always_executes(self):
        """卖出总是执行（允许做空）"""
        quote = self._make_quote(["A"], ["2023-01-01"], [10.0])
        orders = self._make_orders([("A", -100, "2023-01-01")])

        broker = ExecutionBrokerNode(config={"cash": 0})
        result = broker.execute((orders, quote))
        assert len(result.trades) == 1

    def test_slippage_and_commission(self):
        """滑点和手续费计算"""
        quote = self._make_quote(["A"], ["2023-01-01"], [10.0])
        orders = self._make_orders([("A", 100, "2023-01-01")])

        broker = ExecutionBrokerNode(config={
            "cash": 100000, "commission": 0.001, "slippage": 0.001,
        })
        result = broker.execute((orders, quote))

        trade = result.trades[0]
        # price should be adjusted for slippage and commission
        assert trade.adjusted_price != trade.price
        assert trade.fee > 0

    def test_performance_large_orders(self):
        """大量订单性能测试"""
        codes = [f"stock_{i}" for i in range(100)]
        dates = [f"2023-01-{d:02d}" for d in range(1, 22)]
        quote = self._make_quote(codes, dates, [10.0] * len(dates))

        orders_list = []
        for code in codes:
            for d in dates[:10]:
                orders_list.append((code, 100, d))
            for d in dates[10:]:
                orders_list.append((code, -100, d))
        orders = self._make_orders(orders_list)

        import time
        broker = ExecutionBrokerNode(config={
            "cash": 100000000, "commission": 0.001, "slippage": 0.001,
        })
        t0 = time.time()
        result = broker.execute((orders, quote))
        elapsed = time.time() - t0

        assert len(result.trades) > 0
        # 2000 orders should complete in < 1s
        assert elapsed < 1.0


class TestConfigStrategyNodeOptimized:
    """ConfigStrategyNode 优化后测试"""

    def test_buy_signals(self):
        df = pd.DataFrame({
            "Code": ["A", "A", "A"],
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "Close": [10.0, 11.0, 12.0],
            "signal": [1, 1, 0],
        })
        strategy = ConfigStrategyNode(signal_col="signal")
        result = strategy.execute(df)
        assert len(result.signals) == 2
        assert all(s.signal_type == "buy" for s in result.signals)

    def test_sell_signals(self):
        df = pd.DataFrame({
            "Code": ["A", "A"],
            "date": ["2023-01-01", "2023-01-02"],
            "Close": [10.0, 11.0],
            "signal": [-1, -1],
        })
        strategy = ConfigStrategyNode(signal_col="signal")
        result = strategy.execute(df)
        assert len(result.signals) == 2
        assert all(s.signal_type == "sell" for s in result.signals)

    def test_empty_signals(self):
        df = pd.DataFrame({
            "Code": ["A", "A"],
            "date": ["2023-01-01", "2023-01-02"],
            "Close": [10.0, 11.0],
            "signal": [0, 0],
        })
        strategy = ConfigStrategyNode(signal_col="signal")
        result = strategy.execute(df)
        assert len(result.signals) == 0

    def test_mixed_signals(self):
        df = pd.DataFrame({
            "Code": ["A"] * 5,
            "date": [f"2023-01-0{i}" for i in range(1, 6)],
            "Close": [10.0, 11.0, 12.0, 13.0, 14.0],
            "signal": [1, 0, -1, 0, 1],
        })
        strategy = ConfigStrategyNode(signal_col="signal")
        result = strategy.execute(df)
        assert len(result.signals) == 3
        assert result.signals[0].signal_type == "buy"
        assert result.signals[1].signal_type == "sell"
        assert result.signals[2].signal_type == "buy"

    def test_code_lowercase(self):
        df = pd.DataFrame({
            "code": ["A", "A"],
            "date": ["2023-01-01", "2023-01-02"],
            "close": [10.0, 11.0],
            "signal": [1, -1],
        })
        strategy = ConfigStrategyNode(signal_col="signal")
        result = strategy.execute(df)
        assert len(result.signals) == 2

    def test_performance_large_dataframe(self):
        """大量数据性能测试"""
        n = 100000
        df = pd.DataFrame({
            "Code": [f"stock_{i % 1000}" for i in range(n)],
            "date": [f"2023-01-{(i % 28) + 1:02d}" for i in range(n)],
            "Close": np.random.uniform(5, 50, n),
            "signal": np.random.choice([0, 0, 0, 1, -1], n),
        })
        strategy = ConfigStrategyNode(signal_col="signal")

        import time
        t0 = time.time()
        result = strategy.execute(df)
        elapsed = time.time() - t0

        assert len(result.signals) > 0
        # 100K rows should complete in < 1s
        assert elapsed < 1.0
