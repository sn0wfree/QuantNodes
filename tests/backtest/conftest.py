# -*- coding: utf-8 -*-
"""Backtest 模块测试 fixtures"""
import numpy as np
import pandas as pd
import polars as pl
import pytest


@pytest.fixture
def backtest_quote_data():
    """回测用行情数据 DataFrame

    包含 30 天 AAPL 股票数据，日线 OHLCV。
    """
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    return pd.DataFrame({
        'date': [str(d.date()) for d in dates],
        'Code': ['AAPL'] * 30,
        'Open': [100 + i * 0.5 + np.random.randn() * 0.5 for i in range(30)],
        'High': [105 + i * 0.5 + np.random.randn() * 0.5 for i in range(30)],
        'Low': [95 + i * 0.5 + np.random.randn() * 0.5 for i in range(30)],
        'Close': [100 + i * 0.5 + np.random.randn() * 0.5 for i in range(30)],
        'Volume': [1_000_000] * 30,
    })


@pytest.fixture
def backtest_quote_data_multi_stock():
    """多只股票的回测行情数据"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    stocks = ['AAPL', 'GOOG', 'MSFT']
    records = []
    for stock in stocks:
        for i, d in enumerate(dates):
            records.append({
                'date': str(d.date()),
                'Code': stock,
                'Open': 100 + i * 0.5 + np.random.randn() * 0.5,
                'High': 105 + i * 0.5 + np.random.randn() * 0.5,
                'Low': 95 + i * 0.5 + np.random.randn() * 0.5,
                'Close': 100 + i * 0.5 + np.random.randn() * 0.5,
                'Volume': 1_000_000,
            })
    return pd.DataFrame(records)


@pytest.fixture
def backtest_trade_history():
    """历史交易记录 DataFrame"""
    return pd.DataFrame({
        'order_id': ['order_1', 'order_2', 'order_3', 'order_4'],
        'code': ['AAPL', 'AAPL', 'GOOG', 'GOOG'],
        'side': ['buy', 'sell', 'buy', 'sell'],
        'size': [100.0, 50.0, 200.0, 100.0],
        'price': [100.0, 105.0, 200.0, 210.0],
        'adjusted_price': [100.1, 105.1, 200.2, 210.2],
        'fee': [10.01, 5.255, 40.04, 21.02],
        'dt': ['2024-01-01', '2024-01-10', '2024-01-02', '2024-01-12'],
        'status': ['completed'] * 4,
    })


@pytest.fixture
def backtest_equity_curve_data():
    """权益曲线 DataFrame"""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'equity': [100000, 101000, 102500, 101800, 103200, 104500, 103800, 105000, 106200, 108000],
        'cash': [80000, 81000, 82000, 81500, 83000, 84000, 83500, 84500, 85500, 87000],
        'position_value': [20000, 20000, 20500, 20300, 20200, 20500, 20300, 20500, 20700, 21000],
    })


@pytest.fixture
def sample_orders():
    """样例订单列表"""
    from QuantNodes.backtest.strategy_node import Order
    return [
        Order(code='AAPL', size=100, limit_price=100.0, create_date='2024-01-01', order_id='order_1'),
        Order(code='AAPL', size=-50, limit_price=105.0, create_date='2024-01-10', order_id='order_2'),
        Order(code='GOOG', size=200, limit_price=200.0, create_date='2024-01-02', order_id='order_3'),
        Order(code='GOOG', size=-100, limit_price=210.0, create_date='2024-01-12', order_id='order_4'),
    ]


@pytest.fixture
def polars_backtest_data():
    """Polars LazyFrame 回测数据"""
    return pl.LazyFrame({
        'date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
        'code': ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL'],
        'close': [100.0, 101.0, 99.0, 102.0, 103.0],
        'volume': [1_000_000, 1_100_000, 900_000, 1_200_000, 1_300_000],
    })
