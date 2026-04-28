# coding=utf-8
"""
Backtest module for QuantNodes.

Note: Some submodules have Python 3.10 compatibility issues 
(Iterable import from collections instead of collections.abc).
"""

from QuantNodes.backtest.backtest_node import (
    BacktestNode,
    BacktestResult,
    BacktestPipeline,
)
from QuantNodes.backtest.strategy_node import (
    StrategyNode,
    Order,
    Signal,
    OrdersResult,
    MAStrategyNode,
    MomentumStrategyNode,
)
from QuantNodes.backtest.broker_node import (
    BrokerNode,
    Trade,
    TradeResult,
    SimulatedBrokerNode,
    ExecutionBrokerNode,
)
from QuantNodes.backtest.risk_node import (
    RiskNode,
    RiskCheck,
    RiskResult,
    PositionLimitRiskNode,
    StopLossRiskNode,
    CashRiskNode,
    CompositeRiskNode,
)

__all__ = [
    # BacktestNode (BaseNode integration)
    'BacktestNode',
    'BacktestResult',
    'BacktestPipeline',

    # StrategyNode
    'StrategyNode',
    'Order',
    'Signal',
    'OrdersResult',
    'MAStrategyNode',
    'MomentumStrategyNode',

    # BrokerNode
    'BrokerNode',
    'Trade',
    'TradeResult',
    'SimulatedBrokerNode',
    'ExecutionBrokerNode',

    # RiskNode
    'RiskNode',
    'RiskCheck',
    'RiskResult',
    'PositionLimitRiskNode',
    'StopLossRiskNode',
    'CashRiskNode',
    'CompositeRiskNode',

    # Legacy exports (from original backtest module)
    'ScriptsBackTest',
    'Broker',
    'QuoteData',
    'Positions',
    'Order',
    'Orders',
    'Trade',
    'Indicators',
    'Statistics',
]
