# coding=utf-8
"""
Backtest module for QuantNodes.

BaseNode-integrated backtest components.
"""

from QuantNodes.backtest.backtest_node import (
    BacktestNode,
    BacktestResult,
    BacktestPipeline,
)
from QuantNodes.backtest.strategy_node import (
    StrategyNode,
    Order,
    Signal,  # SignalV2 alias for TradeSignal (backward compat)
    TradeSignal,
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
from QuantNodes.backtest.config_strategy import ConfigStrategyNode
from QuantNodes.backtest.config_runner import ConfigBacktestRunner

__all__ = [
    'BacktestNode',
    'BacktestResult',
    'BacktestPipeline',
    'StrategyNode',
    'Order',
    'Signal',  # SignalV2 backward-compat alias
    'TradeSignal',
    'OrdersResult',
    'MAStrategyNode',
    'MomentumStrategyNode',
    'BrokerNode',
    'Trade',
    'TradeResult',
    'SimulatedBrokerNode',
    'ExecutionBrokerNode',
    'RiskNode',
    'RiskCheck',
    'RiskResult',
    'PositionLimitRiskNode',
    'StopLossRiskNode',
    'CashRiskNode',
    'CompositeRiskNode',
    'ConfigStrategyNode',
    'ConfigBacktestRunner',
]
