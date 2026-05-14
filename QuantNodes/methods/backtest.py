# coding=utf-8
"""
Backtest Method

run_backtest(config) -> BacktestResult

External agents use this method to run backtests via API.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BacktestResult:
    status: str
    summary: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    security_status: str = "unknown"
    nodes: Dict[str, Any] = field(default_factory=dict)


CODE_BLOCK_PATTERN = re.compile(r'```(?:python)?\s*(.*?)```', re.DOTALL)


def run_backtest(
    pipeline_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_cash: float = 100000.0,
    commission: float = 0.001,
    **kwargs
) -> BacktestResult:
    """Run a backtest with the given pipeline code.

    Args:
        pipeline_code: Strategy pipeline code that creates strategy, broker, and quote_data
        start_date: Backtest start date (YYYY-MM-DD)
        end_date: Backtest end date (YYYY-MM-DD)
        initial_cash: Initial capital
        commission: Commission rate

    Returns:
        BacktestResult with summary, trades, and statistics
    """
    result = BacktestResult(
        status="success",
        config={
            "start_date": start_date,
            "end_date": end_date,
            "initial_cash": initial_cash,
            "commission": commission,
        }
    )

    try:
        from QuantNodes.ai.sandbox import CodeSandbox
        from QuantNodes.backtest.strategy_node import StrategyNode
        from QuantNodes.backtest.broker_node import SimulatedBrokerNode
        from QuantNodes.backtest.risk_node import RiskNode

        sandbox = CodeSandbox()
        extracted_code = _extract_code(pipeline_code)

        if not extracted_code:
            result.status = "error"
            result.errors = ["No valid code found in pipeline_code"]
            return result

        validation = sandbox.validate(extracted_code)
        if not validation.is_safe:
            result.status = "error"
            result.errors = validation.errors
            result.security_status = "unsafe"
            return result

        result.security_status = "safe"

        import pandas as pd
        import numpy as np
        context = {
            "pd": pd,
            "np": np,
            "QuantNodes": __import__("QuantNodes"),
        }

        namespace = sandbox.validate_and_execute(extracted_code, context)

        strategy = None
        broker = None
        risk_nodes = []
        quote_data = None

        for name, obj in namespace.items():
            if isinstance(obj, StrategyNode) and not isinstance(obj, RiskNode):
                strategy = obj
            elif isinstance(obj, SimulatedBrokerNode):
                broker = obj
            elif isinstance(obj, RiskNode):
                risk_nodes.append(obj)

        quote_data = namespace.get("quote_data") or namespace.get("data")

        if strategy is None:
            result.status = "error"
            result.errors = [
                "No StrategyNode found. Create a strategy variable in your code, e.g.:\n"
                "from QuantNodes.backtest.strategy_node import MAStrategyNode\n"
                "strategy = MAStrategyNode(config={'short_window': 5, 'long_window': 20})"
            ]
            return result

        if quote_data is None:
            result.status = "error"
            result.errors = [
                "No quote_data found. Create a 'quote_data' DataFrame in your code, e.g.:\n"
                "quote_data = pd.read_csv('data.csv')"
            ]
            return result

        if broker is None:
            broker = SimulatedBrokerNode(config={
                "cash": initial_cash,
                "commission": commission,
            })

        if start_date and end_date and "date" in quote_data.columns:
            quote_data = quote_data[
                (quote_data["date"] >= start_date) &
                (quote_data["date"] <= end_date)
            ]

        orders_result = strategy.execute(quote_data)

        filtered_orders = orders_result
        for risk in risk_nodes:
            risk_result = risk.execute((filtered_orders, {}))
            from QuantNodes.backtest.strategy_node import OrdersResult
            filtered = OrdersResult()
            filtered.orders = risk_result.passed_orders
            filtered.signals = orders_result.signals
            filtered_orders = filtered

        trade_result = broker.execute((filtered_orders, quote_data))

        result.summary = {
            "total_trades": len(trade_result.trades) if hasattr(trade_result, "trades") else 0,
            "final_cash": trade_result.cash if hasattr(trade_result, "cash") else initial_cash,
            "total_commission": trade_result.commission if hasattr(trade_result, "commission") else 0,
            "strategy": strategy.__class__.__name__,
            "broker": broker.__class__.__name__,
            "risk_nodes": [r.__class__.__name__ for r in risk_nodes],
            "data_rows": len(quote_data),
        }

        result.nodes = {
            "strategy": strategy.__class__.__name__,
            "broker": broker.__class__.__name__,
            "risk_nodes": [r.__class__.__name__ for r in risk_nodes],
        }

    except Exception as e:
        result.status = "error"
        result.errors = [str(e)]

    return result


def _extract_code(code: str) -> str:
    """Extract code from markdown code blocks if present."""
    match = CODE_BLOCK_PATTERN.search(code)
    if match:
        return match.group(1).strip()
    return code.strip()