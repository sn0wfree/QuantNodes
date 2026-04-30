# coding=utf-8
"""
回测运行工具

封装 QuantNodes 回测引擎，执行真实回测。
"""

from typing import Any, Dict, List, Optional
import re

from QuantNodes.agent.tools.base import Tool


class BacktestTool(Tool):
    """回测运行工具

    通过 CodeSandbox 安全执行策略代码，提取节点，
    然后运行 Strategy→Risk→Broker 回测流程。
    
    pipeline_code 示例:
        import pandas as pd
        from QuantNodes.backtest.strategy_node import MAStrategyNode
        from QuantNodes.backtest.broker_node import SimulatedBrokerNode
        
        strategy = MAStrategyNode(config={'short_window': 5, 'long_window': 20})
        broker = SimulatedBrokerNode(config={'cash': 100000, 'commission': 0.001})
        quote_data = pd.read_csv('data.csv')
    """

    CODE_BLOCK_PATTERN = re.compile(r'```(?:python)?\s*(.*?)```', re.DOTALL)

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        return "backtest"

    @property
    def description(self) -> str:
        return (
            "运行策略回测，返回回测结果（交易次数、最终资金、手续费等）。"
            "pipeline_code 中需创建 strategy、broker 变量和 quote_data DataFrame。"
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pipeline_code": {
                    "type": "string",
                    "description": "策略Pipeline代码，需创建 strategy、broker 变量和 quote_data DataFrame"
                },
                "start_date": {
                    "type": "string",
                    "description": "回测开始日期，格式YYYY-MM-DD"
                },
                "end_date": {
                    "type": "string",
                    "description": "回测结束日期，格式YYYY-MM-DD"
                },
                "initial_cash": {
                    "type": "number",
                    "description": "初始资金",
                    "default": 100000
                },
                "commission": {
                    "type": "number",
                    "description": "手续费率",
                    "default": 0.001
                }
            },
            "required": ["pipeline_code"]
        }

    @property
    def read_only(self) -> bool:
        return False

    @property
    def concurrency_safe(self) -> bool:
        return False

    async def execute(
        self,
        pipeline_code: str,
        start_date: str = None,
        end_date: str = None,
        initial_cash: float = 100000,
        commission: float = 0.001,
        **kwargs
    ) -> Dict[str, Any]:
        result = {
            "status": "success",
            "summary": {},
            "config": {
                "start_date": start_date,
                "end_date": end_date,
                "initial_cash": initial_cash,
                "commission": commission,
            }
        }

        try:
            from QuantNodes.ai.sandbox import CodeSandbox
            from QuantNodes.backtest.strategy_node import StrategyNode
            from QuantNodes.backtest.broker_node import SimulatedBrokerNode
            from QuantNodes.backtest.risk_node import RiskNode

            sandbox = CodeSandbox()
            extracted_code = self._extract_code(pipeline_code)

            if not extracted_code:
                result["status"] = "error"
                result["errors"] = ["No valid code found in pipeline_code"]
                return result

            validation = sandbox.validate(extracted_code)
            if not validation.is_safe:
                result["status"] = "error"
                result["errors"] = validation.errors
                result["security_status"] = "unsafe"
                return result

            result["security_status"] = "safe"

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
                result["status"] = "error"
                result["errors"] = [
                    "No StrategyNode found. Create a strategy variable in your code, e.g.:\n"
                    "from QuantNodes.backtest.strategy_node import MAStrategyNode\n"
                    "strategy = MAStrategyNode(config={'short_window': 5, 'long_window': 20})"
                ]
                return result

            if quote_data is None:
                result["status"] = "error"
                result["errors"] = [
                    "No quote_data found. Create a 'quote_data' DataFrame in your code, e.g.:\n"
                    "quote_data = pd.read_csv('data.csv')"
                ]
                return result

            if broker is None:
                from QuantNodes.backtest.broker_node import SimulatedBrokerNode
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

            result["summary"] = {
                "total_trades": len(trade_result.trades) if hasattr(trade_result, "trades") else 0,
                "final_cash": trade_result.cash if hasattr(trade_result, "cash") else initial_cash,
                "total_commission": trade_result.commission if hasattr(trade_result, "commission") else 0,
                "strategy": strategy.__class__.__name__,
                "broker": broker.__class__.__name__,
                "risk_nodes": [r.__class__.__name__ for r in risk_nodes],
                "data_rows": len(quote_data),
            }

            result["nodes"] = {
                "strategy": strategy.__class__.__name__,
                "broker": broker.__class__.__name__,
                "risk_nodes": [r.__class__.__name__ for r in risk_nodes],
            }

        except Exception as e:
            result["status"] = "error"
            result["errors"] = [str(e)]

        return result

    def _extract_code(self, code: str) -> str:
        match = self.CODE_BLOCK_PATTERN.search(code)
        if match:
            return match.group(1).strip()
        return code.strip()
