# coding=utf-8
"""
配置驱动的回测运行器

直接从 StrategyConfig + Polars 数据执行完整回测，
不经过代码生成，直接调用 backtest/ 引擎。
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import polars as pl

from QuantNodes.agent.config.types import StrategyConfig
from QuantNodes.agent.config.executor import ConfigExecutor
from QuantNodes.backtest.config_strategy import ConfigStrategyNode
from QuantNodes.backtest.backtest_node import BacktestResult
from QuantNodes.backtest.strategy_node import OrdersResult
from QuantNodes.backtest.broker_node import ExecutionBrokerNode
from QuantNodes.backtest.risk_node import PositionLimitRiskNode, RiskNode


class ConfigBacktestRunner:
    """从 StrategyConfig + Polars 数据执行完整回测"""

    def run(
        self, config: StrategyConfig, data: pl.LazyFrame
    ) -> BacktestResult:
        """执行回测

        Args:
            config: 策略配置
            data: Polars LazyFrame 数据

        Returns:
            BacktestResult 包含交易、统计等信息
        """
        if config.backtest is None:
            return BacktestResult()

        # 1. 因子计算 + 信号生成
        executor = ConfigExecutor()
        result = executor.run_backtest(config, data)

        if result.status == "error":
            return BacktestResult()

        # 2. Polars → Pandas
        df = result.data.collect().to_pandas()

        # 3. 列名标准化
        df = self._normalize_columns(df)

        # 4. 确保 signal 列存在
        if "signal" not in df.columns:
            return BacktestResult()

        # 5. 策略 → 风控 → 经纪商
        strategy = ConfigStrategyNode(signal_col="signal")
        orders_result = strategy.execute(df)

        risk_nodes = self._build_risk_nodes(config)
        filtered = self._apply_risk(orders_result, risk_nodes)

        broker = self._build_broker(config)
        trade_result = broker.execute((filtered, df))

        # 6. 计算绩效统计
        return self._compute_statistics(trade_result, df, config)

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一列名大小写"""
        rename_map = {}

        # Code/code → Code
        if "Code" not in df.columns and "code" in df.columns:
            rename_map["code"] = "Code"

        # close → Close
        if "Close" not in df.columns and "close" in df.columns:
            rename_map["close"] = "Close"

        # open → Open
        if "Open" not in df.columns and "open" in df.columns:
            rename_map["open"] = "Open"

        if rename_map:
            df = df.rename(columns=rename_map)

        # 确保 Open 列存在（fallback 到 Close）
        if "Open" not in df.columns and "Close" in df.columns:
            df["Open"] = df["Close"]

        return df

    def _build_risk_nodes(self, config: StrategyConfig) -> List[RiskNode]:
        """从 config 构建风控节点"""
        nodes = []
        bt = config.backtest
        if bt and bt.positions:
            max_pos = bt.positions.get("max_positions")
            if max_pos is not None:
                nodes.append(PositionLimitRiskNode(
                    config={"max_position": max_pos}
                ))
        return nodes

    def _build_broker(self, config: StrategyConfig) -> ExecutionBrokerNode:
        """从 config 构建经纪商"""
        bt = config.backtest
        return ExecutionBrokerNode(config={
            "cash": bt.initial_cash if bt else 1000000,
            "commission": bt.commission if bt else 0.001,
            "slippage": bt.slippage if bt else 0.001,
        })

    def _apply_risk(
        self, orders_result: OrdersResult, risk_nodes: List[RiskNode]
    ) -> OrdersResult:
        """应用风控过滤"""
        current_orders = orders_result
        for node in risk_nodes:
            risk_result = node.execute(current_orders)
            new_orders = OrdersResult()
            new_orders.orders = risk_result.passed_orders
            new_orders.signals = orders_result.signals
            current_orders = new_orders
        return current_orders

    def _compute_statistics(
        self, trade_result, df: pd.DataFrame, config: StrategyConfig
    ) -> BacktestResult:
        """计算绩效统计"""
        bt = config.backtest
        initial_cash = bt.initial_cash if bt else 1000000

        trades_df = trade_result.to_dataframe()

        # 计算总收益率
        total_return = (trade_result.cash - initial_cash) / initial_cash

        # 计算胜率
        win_rate = 0.0
        if len(trades_df) > 0:
            # 按 code 分组计算盈亏（简化版）
            trade_pnls = []
            for code in trades_df["code"].unique():
                code_trades = trades_df[trades_df["code"] == code]
                buy_cost = code_trades[code_trades["side"] == "buy"]["adjusted_price"].sum()
                sell_revenue = code_trades[code_trades["side"] == "sell"]["adjusted_price"].sum()
                pnl = sell_revenue - buy_cost
                trade_pnls.append(pnl)
            if trade_pnls:
                win_rate = sum(1 for p in trade_pnls if p > 0) / len(trade_pnls)

        return BacktestResult(
            trades=trades_df,
            orders=pd.DataFrame(),
            equity_curve=pd.DataFrame(),
            statistics={
                "total_trades": len(trade_result.trades),
                "total_commission": trade_result.commission,
                "executed_value": trade_result.executed_value,
            },
            final_cash=trade_result.cash,
            total_return=total_return,
            win_rate=win_rate,
        )
