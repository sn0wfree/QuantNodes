# coding=utf-8
"""
BacktestNode - 回测引擎节点基类

提供回测引擎节点的基础架构，继承自 BaseNode。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from QuantNodes.core.node import BaseNode, NodeState


@dataclass
class BacktestResult:
    """回测结果容器"""
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    orders: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    statistics: Dict[str, Any] = field(default_factory=dict)
    final_cash: float = 0.0
    final_positions: Dict[str, float] = field(default_factory=dict)
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0


class BacktestNode(BaseNode[pd.DataFrame, BacktestResult], ABC):
    """
    回测引擎节点基类

    提供统一的回测执行接口。

    Subclasses must implement:
        _run_backtest(): 执行具体回测逻辑

    Examples:
        >>> backtest = BacktestNode(config={
        ...     'cash': 100000,
        ...     'commission': 0.001,
        ...     'margin': 0.1
        ... })
        >>> result = backtest.execute(data)
    """

    _enable_validation: bool = True
    _enable_stats: bool = True

    def __init__(self, name: str = None, config: Dict[str, Any] = None, **kwargs):
        default_name = f"{self.__class__.__name__}"
        super().__init__(name=name or default_name, config=config, **kwargs)

        self._result: Optional[BacktestResult] = None
        self._cash: float = self.config.get('cash', 100000)
        self._commission: float = self.config.get('commission', 0.001)
        self._margin: float = self.config.get('margin', 0.1)
        self._trade_on_close: bool = self.config.get('trade_on_close', False)
        self._hedging: bool = self.config.get('hedging', False)

    @abstractmethod
    def _run_backtest(
        self,
        quote_data: pd.DataFrame,
        signals: pd.DataFrame,
        **kwargs
    ) -> BacktestResult:
        """
        执行具体回测逻辑

        Args:
            quote_data: 行情数据 DataFrame
            signals: 交易信号 DataFrame
            **kwargs: 额外执行参数

        Returns:
            回测结果 BacktestResult
        """
        pass

    def _execute(
        self,
        input_data: Union[pd.DataFrame, tuple, None] = None,
        **kwargs
    ) -> BacktestResult:
        """
        执行回测

        Args:
            input_data: 输入数据，可以是：
                - pd.DataFrame: 行情数据
                - tuple: (quote_data, signals) 元组
                - None: 使用 config 中的默认数据
            **kwargs: 额外执行参数

        Returns:
            BacktestResult 回测结果
        """
        if input_data is None:
            quote_data = kwargs.get('quote_data')
            signals = kwargs.get('signals')
        elif isinstance(input_data, pd.DataFrame):
            quote_data = input_data
            signals = kwargs.get('signals')
        elif isinstance(input_data, tuple) and len(input_data) == 2:
            quote_data, signals = input_data
        else:
            raise ValueError(
                f"input_data must be DataFrame or (quote_data, signals) tuple, "
                f"got {type(input_data).__name__}"
            )

        if quote_data is None:
            raise ValueError("quote_data is required")

        self._result = self._run_backtest(quote_data, signals, **kwargs)
        return self._result

    def _validate_input(self, input_data: Any) -> None:
        """验证输入数据"""
        if input_data is None:
            return

        if isinstance(input_data, tuple):
            quote_data, signals = input_data
            if not isinstance(quote_data, pd.DataFrame):
                raise ValueError(f"quote_data must be DataFrame, got {type(quote_data).__name__}")
        elif isinstance(input_data, pd.DataFrame):
            pass
        else:
            raise ValueError(f"input_data must be DataFrame or tuple, got {type(input_data).__name__}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取回测统计信息"""
        if self._result is None:
            return {}
        return self._result.statistics

    def get_equity_curve(self) -> pd.DataFrame:
        """获取权益曲线"""
        if self._result is None:
            return pd.DataFrame()
        return self._result.equity_curve

    def reset(self) -> None:
        """重置回测状态"""
        super().reset()
        self._result = None


class BacktestPipeline:
    """
    回测管道

    将多个回测节点组合在一起执行。
    """

    def __init__(self, nodes: List[BacktestNode]):
        self.nodes = nodes

    def execute(
        self,
        input_data: Any = None,
        **kwargs
    ) -> List[BacktestResult]:
        """执行所有回测节点"""
        results = []
        current_data = input_data

        for node in self.nodes:
            result = node.execute(current_data, **kwargs)
            results.append(result)
            current_data = result

        return results

    def __rshift__(self, other: BacktestNode) -> 'BacktestPipeline':
        """重载 >> 运算符"""
        if isinstance(other, BacktestPipeline):
            return BacktestPipeline(self.nodes + other.nodes)
        return BacktestPipeline(self.nodes + [other])