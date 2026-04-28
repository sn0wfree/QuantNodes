# coding=utf-8
"""
TimeFactorNode - 时间序列运算因子节点

对时间序列数据进行滚动窗口或扩展窗口运算。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Callable

import pandas as pd
import numpy as np

from QuantNodes.factor_node.factor_node import FactorNode


class TimeFactorNode(FactorNode):
    """
    时间序列运算因子节点

    对时间序列数据进行窗口运算，如移动平均、滚动标准差等。

    Examples:
        >>> # 移动平均
        >>> factor = TimeFactorNode(
        ...     window=20,
        ...     operation="mean",
        ...     column="close"
        ... )
        >>> result = factor.execute(df)
        >>>
        >>> # 滚动相关系数
        >>> factor = TimeFactorNode(
        ...     window=60,
        ...     operation="corr",
        ...     columns=["close", "volume"]
        ... )
    """

    def __init__(
        self,
        window: int = 20,
        operation: str = "mean",
        column: Optional[str] = None,
        columns: Optional[List[str]] = None,
        min_periods: Optional[int] = None,
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Args:
            window: 窗口大小
            operation: 运算类型 "mean", "std", "sum", "min", "max", "corr", "cov", "rolling"
            column: 要操作的列名（单列操作）
            columns: 要操作的列名列表（多列操作，如 corr）
            min_periods: 最小观测数
            name: 节点名称
            config: 额外配置
            **kwargs: 额外参数
        """
        super().__init__(Name=name or f"Time{operation.capitalize()}", config=config, **kwargs)
        self.window = window
        self.operation = operation
        self.column = column
        self.columns = columns
        self.min_periods = min_periods or window

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        """执行时间序列运算"""
        if input_data is None:
            raise ValueError("input_data is required for TimeFactorNode")

        if not isinstance(input_data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(input_data)}")

        df = input_data.copy()

        if self.operation == "mean":
            if self.column:
                result = df[self.column].rolling(window=self.window, min_periods=self.min_periods).mean()
            else:
                raise ValueError("column is required for mean operation")

        elif self.operation == "std":
            if self.column:
                result = df[self.column].rolling(window=self.window, min_periods=self.min_periods).std()
            else:
                raise ValueError("column is required for std operation")

        elif self.operation == "sum":
            if self.column:
                result = df[self.column].rolling(window=self.window, min_periods=self.min_periods).sum()
            else:
                raise ValueError("column is required for sum operation")

        elif self.operation == "min":
            if self.column:
                result = df[self.column].rolling(window=self.window, min_periods=self.min_periods).min()
            else:
                raise ValueError("column is required for min operation")

        elif self.operation == "max":
            if self.column:
                result = df[self.column].rolling(window=self.window, min_periods=self.min_periods).max()
            else:
                raise ValueError("column is required for max operation")

        elif self.operation == "corr":
            if self.columns and len(self.columns) == 2:
                result = df[self.columns[0]].rolling(window=self.window, min_periods=self.min_periods).corr(
                    df[self.columns[1]]
                )
            else:
                raise ValueError("columns must be a list of 2 column names for corr operation")

        elif self.operation == "cov":
            if self.columns and len(self.columns) == 2:
                result = df[self.columns[0]].rolling(window=self.window, min_periods=self.min_periods).cov(
                    df[self.columns[1]]
                )
            else:
                raise ValueError("columns must be a list of 2 column names for cov operation")

        elif self.operation == "rolling":
            if callable(getattr(self, '_custom_func', None)):
                result = df[self.column].rolling(window=self.window, min_periods=self.min_periods).apply(
                    self._custom_func, raw=False
                )
            else:
                raise ValueError("rolling operation requires _custom_func to be set")

        else:
            raise ValueError(f"Unknown operation: {self.operation}")

        result.name = self.name
        return result.to_frame()

    def __repr__(self) -> str:
        return f"<TimeFactorNode window={self.window} operation='{self.operation}'>"


class ExpandingFactorNode(FactorNode):
    """
    扩展窗口运算因子节点

    从数据开始到当前点的所有历史数据进行运算。
    """

    def __init__(
        self,
        operation: str = "mean",
        column: Optional[str] = None,
        min_periods: int = 1,
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__(Name=name or f"Expanding{operation.capitalize()}", config=config, **kwargs)
        self.operation = operation
        self.column = column
        self.min_periods = min_periods

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        """执行扩展窗口运算"""
        if input_data is None:
            raise ValueError("input_data is required for ExpandingFactorNode")

        if not isinstance(input_data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(input_data)}")

        df = input_data.copy()
        
        op_funcs = {
            "mean": lambda col: col.expanding(min_periods=self.min_periods).mean(),
            "std": lambda col: col.expanding(min_periods=self.min_periods).std(),
            "sum": lambda col: col.expanding(min_periods=self.min_periods).sum(),
            "min": lambda col: col.expanding(min_periods=self.min_periods).min(),
            "max": lambda col: col.expanding(min_periods=self.min_periods).max(),
        }

        op_func = op_funcs.get(self.operation)
        if op_func is None:
            raise ValueError(f"Unknown operation: {self.operation}")

        if self.column:
            result = op_func(df[self.column])
        else:
            raise ValueError("column is required")

        result.name = self.name
        return result.to_frame()

    def __repr__(self) -> str:
        return f"<ExpandingFactorNode operation='{self.operation}'>"
