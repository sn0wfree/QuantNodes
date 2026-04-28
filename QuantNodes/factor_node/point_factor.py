# coding=utf-8
"""
PointFactorNode - 单点运算因子节点

对数据进行单点运算，不涉及时间序列或截面运算。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union, Callable

import pandas as pd
import numpy as np

from QuantNodes.factor_node.factor_node import FactorNode


class PointFactorNode(FactorNode):
    """
    单点运算因子节点

    对数据进行行级别的运算，每个结果行只依赖于对应行的输入数据。

    Examples:
        >>> # 简单表达式
        >>> factor = PointFactorNode(expression="close / open - 1")
        >>> result = factor.execute(df)
        >>>
        >>> # 自定义函数
        >>> factor = PointFactorNode(func=lambda row: row['close'] / row['open'] - 1)
        >>> result = factor.execute(df)
    """

    def __init__(
        self,
        expression: Optional[str] = None,
        func: Optional[Callable] = None,
        result_name: str = "result",
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Args:
            expression: 计算表达式，如 "close / open - 1"
            func: 自定义计算函数，接受 row 或整个 DataFrame
            result_name: 结果列名
            name: 节点名称
            config: 额外配置
            **kwargs: 额外参数
        """
        super().__init__(Name=name or "PointFactor", config=config, **kwargs)
        self.expression = expression
        self.func = func
        self.result_name = result_name

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        """执行单点运算"""
        if input_data is None:
            raise ValueError("input_data is required for PointFactorNode")
        
        if not isinstance(input_data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(input_data)}")

        df = input_data.copy()

        if self.func is not None:
            if self.func.__code__.co_argcount == 1:
                df[self.result_name] = df.apply(self.func, axis=1)
            else:
                df[self.result_name] = self.func(df)
        elif self.expression is not None:
            df[self.result_name] = df.eval(self.expression)
        else:
            raise ValueError("Either expression or func must be provided")

        return df[[self.result_name]]

    def __repr__(self) -> str:
        if self.expression:
            return f"<PointFactorNode expression='{self.expression}'>"
        return f"<PointFactorNode func={self.func.__name__ if self.func else None}>"


class ArithmeticFactorNode(FactorNode):
    """
    算术运算因子节点

    支持加减乘除等基本算术运算的因子组合。

    Examples:
        >>> factor = ArithmeticFactorNode(
        ...     factors=[close_factor, open_factor],
        ...     operator="div"
        ... )
    """

    def __init__(
        self,
        factors: List[FactorNode],
        operator: str = "add",
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Args:
            factors: 要组合的因子列表
            operator: 运算符 "add", "sub", "mul", "div"
            name: 节点名称
            config: 额外配置
            **kwargs: 额外参数
        """
        super().__init__(Name=name or f"Arithmetic{operator.capitalize()}", config=config, **kwargs)
        self.factors = factors
        self.operator = operator

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        """执行算术运算"""
        if len(self.factors) < 2:
            raise ValueError("At least 2 factors are required")

        results = [f.execute(input_data, **kwargs) for f in self.factors]
        
        def extract_series(df):
            if isinstance(df, pd.DataFrame):
                return df.iloc[:, 0]
            return df

        series_list = [extract_series(r) for r in results]
        
        op_funcs = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: a / b,
        }
        
        op_func = op_funcs.get(self.operator)
        if op_func is None:
            raise ValueError(f"Unknown operator: {self.operator}")

        result = series_list[0]
        for s in series_list[1:]:
            result = op_func(result, s)

        result.name = self.name
        return result.to_frame()

    def __repr__(self) -> str:
        return f"<ArithmeticFactorNode operator='{self.operator}' factors={len(self.factors)}>"
