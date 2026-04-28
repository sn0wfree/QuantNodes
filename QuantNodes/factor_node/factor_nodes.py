# coding=utf-8
"""
FactorNode - 因子计算节点集合

提供各类因子计算节点，包括基类和各种运算节点。
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable

import pandas as pd
import numpy as np

from QuantNodes.core.node import BaseNode

class FactorNode(BaseNode, ABC):
    """
    因子计算节点基类

    所有因子计算节点都继承自此类，提供统一的因子计算接口。

    Subclasses must implement:
        _compute(): 执行具体因子计算

    Examples:
        >>> # 单点运算
        >>> factor = PointFactorNode(expression="close / open - 1")
        >>> result = factor.execute(data)
    """

    def __init__(self, name: str = None, config: Dict[str, Any] = None, **kwargs):
        default_name = f"{self.__class__.__name__}"
        super().__init__(name=name or default_name, config=config, **kwargs)
        self._result: Optional[pd.DataFrame] = None

    @abstractmethod
    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        """
        执行因子计算

        Args:
            input_data: 输入数据（DataFrame 或数据库连接）
            **kwargs: 额外执行参数

        Returns:
            因子计算结果 DataFrame
        """
        pass

    def _execute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        """执行因子计算"""
        self._result = self._compute(input_data, **kwargs)
        return self._result

    def __rshift__(self, other: 'FactorNode') -> 'FactorPipeline':
        """重载 >> 运算符用于因子组合"""
        return FactorPipeline([self, other])

    def then(self, other: 'FactorNode') -> 'FactorPipeline':
        """链式调用"""
        return FactorPipeline([self, other])


class FactorPipeline:
    """
    因子管道

    将多个因子节点组合在一起计算。
    """

    def __init__(self, factors: List[FactorNode]):
        self.factors = factors

    def execute(self, input_data: Any = None, **kwargs) -> Dict[str, pd.DataFrame]:
        """执行所有因子计算"""
        results = {}
        for i, factor in enumerate(self.factors):
            name = factor.name if hasattr(factor, 'name') else f"Factor{i}"
            if name in results:
                name = f"{name}_{i}"
            results[name] = factor.execute(input_data, **kwargs)
        return results

    def __rshift__(self, other: FactorNode) -> 'FactorPipeline':
        """重载 >> 运算符"""
        if isinstance(other, FactorPipeline):
            return FactorPipeline(self.factors + other.factors)
        return FactorPipeline(self.factors + [other])


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


class CrossSectionFactorNode(FactorNode):
    """
    截面运算因子节点

    对某一时间点的所有标的进行横截面运算，如市值加权、排名等。

    Examples:
        >>> # 横截面排名
        >>> factor = CrossSectionFactorNode(
        ...     operation="rank",
        ...     column="return"
        ... )
        >>> result = factor.execute(df)
        >>>
        >>> # 横截面去均值
        >>> factor = CrossSectionFactorNode(
        ...     operation="demean",
        ...     column="return"
        ... )
    """

    def __init__(
        self,
        operation: str = "rank",
        column: Optional[str] = None,
        groupby: Optional[str] = None,
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Args:
            operation: 运算类型 "rank", "zscore", "demean", "mad", "percentile"
            column: 要操作的列名
            groupby: 分组列名（如行业分组）
            name: 节点名称
            config: 额外配置
            **kwargs: 额外参数
        """
        super().__init__(Name=name or f"CrossSection{operation.capitalize()}", config=config, **kwargs)
        self.operation = operation
        self.column = column
        self.groupby = groupby

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        """执行截面运算"""
        if input_data is None:
            raise ValueError("input_data is required for CrossSectionFactorNode")

        if not isinstance(input_data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(input_data)}")

        df = input_data.copy()

        if self.column is None:
            raise ValueError("column is required for CrossSectionFactorNode")

        if self.operation == "rank":
            result = df.groupby('dt' if 'dt' in df.columns else df.index)[self.column].rank(
                pct=True if 'pct' in str(kwargs.get('mode', '')) else False
            )

        elif self.operation == "zscore":
            def zscore(x):
                return (x - x.mean()) / x.std() if x.std() > 0 else x - x.mean()
            
            if self.groupby:
                result = df.groupby(['dt' if 'dt' in df.columns else df.index, self.groupby])[self.column].transform(zscore)
            else:
                result = df.groupby('dt' if 'dt' in df.columns else df.index)[self.column].transform(zscore)

        elif self.operation == "demean":
            if self.groupby:
                result = df.groupby(['dt' if 'dt' in df.columns else df.index, self.groupby])[self.column].transform(
                    lambda x: x - x.mean()
                )
            else:
                result = df.groupby('dt' if 'dt' in df.columns else df.index)[self.column].transform(
                    lambda x: x - x.mean()
                )

        elif self.operation == "mad":
            def mad(x):
                return (x - x.median()).abs().median() * 1.4826
            
            if self.groupby:
                result = df.groupby(['dt' if 'dt' in df.columns else df.index, self.groupby])[self.column].transform(mad)
            else:
                result = df.groupby('dt' if 'dt' in df.columns else df.index)[self.column].transform(mad)

        elif self.operation == "percentile":
            result = df.groupby('dt' if 'dt' in df.columns else df.index)[self.column].rank(pct=True)

        else:
            raise ValueError(f"Unknown operation: {self.operation}")

        result.name = self.name
        return result.to_frame()

    def __repr__(self) -> str:
        return f"<CrossSectionFactorNode operation='{self.operation}' column='{self.column}'>"


class GroupRankFactorNode(FactorNode):
    """
    分组排名因子节点

    按组别进行排名，常用于 Barra 风格因子。
    """

    def __init__(
        self,
        column: str,
        groupby: str,
        ascending: bool = False,
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__(Name=name or "GroupRank", config=config, **kwargs)
        self.column = column
        self.groupby = groupby
        self.ascending = ascending

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        """执行分组排名"""
        if input_data is None:
            raise ValueError("input_data is required for GroupRankFactorNode")

        if not isinstance(input_data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(input_data)}")

        df = input_data.copy()

        if 'dt' in df.columns:
            result = df.groupby(['dt', self.groupby])[self.column].rank(ascending=self.ascending)
        else:
            result = df.groupby(self.groupby)[self.column].rank(ascending=self.ascending)

        result.name = self.name
        return result.to_frame()

    def __repr__(self) -> str:
        return f"<GroupRankFactorNode column='{self.column}' groupby='{self.groupby}'>"


class PanelFactorNode(FactorNode):
    """
    面板运算因子节点

    结合时间序列和截面运算，如日间因子 -> 截面 rank -> 日间平均。

    Examples:
        >>> # 因子去行业中性和市值中性
        >>> factor = PanelFactorNode(
        ...     operations=[
        ...         ("zscore", {"column": "return", "groupby": "industry"}),
        ...         ("zscore", {"column": "return", "groupby": "size"}),
        ...     ],
        ...     combine="add"
        ... )
        >>> result = factor.execute(df)
    """

    def __init__(
        self,
        operations: Optional[List[tuple]] = None,
        combine: str = "add",
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Args:
            operations: 操作列表，每项为 (operation_name, params) 元组
            combine: 组合方式 "add", "mul", "mean"
            name: 节点名称
            config: 额外配置
            **kwargs: 额外参数
        """
        super().__init__(Name=name or "PanelFactor", config=config, **kwargs)
        self.operations = operations or []
        self.combine = combine

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        """执行面板运算"""
        if input_data is None:
            raise ValueError("input_data is required for PanelFactorNode")

        if not isinstance(input_data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(input_data)}")

        df = input_data.copy()
        results = []

        for op_name, params in self.operations:
            if op_name == "zscore":
                column = params.get("column")
                groupby = params.get("groupby")
                result = self._zscore(df, column, groupby)
                results.append(result)

            elif op_name == "demean":
                column = params.get("column")
                groupby = params.get("groupby")
                result = self._demean(df, column, groupby)
                results.append(result)

            elif op_name == "rank":
                column = params.get("column")
                groupby = params.get("groupby")
                pct = params.get("pct", True)
                result = self._rank(df, column, groupby, pct)
                results.append(result)

            else:
                raise ValueError(f"Unknown operation: {op_name}")

        if not results:
            raise ValueError("No operations specified")

        return self._combine_results(results)

    def _zscore(self, df: pd.DataFrame, column: str, groupby: Optional[str] = None) -> pd.Series:
        """Z-score 标准化"""
        if groupby:
            return df.groupby(['dt' if 'dt' in df.columns else df.index, groupby])[column].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )
        else:
            return df.groupby('dt' if 'dt' in df.columns else df.index)[column].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )

    def _demean(self, df: pd.DataFrame, column: str, groupby: Optional[str] = None) -> pd.Series:
        """去均值"""
        if groupby:
            return df.groupby(['dt' if 'dt' in df.columns else df.index, groupby])[column].transform(
                lambda x: x - x.mean()
            )
        else:
            return df.groupby('dt' if 'dt' in df.columns else df.index)[column].transform(
                lambda x: x - x.mean()
            )

    def _rank(self, df: pd.DataFrame, column: str, groupby: Optional[str] = None, pct: bool = True) -> pd.Series:
        """排名"""
        if groupby:
            return df.groupby(['dt' if 'dt' in df.columns else df.index, groupby])[column].rank(pct=pct)
        else:
            return df.groupby('dt' if 'dt' in df.columns else df.index)[column].rank(pct=pct)

    def _combine_results(self, results: List[pd.Series]) -> pd.DataFrame:
        """合并多个结果"""
        if self.combine == "add":
            combined = sum(results)
        elif self.combine == "mul":
            combined = results[0]
            for r in results[1:]:
                combined = combined * r
        elif self.combine == "mean":
            combined = pd.concat(results, axis=1).mean(axis=1)
        else:
            raise ValueError(f"Unknown combine method: {self.combine}")

        combined.name = self.name
        return combined.to_frame()

    def __repr__(self) -> str:
        return f"<PanelFactorNode operations={len(self.operations)}>"


class DelayFactorNode(FactorNode):
    """
    延迟因子节点

    返回过去 N 天的因子值。
    """

    def __init__(
        self,
        base_factor: FactorNode,
        periods: int = 1,
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__(Name=name or f"Delay{periods}", config=config, **kwargs)
        self.base_factor = base_factor
        self.periods = periods

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        """执行延迟操作"""
        if input_data is None:
            raise ValueError("input_data is required for DelayFactorNode")

        base_result = self.base_factor.execute(input_data, **kwargs)
        
        if isinstance(base_result, pd.DataFrame):
            col = base_result.iloc[:, 0]
        else:
            col = base_result
            
        result = col.shift(self.periods)
        result.name = self.name
        return result.to_frame()

    def __repr__(self) -> str:
        return f"<DelayFactorNode periods={self.periods}>"


class DeltaFactorNode(FactorNode):
    """
    变化率因子节点

    计算因子值的变化率或差值。
    """

    def __init__(
        self,
        base_factor: FactorNode,
        periods: int = 1,
        mode: str = "diff",
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Args:
            base_factor: 基础因子
            periods: 滞后期数
            mode: "diff" (差分) 或 "pct_change" (百分比变化)
            name: 节点名称
            config: 额外配置
            **kwargs: 额外参数
        """
        super().__init__(Name=name or f"Delta{mode.capitalize()}", config=config, **kwargs)
        self.base_factor = base_factor
        self.periods = periods
        self.mode = mode

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        """执行差分/变化率操作"""
        if input_data is None:
            raise ValueError("input_data is required for DeltaFactorNode")

        base_result = self.base_factor.execute(input_data, **kwargs)
        
        if isinstance(base_result, pd.DataFrame):
            col = base_result.iloc[:, 0]
        else:
            col = base_result

        if self.mode == "diff":
            result = col.diff(self.periods)
        elif self.mode == "pct_change":
            result = col.pct_change(self.periods)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        result.name = self.name
        return result.to_frame()

    def __repr__(self) -> str:
        return f"<DeltaFactorNode periods={self.periods} mode='{self.mode}'>"
