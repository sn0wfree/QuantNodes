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


# ---------------------------------------------------------------------------
# 共用算子函数（消除 lambda 重复）
# ---------------------------------------------------------------------------

def _zscore_fn(x):
    """Z-score 标准化"""
    std = x.std()
    return (x - x.mean()) / std if std > 0 else x - x.mean()


def _demean_fn(x):
    """去均值"""
    return x - x.mean()


def _mad_fn(x):
    """Median Absolute Deviation"""
    return (x - x.median()).abs().median() * 1.4826


def _add(a, b):
    return a + b


def _sub(a, b):
    return a - b


def _mul(a, b):
    return a * b


def _div(a, b):
    return a / b


_ARITH_OPS = {"add": _add, "sub": _sub, "mul": _mul, "div": _div}

_EXPANDING_ATTRS = {"mean": "mean", "std": "std", "sum": "sum", "min": "min", "max": "max"}

_CS_OPS = {"rank", "zscore", "demean", "mad", "percentile"}


def _groupby_transform(df, column, dt_key, func, groupby=None):
    """通用 groupby + transform"""
    keys = [dt_key] + ([groupby] if groupby else [])
    return df.groupby(keys)[column].transform(func)


# ---------------------------------------------------------------------------
# FactorNode 基类
# ---------------------------------------------------------------------------

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
        pass

    def _execute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        self._result = self._compute(input_data, **kwargs)
        return self._result

    def __rshift__(self, other: 'FactorNode') -> 'FactorPipeline':
        return FactorPipeline([self, other])

    def then(self, other: 'FactorNode') -> 'FactorPipeline':
        return FactorPipeline([self, other])

    # -- 辅助方法（消除子类重复代码） --

    def _validate_input(self, input_data):
        """校验 + copy DataFrame"""
        if input_data is None:
            raise ValueError(f"input_data is required for {self.__class__.__name__}")
        if not isinstance(input_data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(input_data)}")
        return input_data.copy()

    def _finalize(self, result):
        """命名 + to_frame"""
        result.name = self.name
        return result.to_frame()

    @staticmethod
    def _get_dt_key(df):
        """获取日期分组 key"""
        return 'dt' if 'dt' in df.columns else df.index

    @staticmethod
    def _extract_first_col(base_result):
        """从 base_factor 结果提取第一列"""
        if isinstance(base_result, pd.DataFrame):
            return base_result.iloc[:, 0]
        return base_result


# ---------------------------------------------------------------------------
# FactorPipeline
# ---------------------------------------------------------------------------

class FactorPipeline:
    """因子管道"""

    def __init__(self, factors: List[FactorNode]):
        self.factors = factors

    def execute(self, input_data: Any = None, **kwargs) -> Dict[str, pd.DataFrame]:
        results = {}
        for i, factor in enumerate(self.factors):
            name = factor.name if hasattr(factor, 'name') else f"Factor{i}"
            if name in results:
                name = f"{name}_{i}"
            results[name] = factor.execute(input_data, **kwargs)
        return results

    def __rshift__(self, other: FactorNode) -> 'FactorPipeline':
        if isinstance(other, FactorPipeline):
            return FactorPipeline(self.factors + other.factors)
        return FactorPipeline(self.factors + [other])


# ---------------------------------------------------------------------------
# PointFactorNode
# ---------------------------------------------------------------------------

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
        super().__init__(Name=name or "PointFactor", config=config, **kwargs)
        self.expression = expression
        self.func = func
        self.result_name = result_name

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        df = self._validate_input(input_data)

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


# ---------------------------------------------------------------------------
# ArithmeticFactorNode
# ---------------------------------------------------------------------------

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
        super().__init__(Name=name or f"Arithmetic{operator.capitalize()}", config=config, **kwargs)
        self.factors = factors
        self.operator = operator

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        if len(self.factors) < 2:
            raise ValueError("At least 2 factors are required")

        results = [f.execute(input_data, **kwargs) for f in self.factors]
        series_list = [FactorNode._extract_first_col(r) for r in results]

        op_func = _ARITH_OPS.get(self.operator)
        if op_func is None:
            raise ValueError(f"Unknown operator: {self.operator}")

        result = series_list[0]
        for s in series_list[1:]:
            result = op_func(result, s)

        return self._finalize(result)

    def __repr__(self) -> str:
        return f"<ArithmeticFactorNode operator='{self.operator}' factors={len(self.factors)}>"


# ---------------------------------------------------------------------------
# TimeFactorNode
# ---------------------------------------------------------------------------

class TimeFactorNode(FactorNode):
    """
    时间序列运算因子节点

    对时间序列数据进行窗口运算，如移动平均、滚动标准差等。

    Examples:
        >>> # 移动平均
        >>> factor = TimeFactorNode(window=20, operation="mean", column="close")
        >>> result = factor.execute(df)
        >>>
        >>> # 滚动相关系数
        >>> factor = TimeFactorNode(window=60, operation="corr", columns=["close", "volume"])
    """

    _SINGLE_COL_OPS = {"mean", "std", "sum", "min", "max"}
    _DUAL_COL_OPS = {"corr", "cov"}

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
        super().__init__(Name=name or f"Time{operation.capitalize()}", config=config, **kwargs)
        self.window = window
        self.operation = operation
        self.column = column
        self.columns = columns
        self.min_periods = min_periods or window

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        df = self._validate_input(input_data)

        if self.operation in self._SINGLE_COL_OPS:
            if not self.column:
                raise ValueError(f"column is required for {self.operation} operation")
            result = getattr(
                df[self.column].rolling(window=self.window, min_periods=self.min_periods),
                self.operation,
            )()

        elif self.operation in self._DUAL_COL_OPS:
            if not self.columns or len(self.columns) != 2:
                raise ValueError(
                    f"columns must be a list of 2 column names for {self.operation} operation"
                )
            result = getattr(
                df[self.columns[0]].rolling(window=self.window, min_periods=self.min_periods),
                self.operation,
            )(df[self.columns[1]])

        elif self.operation == "rolling":
            if not callable(getattr(self, '_custom_func', None)):
                raise ValueError("rolling operation requires _custom_func to be set")
            result = df[self.column].rolling(
                window=self.window, min_periods=self.min_periods
            ).apply(self._custom_func, raw=False)

        else:
            raise ValueError(f"Unknown operation: {self.operation}")

        return self._finalize(result)

    def __repr__(self) -> str:
        return f"<TimeFactorNode window={self.window} operation='{self.operation}'>"


# ---------------------------------------------------------------------------
# ExpandingFactorNode
# ---------------------------------------------------------------------------

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
        df = self._validate_input(input_data)

        attr = _EXPANDING_ATTRS.get(self.operation)
        if attr is None:
            raise ValueError(f"Unknown operation: {self.operation}")
        if not self.column:
            raise ValueError("column is required")

        result = getattr(df[self.column].expanding(min_periods=self.min_periods), attr)()
        return self._finalize(result)

    def __repr__(self) -> str:
        return f"<ExpandingFactorNode operation='{self.operation}'>"


# ---------------------------------------------------------------------------
# CrossSectionFactorNode
# ---------------------------------------------------------------------------

class CrossSectionFactorNode(FactorNode):
    """
    截面运算因子节点

    对某一时间点的所有标的进行横截面运算，如市值加权、排名等。

    Examples:
        >>> # 横截面排名
        >>> factor = CrossSectionFactorNode(operation="rank", column="return")
        >>> result = factor.execute(df)
        >>>
        >>> # 横截面去均值
        >>> factor = CrossSectionFactorNode(operation="demean", column="return")
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
        super().__init__(Name=name or f"CrossSection{operation.capitalize()}", config=config, **kwargs)
        self.operation = operation
        self.column = column
        self.groupby = groupby

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        df = self._validate_input(input_data)

        if self.column is None:
            raise ValueError("column is required for CrossSectionFactorNode")

        dt_key = self._get_dt_key(df)

        if self.operation == "rank":
            pct = 'pct' in str(kwargs.get('mode', ''))
            result = df.groupby(dt_key)[self.column].rank(pct=pct)

        elif self.operation == "zscore":
            result = _groupby_transform(df, self.column, dt_key, _zscore_fn, self.groupby)

        elif self.operation == "demean":
            result = _groupby_transform(df, self.column, dt_key, _demean_fn, self.groupby)

        elif self.operation == "mad":
            result = _groupby_transform(df, self.column, dt_key, _mad_fn, self.groupby)

        elif self.operation == "percentile":
            result = df.groupby(dt_key)[self.column].rank(pct=True)

        else:
            raise ValueError(f"Unknown operation: {self.operation}")

        return self._finalize(result)

    def __repr__(self) -> str:
        return f"<CrossSectionFactorNode operation='{self.operation}' column='{self.column}'>"


# ---------------------------------------------------------------------------
# GroupRankFactorNode
# ---------------------------------------------------------------------------

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
        df = self._validate_input(input_data)

        dt_key = self._get_dt_key(df)
        if 'dt' in df.columns:
            result = df.groupby(['dt', self.groupby])[self.column].rank(ascending=self.ascending)
        else:
            result = df.groupby(self.groupby)[self.column].rank(ascending=self.ascending)

        return self._finalize(result)

    def __repr__(self) -> str:
        return f"<GroupRankFactorNode column='{self.column}' groupby='{self.groupby}'>"


# ---------------------------------------------------------------------------
# PanelFactorNode
# ---------------------------------------------------------------------------

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
        super().__init__(Name=name or "PanelFactor", config=config, **kwargs)
        self.operations = operations or []
        self.combine = combine

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        df = self._validate_input(input_data)

        results = []
        for op_name, params in self.operations:
            column = params.get("column")
            groupby = params.get("groupby")

            if op_name == "zscore":
                results.append(self._zscore(df, column, groupby))
            elif op_name == "demean":
                results.append(self._demean(df, column, groupby))
            elif op_name == "rank":
                pct = params.get("pct", True)
                results.append(self._rank(df, column, groupby, pct))
            else:
                raise ValueError(f"Unknown operation: {op_name}")

        if not results:
            raise ValueError("No operations specified")

        return self._combine_results(results)

    def _zscore(self, df, column, groupby=None):
        return _groupby_transform(df, column, self._get_dt_key(df), _zscore_fn, groupby)

    def _demean(self, df, column, groupby=None):
        return _groupby_transform(df, column, self._get_dt_key(df), _demean_fn, groupby)

    def _rank(self, df, column, groupby=None, pct=True):
        dt_key = self._get_dt_key(df)
        keys = [dt_key] + ([groupby] if groupby else [])
        return df.groupby(keys)[column].rank(pct=pct)

    def _combine_results(self, results):
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

        return self._finalize(combined)

    def __repr__(self) -> str:
        return f"<PanelFactorNode operations={len(self.operations)}>"


# ---------------------------------------------------------------------------
# DelayFactorNode
# ---------------------------------------------------------------------------

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
        if input_data is None:
            raise ValueError("input_data is required for DelayFactorNode")

        base_result = self.base_factor.execute(input_data, **kwargs)
        col = self._extract_first_col(base_result)
        result = col.shift(self.periods)
        return self._finalize(result)

    def __repr__(self) -> str:
        return f"<DelayFactorNode periods={self.periods}>"


# ---------------------------------------------------------------------------
# DeltaFactorNode
# ---------------------------------------------------------------------------

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
        super().__init__(Name=name or f"Delta{mode.capitalize()}", config=config, **kwargs)
        self.base_factor = base_factor
        self.periods = periods
        self.mode = mode

    def _compute(self, input_data: Any = None, **kwargs) -> pd.DataFrame:
        if input_data is None:
            raise ValueError("input_data is required for DeltaFactorNode")

        base_result = self.base_factor.execute(input_data, **kwargs)
        col = self._extract_first_col(base_result)

        if self.mode == "diff":
            result = col.diff(self.periods)
        elif self.mode == "pct_change":
            result = col.pct_change(self.periods)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return self._finalize(result)

    def __repr__(self) -> str:
        return f"<DeltaFactorNode periods={self.periods} mode='{self.mode}'>"
