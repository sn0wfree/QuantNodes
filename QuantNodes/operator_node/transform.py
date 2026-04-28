# coding=utf-8
"""
TransformNode - 数据转换节点

提供数据转换操作，如选择列、过滤、聚合等。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from QuantNodes.operator_node.base import OperatorNode


class TransformNode(OperatorNode):
    """
    数据转换节点

    对 DataFrame 进行各种转换操作，支持链式调用。

    Examples:
        >>> # 选择列
        >>> node = TransformNode().select(["col1", "col2"])
        >>> result = node.execute(df)
        >>>
        >>> # 过滤行
        >>> node = TransformNode().filter(lambda df: df["value"] > 0)
        >>> result = node.execute(df)
        >>>
        >>> # 聚合
        >>> node = TransformNode().aggregate(group_by=["category"], agg={"value": "sum"})
        >>> result = node.execute(df)
    """

    def __init__(
        self,
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        super().__init__(name=name or "Transform", config=config, **kwargs)
        self._operations: List[Callable] = []

    def select(self, columns: List[str]) -> 'TransformNode':
        """选择列"""
        def op(df: pd.DataFrame) -> pd.DataFrame:
            return df[columns]
        self._operations.append(op)
        return self

    def drop(self, columns: List[str]) -> 'TransformNode':
        """删除列"""
        def op(df: pd.DataFrame) -> pd.DataFrame:
            return df.drop(columns=columns)
        self._operations.append(op)
        return self

    def filter(self, condition: Union[str, Callable]) -> 'TransformNode':
        """过滤行"""
        if isinstance(condition, str):
            def op(df: pd.DataFrame) -> pd.DataFrame:
                return df.query(condition)
        else:
            def op(df: pd.DataFrame) -> pd.DataFrame:
                return df[condition(df)]
        self._operations.append(op)
        return self

    def aggregate(
        self,
        group_by: List[str],
        agg: Dict[str, Union[str, List[str]]]
    ) -> 'TransformNode':
        """聚合操作"""
        def op(df: pd.DataFrame) -> pd.DataFrame:
            return df.groupby(group_by).agg(agg).reset_index()
        self._operations.append(op)
        return self

    def sort_by(self, columns: Union[str, List[str]], ascending: bool = True) -> 'TransformNode':
        """排序"""
        def op(df: pd.DataFrame) -> pd.DataFrame:
            return df.sort_values(by=columns, ascending=ascending)
        self._operations.append(op)
        return self

    def rename(self, columns: Dict[str, str]) -> 'TransformNode':
        """重命名列"""
        def op(df: pd.DataFrame) -> pd.DataFrame:
            return df.rename(columns=columns)
        self._operations.append(op)
        return self

    def fillna(self, value: Any) -> 'TransformNode':
        """填充空值"""
        def op(df: pd.DataFrame) -> pd.DataFrame:
            return df.fillna(value)
        self._operations.append(op)
        return self

    def apply(self, func: Callable, **kwargs) -> 'TransformNode':
        """应用自定义函数"""
        def op(df: pd.DataFrame) -> pd.DataFrame:
            return df.apply(func, **kwargs)
        self._operations.append(op)
        return self

    def then(self, other: 'TransformNode') -> 'TransformNode':
        """链式调用"""
        combined = TransformNode(name=f"{self.name}_then_{other.name}")
        combined._operations = self._operations + other._operations
        return combined

    def _execute_operation(self, input_data: Any = None, **kwargs) -> Any:
        """执行转换"""
        if input_data is None:
            raise ValueError("input_data (DataFrame) is required")

        if not isinstance(input_data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(input_data)}")

        result = input_data
        for op in self._operations:
            result = op(result)

        return result

    def __repr__(self) -> str:
        return f"<TransformNode operations={len(self._operations)}>"
