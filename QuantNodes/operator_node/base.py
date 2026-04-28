# coding=utf-8
"""
OperatorNode 基类模块

提供操作节点的基础架构，继承自 BaseNode。
用于数据操作、SQL 构建和转换。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from QuantNodes.core.node import BaseNode


class OperatorNode(BaseNode, ABC):
    """
    操作节点基类

    所有数据操作节点都继承自此类，提供统一的操作接口。
    支持链式调用，可以像构建 SQL 一样组合操作。

    Subclasses must implement:
        _execute_operation(): 执行具体操作

    Examples:
        >>> query_node = TableQueryNode(table="users")
        >>> result = query_node.execute()
    """

    def __init__(self, name: str = None, config: Dict[str, Any] = None, **kwargs):
        super().__init__(name=name or self.__class__.__name__, config=config, **kwargs)
        self._chained_result: Any = None

    @abstractmethod
    def _execute_operation(self, input_data: Any = None, **kwargs) -> Any:
        """
        执行具体操作

        Args:
            input_data: 输入数据
            **kwargs: 执行参数

        Returns:
            操作结果
        """
        pass

    def _execute(self, input_data: Any = None, **kwargs) -> Any:
        """执行操作"""
        return self._execute_operation(input_data, **kwargs)

    def then(self, other: 'OperatorNode') -> 'ChainOperator':
        """
        链式调用：将操作链接到另一个操作

        Args:
            other: 下一个操作节点

        Returns:
            ChainOperator 包含两个操作的链接
        """
        return ChainOperator([self, other])

    def __rshift__(self, other: 'OperatorNode') -> 'ChainOperator':
        """重载 >> 运算符支持链式调用"""
        return self.then(other)


class ChainOperator(OperatorNode):
    """
    链式操作节点

    将多个 OperatorNode 链接在一起执行。
    """

    def __init__(self, operators: List[OperatorNode], name: str = None, config: Dict[str, Any] = None):
        super().__init__(name=name or "Chain", config=config)
        self.operators = operators

    def _execute_operation(self, input_data: Any = None, **kwargs) -> Any:
        """依次执行每个操作"""
        result = input_data
        for op in self.operators:
            result = op.execute(result, **kwargs)
        return result

    def _get_serializable_fields(self) -> Dict[str, Any]:
        return {"operators": [op.serialize() for op in self.operators]}

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'ChainOperator':
        operators = [BaseNode.deserialize(op_data) for op_data in data["operators"]]
        return ChainOperator(operators=operators, name=data.get("name"))
