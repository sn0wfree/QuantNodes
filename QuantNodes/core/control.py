# coding=utf-8
"""
控制流节点模块

本模块提供流程控制节点，支持条件分支、分组映射、循环执行等功能：
1. IfNode: 条件分支，根据输入决定执行哪个分支
2. MapNode: 分组映射，对每个分组执行相同节点
3. WhileNode: 条件循环，满足条件则循环执行

支持多种条件表达方式：
1. DSL 构建: Cond('value') > 50
2. 字符串表达式: "df['value'] > 50"
3. Lambda 函数: lambda x: x > 50 (向后兼容)
"""

from __future__ import annotations

import logging
import pandas as pd
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar, Union

from QuantNodes.core.node import BaseNode, SerializationError
from QuantNodes.core.expression import Expression, ExpressionBuilder, LambdaExpression
from QuantNodes.core.serializable import serializable


I = TypeVar('I')  # 分组输入类型
O = TypeVar('O')  # 分组输出类型


def _wrap_condition(
    condition: Union[Expression, ExpressionBuilder, Callable[[Any], bool], str]
) -> Expression:
    """将条件转换为 Expression 对象"""
    if isinstance(condition, ExpressionBuilder):
        return condition._expr
    if isinstance(condition, Expression):
        return condition
    if isinstance(condition, str):
        return Expression.parse(condition)
    if callable(condition):
        return LambdaExpression(condition)
    raise TypeError(f"Unsupported condition type: {type(condition)}")


@serializable
class IfNode(BaseNode):
    """
    条件分支节点

    如果条件满足，执行 true_branch，否则执行 false_branch。

    Examples:
        >>> # DSL 构建方式
        >>> IfNode(
        ...     condition=Cond('value') > 10,
        ...     true_branch=BigStrategy(),
        ...     false_branch=SmallStrategy(),
        ... )
        >>>
        >>> # 字符串表达式方式
        >>> IfNode("df['value'] > 10", BigStrategy(), SmallStrategy())
        >>>
        >>> # Lambda 方式（向后兼容）
        >>> IfNode(lambda x: x > 10, BigStrategy())
    """

    def __init__(self,
                 condition: Union[Expression, ExpressionBuilder, Callable[[Any], bool], str],
                 true_branch: BaseNode,
                 false_branch: Optional[BaseNode] = None,
                 name: str = None,
                 config: Dict[str, Any] = None):
        """
        Args:
            condition: 条件判断表达式，支持多种格式
            true_branch: 条件为 True 时执行的节点
            false_branch: 条件为 False 时执行的节点，None 表示直接返回原输入
            name: 节点名称
            config: 配置字典
        """
        super().__init__(name=name or "IfNode", config=config)
        self.condition = _wrap_condition(condition)
        self.true_branch = true_branch
        self.false_branch = false_branch
        self._last_branch_taken: Optional[bool] = None
        self.logger = logging.getLogger(f"node.{self.node_id}")

    def _execute(self, input_data: Any = None, **kwargs) -> Any:
        """执行条件分支"""
        cond_result = self.condition.evaluate(input_data)
        self._last_branch_taken = cond_result

        if cond_result:
            self.logger.debug("Condition is True, executing true branch")
            return self.true_branch.execute(input_data, **kwargs)
        elif self.false_branch is not None:
            self.logger.debug("Condition is False, executing false branch")
            return self.false_branch.execute(input_data, **kwargs)
        else:
            self.logger.debug("Condition is False, no false branch, returning input")
            return input_data

    def _get_serializable_fields(self) -> Dict[str, Any]:
        """返回需要序列化的额外字段"""
        if isinstance(self.condition, LambdaExpression):
            raise SerializationError(
                "IfNode with lambda condition cannot be serialized. "
                "Please use Cond() DSL or string expression instead."
            )
        result = {
            "condition": self.condition.serialize(),
            "true_branch": self.true_branch.serialize(),
        }
        if self.false_branch:
            result["false_branch"] = self.false_branch.serialize()
        return result

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'IfNode':
        """从字典反序列化重建 IfNode"""
        condition = Expression.deserialize(data["condition"])
        true_branch = BaseNode.deserialize(data["true_branch"])
        false_branch_data = data.get("false_branch")
        false_branch = BaseNode.deserialize(false_branch_data) if false_branch_data else None

        return IfNode(
            condition=condition,
            true_branch=true_branch,
            false_branch=false_branch,
            name=data.get("name"),
            config=data.get("config", {})
        )

    def to_info(self) -> Dict[str, Any]:
        """导出节点信息"""
        result = super().to_info()
        result['condition'] = repr(self.condition)
        result['condition_dict'] = self.condition.serialize()
        result['true_branch'] = self.true_branch.to_info()
        result['false_branch'] = self.false_branch.to_info() if self.false_branch else None
        result['last_branch_taken'] = self._last_branch_taken
        return result


@serializable
class MapNode(BaseNode, Generic[I, O]):
    """
    分组映射节点

    对输入数据进行分组，对每个分组执行相同的节点，最后合并结果。
    支持并行处理，适合内存友好的流式回测。

    Examples:
        >>> # 按列名分组
        >>> MapNode(
        ...     node=BacktestNode(),
        ...     group_by='date',
        ...     max_workers=4,
        ... )
        >>>
        >>> # DSL 表达式分组
        >>> MapNode(
        ...     node=FactorComputeNode(),
        ...     group_by=Cond('code').str[:3],
        ...     parallel=False,
        ... )
    """

    def __init__(self,
                 node: BaseNode[I, O],
                 group_by: Union[str, Expression, ExpressionBuilder, Callable[[Any], Any]],
                 max_workers: Optional[int] = None,
                 parallel: bool = True,
                 name: str = None,
                 config: Dict[str, Any] = None):
        """
        Args:
            node: 要在每个分组上执行的节点
            group_by: 分组方式，支持多种格式
            max_workers: 最大工作线程数，None 表示自动选择
            parallel: 是否并行执行，False 表示串行（调试用）
            name: 节点名称
            config: 配置字典
        """
        super().__init__(name=name or "MapNode", config=config)
        self.node = node
        if group_by is None:
            self.group_by_expr = None
        elif isinstance(group_by, str):
            self.group_by_expr = group_by
        else:
            self.group_by_expr = _wrap_condition(group_by)
        self.max_workers = max_workers
        self.parallel = parallel
        self.logger = logging.getLogger(f"node.{self.node_id}")

    def _execute(self, input_data: Any = None, **kwargs) -> List[O]:
        """分组执行"""
        groups = self._group(input_data)
        self.logger.debug(f"Split into {len(groups)} groups")

        if self.parallel:
            results = self._execute_parallel(groups, **kwargs)
        else:
            results = self._execute_serial(groups, **kwargs)

        return results

    def _group(self, input_data: Any) -> List[tuple[Any, Any]]:
        """对输入数据进行分组"""
        if isinstance(input_data, pd.DataFrame):
            if isinstance(self.group_by_expr, str):
                return list(input_data.groupby(self.group_by_expr))
            elif isinstance(self.group_by_expr, Expression):
                keys = self.group_by_expr.evaluate(input_data)
                return list(input_data.groupby(keys))
            else:
                raise ValueError(f"Unsupported group_by type: {type(self.group_by_expr)}")
        else:
            if isinstance(input_data, (list, tuple)):
                if isinstance(self.group_by_expr, Expression):
                    from itertools import groupby
                    sorted_data = sorted(input_data, key=self.group_by_expr.evaluate)
                    return [
                        (k, list(v))
                        for k, v in groupby(sorted_data, key=self.group_by_expr.evaluate)
                    ]
                else:
                    return [(i, item) for i, item in enumerate(input_data)]
            else:
                return [(None, input_data)]

    def _execute_serial(self, groups: List[tuple[Any, Any]], **kwargs) -> List[O]:
        """串行执行"""
        results = []
        for key, group in groups:
            self.logger.debug(f"Processing group: {key}")
            result = self.node.execute(group, **kwargs)
            results.append((key, result))
        return results

    def _execute_parallel(self, groups: List[tuple[Any, Any]], **kwargs) -> List[O]:
        """并行执行"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_key = {
                executor.submit(self.node.execute, group, **kwargs): key
                for key, group in groups
            }

            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    result = future.result()
                    results.append((key, result))
                except Exception as e:
                    self.logger.error(f"Group '{key}' failed: {e}")
                    raise

        return results

    def _get_serializable_fields(self) -> Dict[str, Any]:
        """返回需要序列化的额外字段"""
        return {
            "node": self.node.serialize(),
            "group_by": (
                self.group_by_expr
                if isinstance(self.group_by_expr, str)
                else self.group_by_expr.serialize()
            ),
            "max_workers": self.max_workers,
            "parallel": self.parallel,
        }

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'MapNode':
        """从字典反序列化重建 MapNode"""
        from QuantNodes.core.expression import Expression

        node = BaseNode.deserialize(data["node"])
        group_by = data["group_by"]
        if isinstance(group_by, dict):
            group_by = Expression.deserialize(group_by)

        return MapNode(
            node=node,
            group_by=group_by,
            max_workers=data.get("max_workers"),
            parallel=data.get("parallel", True),
            name=data.get("name"),
            config=data.get("config", {})
        )

    def to_info(self) -> Dict[str, Any]:
        """导出节点信息"""
        result = super().to_info()
        result['group_by'] = (
            repr(self.group_by_expr)
            if isinstance(self.group_by_expr, Expression)
            else self.group_by_expr
        )
        result['node'] = self.node.to_info()
        result['max_workers'] = self.max_workers
        result['parallel'] = self.parallel
        return result


@serializable
class WhileNode(BaseNode):
    """
    条件循环节点

    只要条件满足，就循环执行 body 节点。
    每次迭代的输出作为下一次迭代的输入。

    Examples:
        >>> # DSL 方式
        >>> WhileNode(
        ...     condition=Cond.attr('metrics').sharpe < 1.5,
        ...     body=ParameterTuningNode(),
        ...     max_iterations=10,
        ... )
        >>>
        >>> # 字符串表达式方式
        >>> WhileNode("result.metrics.sharpe < 1.5", ParameterTuningNode(), 10)
    """

    def __init__(self,
                 condition: Union[Expression, ExpressionBuilder, Callable[[Any], bool], str],
                 body: BaseNode,
                 max_iterations: int = 1000,
                 name: str = None,
                 config: Dict[str, Any] = None):
        """
        Args:
            condition: 循环继续条件，接收当前结果返回 bool
            body: 循环体节点
            max_iterations: 最大迭代次数，防止死循环
            name: 节点名称
            config: 配置字典
        """
        super().__init__(name=name or "WhileNode", config=config)
        self.condition = _wrap_condition(condition)
        self.body = body
        self.max_iterations = max_iterations
        self._iteration_count: int = 0
        self.logger = logging.getLogger(f"node.{self.node_id}")

    def _execute(self, input_data: Any = None, **kwargs) -> Any:
        """循环执行"""
        result = input_data
        self._iteration_count = 0

        while self.condition.evaluate(result) and self._iteration_count < self.max_iterations:
            self.logger.debug(f"Iteration {self._iteration_count + 1}/{self.max_iterations}")
            result = self.body.execute(result, **kwargs)
            self._iteration_count += 1

        if self._iteration_count >= self.max_iterations:
            self.logger.warning(
                f"WhileNode reached max iterations {self.max_iterations}, "
                f"stopping early. Last condition result: {self.condition.evaluate(result)}"
            )

        return result

    @property
    def iteration_count(self) -> int:
        """返回最后一次执行的迭代次数"""
        return self._iteration_count

    def _get_serializable_fields(self) -> Dict[str, Any]:
        """返回需要序列化的额外字段"""
        if isinstance(self.condition, LambdaExpression):
            raise SerializationError(
                "WhileNode with lambda condition cannot be serialized. "
                "Please use Cond() DSL or string expression instead."
            )
        return {
            "condition": self.condition.serialize(),
            "body": self.body.serialize(),
            "max_iterations": self.max_iterations,
        }

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'WhileNode':
        """从字典反序列化重建 WhileNode"""
        from QuantNodes.core.expression import Expression

        condition = Expression.deserialize(data["condition"])
        body = BaseNode.deserialize(data["body"])

        return WhileNode(
            condition=condition,
            body=body,
            max_iterations=data["max_iterations"],
            name=data.get("name"),
            config=data.get("config", {})
        )

    def to_info(self) -> Dict[str, Any]:
        """导出节点信息"""
        result = super().to_info()
        result['condition'] = repr(self.condition)
        result['condition_dict'] = self.condition.serialize()
        result['body'] = self.body.to_info()
        result['max_iterations'] = self.max_iterations
        result['last_iteration_count'] = self._iteration_count
        return result
