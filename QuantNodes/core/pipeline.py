# coding=utf-8
"""
Pipeline 组合原语模块

本模块提供三种核心的节点组合方式：
1. Pipeline: 线性管道，按顺序执行节点列表
2. Parallel: 并行分叉，所有分支接收相同输入，并行执行
3. Join: 聚合组合，接收字典输入，通过用户函数聚合为单个输出

使用方式：
    >>> from QuantNodes.core import Pipeline, Parallel, Join
    >>>
    >>> # 线性管道
    >>> p = DatabaseNode() >> FactorNode() >> BacktestNode()
    >>> result = p.execute(data)
    >>>
    >>> # 并行分叉
    >>> factors = Parallel({
    ...     'mom': MomentumFactor(),
    ...     'vol': VolatilityFactor(),
    ...     'value': ValueFactor(),
    ... })
    >>>
    >>> # 聚合组合
    >>> combine = Join(lambda mom, vol, value: mom * 0.5 + vol * 0.3 + value * 0.2)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Callable, Optional, Union
from functools import reduce

from QuantNodes.core.node import BaseNode, NodeState, register_node, SerializationError


@register_node
class Pipeline(BaseNode):
    """
    线性管道节点

    按顺序执行节点列表，前一个节点的输出作为后一个节点的输入。
    Pipeline 本身也是一个 Node，可以嵌套使用。

    Examples:
        >>> # 两种创建方式等价
        >>> p1 = Pipeline([NodeA(), NodeB(), NodeC()])
        >>> p2 = NodeA() >> NodeB() >> NodeC()
        >>>
        >>> # 嵌套使用
        >>> nested = Pipeline([
        ...     LoadDataNode(),
        ...     Pipeline([CleanDataNode(), NormalizeNode()]),
        ...     SaveResultNode(),
        ... ])
    """

    def __init__(self, nodes: List[BaseNode], name: str = None, config: Dict[str, Any] = None):
        """
        Args:
            nodes: 节点列表，按顺序执行
            name: 管道名称，默认为 "Pipeline"
            config: 配置字典
        """
        super().__init__(name=name or "Pipeline", config=config)
        self.nodes: List[BaseNode] = list(nodes)
        self.logger = logging.getLogger(f"node.{self.node_id}")

    def _execute(self, input_data: Any = None, **kwargs) -> Any:
        """依次执行所有节点"""
        result = input_data

        for i, node in enumerate(self.nodes):
            self.logger.debug(f"Executing pipeline node {i+1}/{len(self.nodes)}: {node.name}")
            result = node.execute(result, **kwargs)

        return result

    def __rshift__(self, other: BaseNode) -> 'Pipeline':
        """重载 >> 运算符，支持链式扩展"""
        if isinstance(other, Pipeline):
            return Pipeline(self.nodes + other.nodes)
        return Pipeline(self.nodes + [other])

    def __iter__(self):
        """迭代器支持"""
        return iter(self.nodes)

    def __len__(self) -> int:
        """返回节点数量"""
        return len(self.nodes)

    def __getitem__(self, index: int) -> BaseNode:
        """支持索引访问"""
        return self.nodes[index]

    def _get_serializable_fields(self) -> Dict[str, Any]:
        """返回需要序列化的额外字段"""
        return {"nodes": [node.serialize() for node in self.nodes]}

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'Pipeline':
        """从字典反序列化重建 Pipeline"""
        nodes = [BaseNode.deserialize(n) for n in data["nodes"]]
        return Pipeline(
            nodes=nodes,
            name=data.get("name"),
            config=data.get("config", {})
        )

    def to_info(self) -> Dict[str, Any]:
        """导出管道信息"""
        result = super().to_info()
        result['nodes'] = [node.to_info() for node in self.nodes]
        return result


@register_node
class Parallel(BaseNode):
    """
    并行分叉节点

    所有分支接收相同的输入，并行执行，返回字典格式的结果。
    注意：当前版本为多线程实现，计算密集型任务可能不会提速。
    后续版本会支持多进程和分布式执行。

    Examples:
        >>> p = Parallel({
        ...     'factor_a': FactorA(),
        ...     'factor_b': FactorB(),
        ...     'factor_c': FactorC(),
        ... })
        >>> result = p.execute(data)
        >>> # result = {'factor_a': ..., 'factor_b': ..., 'factor_c': ...}
    """

    def __init__(self, branches: Dict[str, BaseNode],
                 name: str = None,
                 max_workers: Optional[int] = None,
                 parallel: bool = True,
                 config: Dict[str, Any] = None):
        """
        Args:
            branches: 分支节点字典，key 为结果字典的 key
            name: 节点名称
            max_workers: 最大工作线程数，None 表示自动选择
            parallel: 是否并行执行，False 表示串行（调试用）
            config: 配置字典
        """
        super().__init__(name=name or "Parallel", config=config)
        self.branches: Dict[str, BaseNode] = branches
        self.max_workers = max_workers
        self.parallel = parallel
        self.logger = logging.getLogger(f"node.{self.node_id}")

    def _execute(self, input_data: Any = None, **kwargs) -> Dict[str, Any]:
        """执行所有分支"""
        if self.parallel:
            return self._execute_parallel(input_data, **kwargs)
        else:
            return self._execute_serial(input_data, **kwargs)

    def _execute_serial(self, input_data: Any = None, **kwargs) -> Dict[str, Any]:
        """串行执行（调试用）"""
        results = {}
        for name, node in self.branches.items():
            self.logger.debug(f"Executing parallel branch: {name}")
            results[name] = node.execute(input_data, **kwargs)
        return results

    def _execute_parallel(self, input_data: Any = None, **kwargs) -> Dict[str, Any]:
        """并行执行（使用线程池）"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_name = {
                executor.submit(node.execute, input_data, **kwargs): name
                for name, node in self.branches.items()
            }

            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    self.logger.error(f"Parallel branch '{name}' failed: {e}")
                    raise

        return results

    def __or__(self, other: 'Parallel') -> 'Parallel':
        """重载 | 运算符，支持合并 Parallel 节点"""
        if not isinstance(other, Parallel):
            raise TypeError(f"Can only combine Parallel with Parallel, got {type(other)}")
        merged = {**self.branches, **other.branches}
        return Parallel(merged)

    def _get_serializable_fields(self) -> Dict[str, Any]:
        """返回需要序列化的额外字段"""
        return {
            "branches": {name: node.serialize() for name, node in self.branches.items()},
            "max_workers": self.max_workers,
            "parallel": self.parallel,
        }

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'Parallel':
        """从字典反序列化重建 Parallel"""
        branches = {
            name: BaseNode.deserialize(n)
            for name, n in data["branches"].items()
        }
        return Parallel(
            branches=branches,
            name=data.get("name"),
            max_workers=data.get("max_workers"),
            parallel=data.get("parallel", True),
            config=data.get("config", {})
        )

    def to_info(self) -> Dict[str, Any]:
        """导出节点信息"""
        result = super().to_info()
        result['branches'] = {name: node.to_info() for name, node in self.branches.items()}
        result['max_workers'] = self.max_workers
        result['parallel'] = self.parallel
        return result


@register_node
class Join(BaseNode):
    """
    聚合组合节点

    接收字典输入，通过用户定义的函数聚合为单个输出。
    通常与 Parallel 节点配合使用，将多个分支的结果合并。

    Examples:
        >>> # 1. 使用关键字参数函数
        >>> Join(lambda mom, vol, value: mom * 0.5 + vol * 0.3 + value * 0.2)
        >>>
        >>> # 2. 使用字典参数
        >>> Join(lambda factors: factors['mom'] + factors['vol'])
        >>>
        >>> # 3. 典型用法：Parallel + Join
        >>> pipeline = (
        ...     Parallel({
        ...         'mom': MomentumFactor(),
        ...         'vol': VolatilityFactor(),
        ...     })
        ...     >> Join(lambda mom, vol: mom / vol)
        ... )
    """

    def __init__(self, join_func: Callable, name: str = None, config: Dict[str, Any] = None):
        """
        Args:
            join_func: 聚合函数
                - 如果接受关键字参数，会传入字典的 key-value
                - 如果只接受一个参数，会传入整个字典
            name: 节点名称
            config: 配置字典
        """
        super().__init__(name=name or "Join", config=config)
        self.join_func = join_func
        self.logger = logging.getLogger(f"node.{self.node_id}")

    def _execute(self, input_data: Dict[str, Any], **kwargs) -> Any:
        """执行聚合"""
        if not isinstance(input_data, dict):
            raise ValueError(f"Join node requires dict input, got {type(input_data)}")

        import inspect
        sig = inspect.signature(self.join_func)
        params = list(sig.parameters.keys())

        if len(params) == 1:
            return self.join_func(input_data)

        return self.join_func(**input_data)

    def _get_serializable_fields(self) -> Dict[str, Any]:
        """返回需要序列化的额外字段"""
        func = self.join_func

        if hasattr(func, "__name__") and func.__name__ == "<lambda>":
            raise SerializationError(
                "Join node with anonymous lambda cannot be serialized. "
                "Please use a named function or Expression DSL instead."
            )

        return {
            "join_func": {
                "type": "named_function",
                "module": func.__module__,
                "qualname": func.__qualname__
            }
        }

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'Join':
        """从字典反序列化重建 Join"""
        import importlib

        func_info = data["join_func"]
        if func_info["type"] != "named_function":
            raise ValueError(f"Unsupported function type: {func_info['type']}")

        module = importlib.import_module(func_info["module"])
        func = getattr(module, func_info["qualname"])

        return Join(
            join_func=func,
            name=data.get("name"),
            config=data.get("config", {})
        )

    def to_info(self) -> Dict[str, Any]:
        """导出节点信息"""
        result = super().to_info()
        result['join_func'] = self.join_func.__name__ if hasattr(self.join_func, '__name__') else str(self.join_func)
        return result
