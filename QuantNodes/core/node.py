# coding=utf-8
"""
统一节点架构核心模块

本模块定义了所有节点的统一基类 BaseNode，以及节点状态枚举和统计类。

序列化设计：
- serialize() / deserialize(): 纯逻辑序列化，用于保存/重建
- to_info(): 运行时信息导出，用于监控/调试
- @register_node: 节点类注册装饰器，用于反序列化
"""

from __future__ import annotations

import uuid
import logging
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, TypeVar, Generic, Callable, List, Type

from QuantNodes.core.base import QuantNodesBase, QuantNodesError, ValidationError


T = TypeVar('T')  # 输入类型
R = TypeVar('R')  # 输出类型


# ============================================================================
# 节点类注册表
# ============================================================================

_NODE_CLASSES: Dict[str, Type['BaseNode']] = {}


def register_node(cls: Type['BaseNode']) -> Type['BaseNode']:
    """
    装饰器：注册节点类用于反序列化

    用法：
        @register_node
        class MyNode(BaseNode):
            ...

    所有需要支持反序列化的节点类都必须使用此装饰器。
    """
    _NODE_CLASSES[cls.__name__] = cls
    return cls


class NodeState(str, Enum):
    """节点状态枚举"""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class NodeStats:
    """节点执行统计"""

    __slots__ = [
        'execute_count',
        'success_count',
        'failed_count',
        'total_time_ms',
        'avg_time_ms',
        'last_execute_at',
    ]

    def __init__(self):
        self.execute_count: int = 0
        self.success_count: int = 0
        self.failed_count: int = 0
        self.total_time_ms: float = 0.0
        self.avg_time_ms: float = 0.0
        self.last_execute_at: Optional[datetime] = None

    def update(self, elapsed_ms: float, success: bool) -> None:
        """更新统计数据"""
        self.execute_count += 1
        if success:
            self.success_count += 1
        else:
            self.failed_count += 1
        self.total_time_ms += elapsed_ms
        if self.execute_count > 0:
            self.avg_time_ms = self.total_time_ms / self.execute_count
        self.last_execute_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'execute_count': self.execute_count,
            'success_count': self.success_count,
            'failed_count': self.failed_count,
            'total_time_ms': self.total_time_ms,
            'avg_time_ms': self.avg_time_ms,
            'last_execute_at': self.last_execute_at.isoformat() if self.last_execute_at else None,
        }


class NodeExecutionError(QuantNodesError):
    """节点执行异常"""
    code = "NODE_EXECUTION_ERROR"

    def __init__(self, node_name: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.node_name = node_name


class SerializationError(QuantNodesError):
    """节点序列化异常"""
    code = "SERIALIZATION_ERROR"


class BaseNode(QuantNodesBase, ABC, Generic[T, R]):
    """
    所有节点的统一基类

    核心契约：input_data -> execute() -> output_data

    子类必须实现：
        _execute(input_data, **kwargs): 核心执行逻辑
        _from_dict_impl(data): 反序列化实现

    子类可覆盖：
        ConfigSchema: 配置模型类
        InputSchema: 输入数据校验模型
        OutputSchema: 输出数据校验模型
        before_execute(): 执行前钩子
        after_execute(): 执行后钩子
        validate(): 配置校验
        _get_serializable_fields(): 序列化扩展

    配置开关（类属性）：
        _enable_validation: 是否启用配置校验
        _enable_stats: 是否启用执行统计
        _enable_cache: 是否启用结果缓存
        _enable_hooks: 是否启用钩子函数
    """

    # 子类可覆盖的配置开关
    _enable_validation: bool = True
    _enable_stats: bool = True
    _enable_cache: bool = False
    _enable_hooks: bool = True

    def __init__(self, name: str = None, config: Dict[str, Any] = None, **kwargs):
        """
        节点初始化

        Args:
            name: 节点名称，默认为类名
            config: 配置字典
            **kwargs: 额外配置参数，与 config 合并
        """
        super().__init__(name=name)

        self.node_id = f"{self.name}_{uuid.uuid4().hex[:8]}"

        self.state: NodeState = NodeState.IDLE
        self._last_error: Optional[Exception] = None
        self._last_result: Optional[R] = None
        self._last_input: Optional[T] = None

        self.config: Dict[str, Any] = {**(config or {}), **kwargs}

        self.logger = logging.getLogger(f"node.{self.node_id}")

        self.stats: Optional[NodeStats] = NodeStats() if self._enable_stats else None

        self._cache: Dict[str, R] = {}

    @abstractmethod
    def _execute(self, input_data: T = None, **kwargs) -> R:
        """
        子类实现的核心执行逻辑

        Args:
            input_data: 输入数据
            **kwargs: 额外执行参数

        Returns:
            执行结果
        """
        pass

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'BaseNode':
        """
        子类实现的具体反序列化逻辑

        默认实现使用 data 中的 name 和 config 重建节点。
        如果节点有额外参数（如 Pipeline 的 nodes），子类必须重写此方法。

        Args:
            data: 序列化字典

        Returns:
            重建的节点实例
        """
        return cls(name=data.get("name"), config=data.get("config", {}))

    # =========================================================================
    # 序列化接口（用于保存/重建）
    # =========================================================================

    def serialize(self) -> Dict[str, Any]:
        """
        序列化节点配置（不含运行时数据）

        用于保存到文件/数据库/传输，以及重建节点。
        不包含运行时数据如 node_id, state, stats 等。

        Returns:
            包含节点配置的字典
        """
        result = {
            "type": self.__class__.__name__,
            "name": self.name,
            "config": self._filter_config(self.config),
            "_schema_version": "1.0",
        }
        result.update(self._get_serializable_fields())
        return result

    def _filter_config(self, config: Dict) -> Dict:
        """
        过滤配置字典中的不可序列化项

        子类可重写以排除特定的配置项。
        """
        return config.copy() if config else {}

    def _get_serializable_fields(self) -> Dict[str, Any]:
        """
        子类扩展：返回需要序列化的额外字段

        默认实现返回空字典。复合节点需要重写此方法
        以包含子节点等额外字段。

        Returns:
            额外序列化字段字典
        """
        return {}

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'BaseNode':
        """
        从字典反序列化重建节点

        Args:
            data: 序列化字典

        Returns:
            重建的节点实例

        Raises:
            ValueError: 未知节点类型或数据格式错误
        """
        schema_version = data.get("_schema_version", "0.0")

        node_type = data.get("type")
        if not node_type:
            raise ValueError("Missing 'type' in serialized data")

        node_cls = _NODE_CLASSES.get(node_type)
        if not node_cls:
            raise ValueError(
                f"Unknown node type: {node_type}. "
                f"Available types: {list(_NODE_CLASSES.keys())}"
            )

        return node_cls._from_dict_impl(data)

    # =========================================================================
    # 信息导出接口（用于监控/调试）
    # =========================================================================

    def to_info(self) -> Dict[str, Any]:
        """
        导出运行时信息（用于监控/调试）

        包含 node_id, state, stats 等运行时数据，
        不适合用于序列化重建。

        Returns:
            包含运行时信息的字典
        """
        return {
            'node_id': self.node_id,
            'name': self.name,
            'class': self.__class__.__name__,
            'state': self.state.value,
            'config': self.config,
            'stats': self.stats.to_dict() if self.stats else None,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        导出节点信息（兼容性方法）

        警告：此方法用于监控/调试目的。
        如需序列化/重建，请使用 serialize() 方法。

        Returns:
            包含节点信息的字典
        """
        warnings.warn(
            "to_dict() is for monitoring/debugging. "
            "Use serialize() for serialization.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.to_info()

    def execute(self, input_data: T = None, *, validate_input: bool = None, **kwargs) -> R:
        """
        统一执行入口

        Args:
            input_data: 输入数据
            validate_input: 是否校验输入，None 则使用类默认配置
            **kwargs: 执行参数

        Returns:
            执行结果

        Raises:
            NodeExecutionError: 节点执行失败
        """
        start_time = datetime.now()
        self.state = NodeState.RUNNING
        self._last_error = None
        self._last_input = input_data

        try:
            # 1. 前置钩子
            if self._enable_hooks:
                self.before_execute(input_data, **kwargs)

            # 2. 输入校验（可选）
            should_validate = validate_input if validate_input is not None else self._enable_validation
            if should_validate:
                self._validate_input(input_data)

            # 3. 核心执行
            result = self._execute(input_data, **kwargs)

            # 4. 输出校验（可选）
            if should_validate:
                self._validate_output(result)

            self.state = NodeState.SUCCESS
            self._last_result = result
            return result

        except Exception as e:
            self.state = NodeState.FAILED
            self._last_error = e
            self._handle_error(e, input_data, **kwargs)

        finally:
            # 5. 统计更新
            if self._enable_stats:
                elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
                self.stats.update(elapsed_ms, success=(self.state == NodeState.SUCCESS))

            # 6. 后置钩子（无论成功失败）
            if self._enable_hooks:
                self.after_execute(result if self.state == NodeState.SUCCESS else None, **kwargs)

    def __call__(self, input_data: T = None, **kwargs) -> R:
        """使得节点可以像函数一样调用"""
        return self.execute(input_data, **kwargs)

    def __rshift__(self, other: 'BaseNode') -> 'Pipeline':
        """管道运算符：A >> B 等价于 Pipeline([A, B])"""
        from QuantNodes.core.pipeline import Pipeline
        if isinstance(self, Pipeline):
            return Pipeline(self.nodes + [other])
        return Pipeline([self, other])

    # --- 生命周期钩子 ---

    def before_execute(self, input_data: T, **kwargs) -> None:
        """执行前钩子，子类可重写"""
        pass

    def after_execute(self, result: Optional[R], **kwargs) -> None:
        """执行后钩子，子类可重写"""
        pass

    def validate(self) -> bool:
        """配置校验，子类可重写扩展"""
        return True

    # --- 内部辅助方法 ---

    def _validate_input(self, input_data: T) -> None:
        """输入数据校验，子类可重写"""
        pass

    def _validate_output(self, result: R) -> None:
        """输出数据校验，子类可重写"""
        pass

    def _handle_error(self, error: Exception, input_data: T, **kwargs) -> None:
        """统一错误处理"""
        self.logger.error(
            "Node %s execution failed: %s",
            self.node_id,
            str(error),
            exc_info=True,
            extra={
                'node_name': self.name,
                'input_type': type(input_data).__name__,
                'kwargs_keys': list(kwargs.keys())
            }
        )

        # 重新抛出包装后的异常
        if not isinstance(error, QuantNodesError):
            raise NodeExecutionError(
                node_name=self.name,
                message=str(error),
                details={
                    'error_type': type(error).__name__,
                    'node_id': self.node_id,
                    **kwargs
                }
            ) from error
        raise

    # --- 便捷方法 ---

    def reset(self) -> None:
        """重置节点状态"""
        self.state = NodeState.IDLE
        self._last_error = None
        self._last_result = None
        self._last_input = None
        if self._enable_stats:
            self.stats = NodeStats()
        self._cache.clear()

    def copy(self) -> 'BaseNode[T, R]':
        """创建节点的副本"""
        return self.__class__(name=self.name, config=self.config.copy())
