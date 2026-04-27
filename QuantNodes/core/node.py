# coding=utf-8
"""
统一节点架构核心模块

本模块定义了所有节点的统一基类 BaseNode，以及节点状态枚举和统计类。
"""

from __future__ import annotations

import uuid
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, TypeVar, Generic, Callable, List

from QuantNodes.core.base import QuantNodesBase, QuantNodesError, ValidationError


T = TypeVar('T')  # 输入类型
R = TypeVar('R')  # 输出类型


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


class BaseNode(QuantNodesBase, ABC, Generic[T, R]):
    """
    所有节点的统一基类

    核心契约：input_data -> execute() -> output_data

    子类必须实现：
        _execute(input_data, **kwargs): 核心执行逻辑

    子类可覆盖：
        ConfigSchema: 配置模型类
        InputSchema: 输入数据校验模型
        OutputSchema: 输出数据校验模型
        before_execute(): 执行前钩子
        after_execute(): 执行后钩子
        validate(): 配置校验

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

        # 节点唯一标识
        self.node_id = f"{self.name}_{uuid.uuid4().hex[:8]}"

        # 状态管理
        self.state: NodeState = NodeState.IDLE
        self._last_error: Optional[Exception] = None
        self._last_result: Optional[R] = None
        self._last_input: Optional[T] = None

        # 配置处理
        self.config: Dict[str, Any] = {**(config or {}), **kwargs}

        # 日志器
        self.logger = logging.getLogger(f"node.{self.node_id}")

        # 执行统计
        self.stats: Optional[NodeStats] = NodeStats() if self._enable_stats else None

        # 简单缓存（仅缓存上一次的输入输出）
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

    def to_dict(self) -> Dict[str, Any]:
        """导出节点信息"""
        return {
            'node_id': self.node_id,
            'name': self.name,
            'class': self.__class__.__name__,
            'state': self.state.value,
            'config': self.config,
            'stats': self.stats.to_dict() if self.stats else None,
        }

    def copy(self) -> 'BaseNode[T, R]':
        """创建节点的副本"""
        return self.__class__(name=self.name, config=self.config.copy())
