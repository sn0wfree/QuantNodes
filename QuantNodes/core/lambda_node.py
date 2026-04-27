# coding=utf-8
"""
LambdaNode 便捷节点类

用于快速将函数包装为节点，适合简单的转换逻辑。
"""

from typing import Any, Callable
from QuantNodes.core.node import BaseNode


class LambdaNode(BaseNode):
    """
    将函数包装为节点

    Examples:
        >>> # 简单函数
        >>> add_one = LambdaNode(lambda x: x + 1)
        >>> add_one.execute(5)  # 6
        >>>
        >>> # 更复杂的逻辑
        >>> rolling_mean = LambdaNode(
        ...     lambda df, ctx: df.rolling(ctx.config['window']).mean(),
        ...     config={'window': 20}
        ... )
    """

    def __init__(self, func: Callable[[Any, Any], Any], name: str = None):
        """
        Args:
            func: 执行函数，签名为 func(input_data, context) -> result
            name: 节点名称，默认为函数名
        """
        self.func = func
        name = name or (func.__name__ if hasattr(func, '__name__') else 'Lambda')
        super().__init__(name=name)

    def _execute(self, input_data: Any = None, **kwargs) -> Any:
        return self.func(input_data, self)
