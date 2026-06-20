# coding=utf-8
"""
LambdaNode 便捷节点类

用于快速将函数包装为节点，适合简单的转换逻辑。
"""

from typing import Any, Callable, Dict
from QuantNodes.core.node import BaseNode, SerializationError
from QuantNodes.core.serializable import serializable


@serializable
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

    def __init__(
        self, func: Callable[[Any, Any], Any], name: str = None,
        config: Dict[str, Any] = None,
    ):
        """
        Args:
            func: 执行函数，签名为 func(input_data, context) -> result
            name: 节点名称，默认为函数名
            config: 配置字典
        """
        self.func = func
        name = name or (func.__name__ if hasattr(func, '__name__') else 'Lambda')
        super().__init__(name=name, config=config)

    def _execute(self, input_data: Any = None, **kwargs) -> Any:
        return self.func(input_data, self)

    def _get_serializable_fields(self) -> Dict[str, Any]:
        """返回需要序列化的额外字段"""
        func = self.func

        if hasattr(func, "__name__") and func.__name__ == "<lambda>":
            raise SerializationError(
                "LambdaNode with anonymous lambda cannot be serialized. "
                "Please use a named function instead."
            )

        return {
            "func": {
                "type": "named_function",
                "module": func.__module__,
                "qualname": func.__qualname__
            }
        }

    @classmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> 'LambdaNode':
        """从字典反序列化重建 LambdaNode"""
        import importlib

        func_info = data["func"]
        if func_info["type"] != "named_function":
            raise ValueError(f"Unsupported function type: {func_info['type']}")

        module = importlib.import_module(func_info["module"])
        func = getattr(module, func_info["qualname"])

        return LambdaNode(
            func=func,
            name=data.get("name"),
            config=data.get("config", {})
        )
