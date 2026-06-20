# coding=utf-8
"""
Tool基类与Schema验证
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ToolExecutionResult:
    """工具执行结果"""
    tool_name: str
    success: bool
    content: Any
    error: str | None = None


class Tool(ABC):
    """所有工具的抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称 (snake_case)"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具自然语言描述"""
        pass

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema参数定义"""
        pass

    @property
    def read_only(self) -> bool:
        """是否为只读工具（可并发执行）"""
        return False

    @property
    def concurrency_safe(self) -> bool:
        """是否并发安全"""
        return True

    def cast_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """参数类型转换"""
        return params

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """验证参数，返回错误列表（空=有效）"""
        errors: List[str] = []
        schema = self.parameters

        if schema.get("type") == "object":
            required = schema.get("required", [])
            for key in required:
                if key not in params:
                    errors.append(f"Missing required parameter: {key}")

        return errors

    def to_openai_schema(self) -> Dict[str, Any]:
        """转换为OpenAI Function Call格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

    async def _dispatch(self, action: str, registry: Dict[str, Any], **kwargs: Any) -> Any:
        """Look up `action` in `registry` and call it with kwargs (Phase J2).

        Replaces the 4-times-repeated:
            fn = dispatch.get(action)
            if not fn: raise ValueError(...)
            return await fn(**kwargs)

        Subclasses call `return await self._dispatch(action, {...})` from execute().
        """
        fn = registry.get(action)
        if not fn:
            raise ValueError(f"Unknown action: {action}")
        return await fn(**kwargs)

    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """执行工具，返回字符串或内容块列表"""
        pass
