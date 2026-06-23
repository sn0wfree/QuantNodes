# coding=utf-8
"""Tool base class — inherits from HKUDS nanobot 0.2.1 ``Tool``.

v3.0.0 迁移到上游后，quant 工具的父类由本地 ``Tool`` 改为
``nanobot.agent.tools.base.Tool``。本文件保留薄包装以兼容 15 个 quant
工具的 ``from .base import Tool`` 导入路径。
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List

from nanobot.agent.tools.base import Tool as _NanobotTool


@dataclass
class ToolExecutionResult:
    """工具执行结果（向后兼容）。"""

    tool_name: str
    success: bool
    content: Any
    error: str | None = None


class Tool(_NanobotTool, ABC):
    """所有 quant 工具的薄包装父类。

    行为完全继承自 nanobot 的 ``Tool``，仅补充：
    - ``to_openai_schema`` 别名（与 ``to_schema`` 同语义，保留旧名）
    - ``_dispatch`` 辅助方法（量化工具内部 action 分发用）
    """

    @property
    def to_openai_schema(self) -> Any:
        """向后兼容别名 — 调用上游 ``to_schema``。"""
        return self.to_schema

    async def _dispatch(self, action: str, registry: Dict[str, Any], **kwargs: Any) -> Any:
        """Look up ``action`` in ``registry`` and call it with kwargs.

        Replaces 4-times-repeated::

            fn = dispatch.get(action)
            if not fn: raise ValueError(...)
            return await fn(**kwargs)

        Subclasses call ``return await self._dispatch(action, {...})`` from execute().
        """
        fn = registry.get(action)
        if not fn:
            raise ValueError(f"Unknown action: {action}")
        return await fn(**kwargs)


__all__ = ["Tool", "ToolExecutionResult"]
