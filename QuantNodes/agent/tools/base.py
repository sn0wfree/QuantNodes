# coding=utf-8
"""Tool base class — inherits from HKUDS nanobot 0.2.1 ``Tool``.

v3.0.0 迁移到上游后，quant 工具的父类由本地 ``Tool`` 改为
``nanobot.agent.tools.base.Tool``。本文件保留薄包装以兼容 14 个 quant
工具的 ``from .base import Tool`` 导入路径。

**v3.0.0 Stage 5.3** 起，``nanobot-ai`` 为可选依赖。本文件在未装 extras
时提供本地降级父类 ``Tool``，让 quant 工具可在没有 nanobot 的环境下
独立使用（CLI、API、Wiki 等场景不需要 agent）。当用户调用 ``Agent(...)``
或注册工具到 nanobot registry 时才会触发真正的 nanobot import。
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict

# Note: do NOT ``from QuantNodes.agent import NANOBOT_AVAILABLE`` here — this
# file is imported as part of ``QuantNodes.agent.__init__`` (via
# ``nanobot_bridge.py -> tools/__init__.py -> base.py``) and a back-import
# would create a circular dependency with partially-initialized module state.
# Instead we probe nanobot here, independently.
try:
    from nanobot.agent.tools.base import Tool as _NanobotTool

    _NANOBOT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in nanobot-less envs
    _NanobotTool = ABC  # type: ignore[assignment,misc]
    _NANOBOT_AVAILABLE = False


@dataclass
class ToolExecutionResult:
    """工具执行结果（向后兼容）。"""

    tool_name: str
    success: bool
    content: Any
    error: str | None = None


if _NANOBOT_AVAILABLE:  # pragma: no cover - exercised in [agent] installs
    class Tool(_NanobotTool, ABC):  # type: ignore[misc]
        """所有 quant 工具的薄包装父类 — 继承自 nanobot 的 ``Tool``。

        仅补充：
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
else:

    class Tool(ABC):  # type: ignore[no-redef]
        """Stand-alone Tool ABC for quant-only deployments (no nanobot).

        When ``nanobot-ai`` is not installed, quant tools still need a
        concrete parent class so that ``class MyTool(Tool)`` works. This
        is the lightest possible stand-in: just enough to allow
        ``from .base import Tool`` and the ``_dispatch`` helper.

        Calls to ``to_openai_schema`` raise a clear error guiding the
        user to install the [agent] extra. Tool registration with
        nanobot's ToolRegistry is also unavailable — agents that need
        registration must install ``quantnodes[agent]``.
        """

        name: str = ""
        description: str = ""
        parameters: Dict[str, Any] = {}

        @property
        def to_openai_schema(self) -> Any:
            """Raises — only available with ``nanobot-ai`` installed."""
            raise RuntimeError(
                "to_openai_schema requires nanobot-ai. "
                "Install: pip install 'quantnodes[agent]'"
            )

        async def execute(self, **kwargs: Any) -> Any:  # pragma: no cover
            raise RuntimeError(
                "Tool.execute() requires nanobot-ai. "
                "Install: pip install 'quantnodes[agent]'"
            )

        async def _dispatch(self, action: str, registry: Dict[str, Any], **kwargs: Any) -> Any:
            """Look up ``action`` in ``registry`` and call it with kwargs.

            Available even without nanobot, so quant tools' dispatcher
            patterns work in pure-quant deployments.
            """
            fn = registry.get(action)
            if not fn:
                raise ValueError(f"Unknown action: {action}")
            return await fn(**kwargs)


__all__ = ["Tool", "ToolExecutionResult"]
