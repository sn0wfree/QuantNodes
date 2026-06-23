# coding=utf-8
"""
工具注册表

管理工具的注册、查询与执行
"""

from typing import Any, Dict, List
import asyncio

from .base import Tool, ToolExecutionResult


class ToolRegistry:
    """工具注册表与执行入口"""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._cached_schemas: List[Dict[str, Any]] | None = None

    def register(self, tool: Tool) -> None:
        """注册工具"""
        self._tools[tool.name] = tool
        self._cached_schemas = None

    def unregister(self, name: str) -> None:
        """注销工具"""
        if name in self._tools:
            del self._tools[name]
            self._cached_schemas = None

    def get(self, name: str) -> Tool | None:
        """获取工具"""
        return self._tools.get(name)

    def list_tools(self) -> List[Tool]:
        """列出所有已注册工具"""
        return list(self._tools.values())

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI function-calling schemas for all tools (cached).

        v3.0.0 contract: if a tool cannot produce a schema (e.g. the
        local ``Tool`` ABC when ``nanobot-ai`` is not installed), we
        fall back to a minimal ``{"type": "function", "function": {"name": ..., "description": ..., "parameters": {"type": "object", "properties": {}}}}``
        shape so the registry's schema list is always usable.
        """
        if self._cached_schemas is None:
            schemas: List[Dict[str, Any]] = []
            for tool in self._tools.values():
                try:
                    schemas.append(tool.to_openai_schema())
                except Exception:
                    # Fallback: minimal schema from raw fields
                    schemas.append({
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.parameters,
                        },
                    })
            self._cached_schemas = schemas
        return self._cached_schemas

    async def execute_tool(self, name: str, **kwargs: Any) -> ToolExecutionResult:
        """Execute a single tool by name.

        v3.0.0 contract: we attempt to call ``tool.cast_params(kwargs)`` and
        ``tool.validate_params(params)`` for forward-compat with the
        upstream nanobot Tool API. If those methods don't exist on the
        tool (e.g. local ``Tool`` ABC when ``nanobot-ai`` is not
        installed, or third-party tools that don't implement them), we
        gracefully fall back to passing the raw ``kwargs`` to
        ``tool.execute()``.
        """
        tool = self._tools.get(name)
        if not tool:
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                content=None,
                error=f"Tool '{name}' not found"
            )

        try:
            # Optional pre-processing hooks. Missing methods are OK —
            # the bare v3.0.0 ``Tool`` ABC doesn't define them.
            params: Dict[str, Any] = kwargs
            cast = getattr(tool, "cast_params", None)
            if callable(cast):
                params = cast(kwargs)
            validate = getattr(tool, "validate_params", None)
            if callable(validate):
                errors = validate(params)
                if errors:
                    return ToolExecutionResult(
                        tool_name=name,
                        success=False,
                        content=None,
                        error=f"Parameter validation failed: {', '.join(errors)}"
                    )

            result = await tool.execute(**params)
            return ToolExecutionResult(
                tool_name=name,
                success=True,
                content=result
            )
        except Exception as e:
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                content=None,
                error=str(e)
            )

    async def execute_tools_parallel(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[ToolExecutionResult]:
        """并发执行多个工具（只读工具并发，有副作用工具串行）"""
        results: List[ToolExecutionResult] = []

        # 分组：只读工具并发执行，其他串行
        read_only_calls: List[Dict[str, Any]] = []
        write_calls: List[Dict[str, Any]] = []

        for call in tool_calls:
            tool = self._tools.get(call.get("name", ""))
            if tool and tool.read_only:
                read_only_calls.append(call)
            else:
                write_calls.append(call)

        # 并发执行只读工具
        if read_only_calls:
            tasks = [
                self.execute_tool(call["name"], **call.get("arguments", {}))
                for call in read_only_calls
            ]
            results.extend(await asyncio.gather(*tasks))

        # 串行执行有副作用工具
        for call in write_calls:
            name = call.get("name", "")
            result = await self.execute_tool(name, **call.get("arguments", {}))
            results.append(result)

        return results
