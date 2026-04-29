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
        """获取所有工具的OpenAI Schema（带缓存）"""
        if self._cached_schemas is None:
            self._cached_schemas = [tool.to_openai_schema() for tool in self._tools.values()]
        return self._cached_schemas

    async def execute_tool(self, name: str, **kwargs: Any) -> ToolExecutionResult:
        """执行单个工具"""
        tool = self._tools.get(name)
        if not tool:
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                content=None,
                error=f"Tool '{name}' not found"
            )

        try:
            params = tool.cast_params(kwargs)
            errors = tool.validate_params(params)
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
            result = await self.execute_tool(call["name"], **call.get("arguments", {}))
            results.append(result)

        return results
