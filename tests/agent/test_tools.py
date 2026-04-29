# coding=utf-8
"""
测试工具系统
"""

import asyncio
from QuantNodes.agent.tools import ToolRegistry, EchoTool
from QuantNodes.agent.tools.base import Tool


class TestEchoTool:
    def test_name(self):
        tool = EchoTool()
        assert tool.name == "echo"

    def test_parameters(self):
        tool = EchoTool()
        assert "message" in tool.parameters["properties"]

    def test_execute(self):
        async def _test():
            tool = EchoTool()
            result = await tool.execute(message="hello")
            assert result == "hello"

        asyncio.run(_test())


class TestToolRegistry:
    def test_register(self):
        registry = ToolRegistry()
        registry.register(EchoTool())
        assert len(registry.list_tools()) == 1

    def test_get(self):
        registry = ToolRegistry()
        tool = EchoTool()
        registry.register(tool)
        assert registry.get("echo") == tool

    def test_execute_tool(self):
        async def _test():
            registry = ToolRegistry()
            registry.register(EchoTool())
            result = await registry.execute_tool("echo", message="test")
            assert result.success is True
            assert result.content == "test"

        asyncio.run(_test())

    def test_execute_missing_tool(self):
        async def _test():
            registry = ToolRegistry()
            result = await registry.execute_tool("nonexistent", message="test")
            assert result.success is False
            assert "not found" in result.error

        asyncio.run(_test())

    def test_validation_missing_param(self):
        async def _test():
            registry = ToolRegistry()
            registry.register(EchoTool())
            result = await registry.execute_tool("echo")
            assert result.success is False
            assert "Missing required parameter" in result.error

        asyncio.run(_test())
