# coding=utf-8
"""QuantNodes.agent.tools.echo 单元测试"""
import pytest

from QuantNodes.agent.tools.echo import EchoTool


class TestEchoTool:
    def test_name_is_echo(self):
        tool = EchoTool()
        assert tool.name == "echo"

    def test_description_is_set(self):
        tool = EchoTool()
        assert "测试" in tool.description or "echo" in tool.description.lower()

    def test_parameters_has_message_field(self):
        tool = EchoTool()
        params = tool.parameters
        assert "message" in params["properties"]

    def test_message_field_is_string_type(self):
        tool = EchoTool()
        params = tool.parameters
        assert params["properties"]["message"]["type"] == "string"

    def test_message_field_is_required(self):
        tool = EchoTool()
        params = tool.parameters
        assert "message" in params["required"]

    def test_read_only_is_true(self):
        tool = EchoTool()
        assert tool.read_only is True

    @pytest.mark.asyncio
    async def test_execute_returns_message(self):
        tool = EchoTool()
        result = await tool.execute(message="Hello World")
        assert result == "Hello World"

    @pytest.mark.asyncio
    async def test_execute_with_empty_string(self):
        tool = EchoTool()
        result = await tool.execute(message="")
        assert result == ""

    @pytest.mark.asyncio
    async def test_execute_with_special_characters(self):
        tool = EchoTool()
        result = await tool.execute(message="Hello\nWorld\t!")
        assert result == "Hello\nWorld\t!"
