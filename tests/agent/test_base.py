# coding=utf-8
"""QuantNodes.agent.tools.base 单元测试"""
import pytest

from QuantNodes.agent.tools.base import Tool, ToolExecutionResult


class DummyTool(Tool):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "A dummy tool for testing"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
                "param2": {"type": "number"},
            },
            "required": ["param1"]
        }

    async def execute(self, param1: str, param2: float = 0, **kwargs):
        return f"executed: {param1}, {param2}"


class TestToolExecutionResult:
    def test_creation(self):
        result = ToolExecutionResult(
            tool_name="test",
            success=True,
            content={"key": "value"},
            error=None
        )
        assert result.tool_name == "test"
        assert result.success is True
        assert result.content == {"key": "value"}
        assert result.error is None

    def test_with_error(self):
        result = ToolExecutionResult(
            tool_name="test",
            success=False,
            content=None,
            error="Something went wrong"
        )
        assert result.success is False
        assert result.error == "Something went wrong"


class TestTool:
    def test_name_property(self):
        tool = DummyTool()
        assert tool.name == "dummy"

    def test_description_property(self):
        tool = DummyTool()
        assert "dummy" in tool.description.lower()

    def test_parameters_property(self):
        tool = DummyTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "param1" in params["required"]

    def test_read_only_default(self):
        tool = DummyTool()
        assert tool.read_only is False

    def test_concurrency_safe_default(self):
        tool = DummyTool()
        assert tool.concurrency_safe is True

    def test_cast_params(self):
        tool = DummyTool()
        params = tool.cast_params({"param1": "value"})
        assert params == {"param1": "value"}

    def test_validate_params_valid(self):
        tool = DummyTool()
        errors = tool.validate_params({"param1": "value"})
        assert errors == []

    def test_validate_params_missing_required(self):
        tool = DummyTool()
        errors = tool.validate_params({"param2": 1.0})
        assert len(errors) > 0
        assert "param1" in errors[0]

    def test_to_openai_schema(self):
        tool = DummyTool()
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "dummy"

    @pytest.mark.asyncio
    async def test_execute(self):
        tool = DummyTool()
        result = await tool.execute(param1="test", param2=1.0)
        assert result == "executed: test, 1.0"
