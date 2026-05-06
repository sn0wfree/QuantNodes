# coding=utf-8
"""QuantNodes.agent.tools.strategy 单元测试"""
import pytest

from QuantNodes.agent.tools.strategy import StrategyTool


class TestStrategyTool:
    def test_creation_without_llm(self):
        tool = StrategyTool()
        assert tool is not None

    def test_creation_with_llm(self):
        mock_llm = object()
        tool = StrategyTool(llm_client=mock_llm)
        assert tool._llm_client is mock_llm

    def test_name(self):
        tool = StrategyTool()
        assert tool.name == "strategy"

    def test_description(self):
        tool = StrategyTool()
        assert "策略" in tool.description or "Pipeline" in tool.description

    def test_parameters(self):
        tool = StrategyTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "description" in params["properties"]
        assert "validate" in params["properties"]
        assert params["required"] == ["description"]

    def test_read_only(self):
        tool = StrategyTool()
        assert tool.read_only is False

    def test_to_openai_schema(self):
        tool = StrategyTool()
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "strategy"

    @pytest.mark.asyncio
    async def test_execute_without_llm(self):
        tool = StrategyTool()
        result = await tool.execute(description="生成一个动量策略")
        assert result["status"] == "needs_configuration"
        assert "LLM client" in result["message"]
        assert result["code"] == ""
        assert result["is_valid"] is False

    @pytest.mark.asyncio
    async def test_execute_validate_default_true(self):
        tool = StrategyTool()
        result = await tool.execute(description="测试")
        assert result["status"] == "needs_configuration"
