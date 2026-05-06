# coding=utf-8
"""QuantNodes.integration.test_agent_tools_pipeline 集成测试"""
import pytest

from QuantNodes.agent.tools.pipeline import PipelineTool
from QuantNodes.agent.tools.factor import FactorTool
from QuantNodes.agent.tools.strategy import StrategyTool


class TestAgentToolsPipelineIntegration:
    """测试 Agent 工具链的端到端流程"""

    @pytest.mark.asyncio
    async def test_pipeline_validation_then_factor_analysis(self):
        pipeline_tool = PipelineTool()
        factor_tool = FactorTool()

        pipeline_code = """
from QuantNodes.operator_node.transform import TransformNode
node = TransformNode().select(["col1", "col2"])
"""
        validation_result = await pipeline_tool.execute(code=pipeline_code)
        assert validation_result["is_valid"] is True
        assert len(validation_result["nodes"]) > 0

        factor_code = """
import polars as pl
result = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-01"],
    "code": ["A", "B"],
    "factor_value": [0.1, 0.2],
    "forward_return": [0.01, 0.02],
})
"""
        factor_result = await factor_tool.execute(
            factor_code=factor_code,
            analysis_type="ic"
        )
        assert "analysis" in factor_result

    @pytest.mark.asyncio
    async def test_strategy_generation_without_llm(self):
        strategy_tool = StrategyTool()

        result = await strategy_tool.execute(
            description="生成一个简单的移动平均线策略"
        )

        assert result["status"] == "needs_configuration"
        assert result["code"] == ""
        assert result["is_valid"] is False

    @pytest.mark.asyncio
    async def test_pipeline_security_check(self):
        pipeline_tool = PipelineTool()

        unsafe_code = """
import os
os.system("rm -rf /")
"""
        result = await pipeline_tool.execute(code=unsafe_code)
        assert result["is_valid"] is False or result["security_status"] == "unsafe"

    @pytest.mark.asyncio
    async def test_multiple_pipeline_validations(self):
        pipeline_tool = PipelineTool()

        codes = [
            "TransformNode().select(['a'])",
            "from QuantNodes.operator_node.transform import TransformNode",
            "x = 1",
        ]

        for code in codes:
            result = await pipeline_tool.execute(code=code)
            assert result["is_valid"] is True


class TestToolSchemaIntegration:
    """测试工具 Schema 的一致性"""

    def test_tools_have_openai_schema(self):
        tools = [
            PipelineTool(),
            FactorTool(),
            StrategyTool(),
        ]

        for tool in tools:
            schema = tool.to_openai_schema()
            assert schema["type"] == "function"
            assert "function" in schema
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]

    def test_tools_parameters_are_valid_json_schema(self):
        tools = [
            PipelineTool(),
            FactorTool(),
        ]

        for tool in tools:
            params = tool.parameters
            assert params["type"] == "object"
            assert "properties" in params
            assert isinstance(params["properties"], dict)
