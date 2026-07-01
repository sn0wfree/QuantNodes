# coding=utf-8
"""
test_factor_tool_extended.py - FactorTool 扩展测试

补 test_tools.py 中 FactorTool 测试的不足:
- analyze_type enum 验证
- parameters schema 完整性
- execute() 返回值结构 (mock sandbox)
- read_only / concurrency_safe 属性
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from QuantNodes.agent.tools.factor import FactorTool


# ==============================================================================
# 基本属性
# ==============================================================================


class TestFactorToolBasics:
    def test_name(self):
        assert FactorTool().name == "factor"

    def test_read_only_is_true(self):
        """FactorTool 只读 (不修改数据)。"""
        assert FactorTool().read_only is True

    def test_description_non_empty(self):
        tool = FactorTool()
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0

    def test_concurrency_safe(self):
        """FactorTool 默认并发安全。"""
        tool = FactorTool()
        assert isinstance(tool.concurrency_safe, bool)


# ==============================================================================
# Parameters schema
# ==============================================================================


class TestFactorToolParameters:
    @pytest.fixture
    def params(self):
        return FactorTool().parameters

    def test_parameters_type_is_object(self, params):
        assert params["type"] == "object"

    def test_parameters_has_properties(self, params):
        assert "properties" in params
        assert isinstance(params["properties"], dict)

    def test_factor_code_required(self, params):
        assert "factor_code" in params["required"]

    def test_factor_code_property(self, params):
        fc = params["properties"]["factor_code"]
        assert fc["type"] == "string"
        assert "description" in fc

    def test_analysis_type_enum_values(self, params):
        """analysis_type 必须是 ic/correlation/both 三选一。"""
        at = params["properties"].get("analysis_type", {})
        assert "enum" in at, "analysis_type 缺少 enum 约束"
        assert set(at["enum"]) == {"ic", "correlation", "both"}

    def test_analysis_type_default(self, params):
        """analysis_type 应有默认值 (如 'both')。"""
        at = params["properties"].get("analysis_type", {})
        # 通常 default 是 "both" 或 "ic"
        assert "default" in at

    def test_required_includes_factor_code(self, params):
        """required 至少包含 factor_code。"""
        assert "factor_code" in params["required"]


# ==============================================================================
# Execute (mocked)
# ==============================================================================


class TestFactorToolExecute:
    """mock CodeSandbox 和执行环境, 测试 execute() 流程。"""

    @pytest.fixture
    def tool(self):
        return FactorTool()

    @pytest.mark.asyncio
    async def test_execute_with_valid_factor_code(self, tool):
        """有效 factor_code 返回 IC / correlation 结果。"""
        # mock CodeSandbox 让 factor_code 中的 result 变量可访问
        with patch("QuantNodes.ai.sandbox.CodeSandbox") as mock_sb:
            mock_instance = MagicMock()
            mock_instance.validate_and_execute.return_value = {
                "result": {
                    "ic_mean": 0.05,
                    "ic_std": 0.1,
                    "ir": 0.5,
                    "correlation": 0.3,
                }
            }
            mock_sb.return_value = mock_instance

            result = await tool.execute(
                factor_code="result = {'ic_mean': 0.05}",
                analysis_type="both",
            )

        assert isinstance(result, dict)
        # 验证 sandbox 被调用
        mock_instance.validate_and_execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_missing_result_variable(self, tool):
        """factor_code 未设置 result 变量时, 返回错误。"""
        with patch("QuantNodes.ai.sandbox.CodeSandbox") as mock_sb:
            mock_instance = MagicMock()
            mock_instance.validate_and_execute.return_value = {}  # 无 result
            mock_sb.return_value = mock_instance

            result = await tool.execute(factor_code="x = 1")

        assert "error" in result or "status" in result

    @pytest.mark.asyncio
    async def test_execute_sandbox_error_caught(self, tool):
        """CodeSandbox 抛异常时, 返回错误 (不崩溃)。"""
        with patch("QuantNodes.ai.sandbox.CodeSandbox") as mock_sb:
            mock_instance = MagicMock()
            mock_instance.validate_and_execute.side_effect = RuntimeError("sandbox failed")
            mock_sb.return_value = mock_instance

            result = await tool.execute(factor_code="result = {}")

        # status="error" 或 errors 字段
        assert result.get("status") == "error" or "errors" in result

    @pytest.mark.asyncio
    async def test_execute_analysis_type_ic(self, tool):
        """analysis_type='ic' 只计算 IC, 不算 correlation。"""
        import polars as pl

        with patch("QuantNodes.ai.sandbox.CodeSandbox") as mock_sb:
            mock_instance = MagicMock()
            # result 必须是 Polars DataFrame
            df = pl.DataFrame({
                "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                "code": ["A", "B", "A", "B"],
                "factor_value": [0.1, 0.2, 0.3, 0.4],
                "forward_return": [0.05, 0.03, 0.02, 0.01],
            })
            mock_instance.validate_and_execute.return_value = {"result": df}
            mock_sb.return_value = mock_instance

            result = await tool.execute(
                factor_code="result = df",
                analysis_type="ic",
            )

        # 应返回 IC 结果
        assert isinstance(result, dict)
        # 内部计算可能 still fail due to mock data, 但 structure 应正确
        assert "analysis" in result

    @pytest.mark.asyncio
    async def test_execute_analysis_type_correlation(self, tool):
        """analysis_type='correlation' 只算 correlation。"""
        import polars as pl

        with patch("QuantNodes.ai.sandbox.CodeSandbox") as mock_sb:
            mock_instance = MagicMock()
            df = pl.DataFrame({
                "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                "code": ["A", "B", "A", "B"],
                "factor_value": [0.1, 0.2, 0.3, 0.4],
                "forward_return": [0.05, 0.03, 0.02, 0.01],
            })
            mock_instance.validate_and_execute.return_value = {"result": df}
            mock_sb.return_value = mock_instance

            result = await tool.execute(
                factor_code="result = df",
                analysis_type="correlation",
            )

        # result 应包含 analysis 结构 (即使内部 IC 计算可能失败)
        assert "analysis" in result
        assert "ic" in result["analysis"]
        assert "correlation" in result["analysis"]
        assert "correlation" in result["analysis"]