# coding=utf-8
"""QuantNodes.agent.tools.sandbox 单元测试"""
import pytest
from unittest.mock import MagicMock

from QuantNodes.agent.tools.sandbox import SandboxTool


class TestSandboxTool:
    def test_name_is_sandbox(self):
        tool = SandboxTool()
        assert tool.name == "sandbox"

    def test_description_is_set(self):
        tool = SandboxTool()
        assert "安全" in tool.description or "验证" in tool.description

    def test_parameters_has_code_field(self):
        tool = SandboxTool()
        params = tool.parameters
        assert "code" in params["properties"]

    def test_code_field_is_string_type(self):
        tool = SandboxTool()
        params = tool.parameters
        assert params["properties"]["code"]["type"] == "string"

    def test_code_field_is_required(self):
        tool = SandboxTool()
        assert "code" in tool.parameters["required"]

    def test_read_only_is_true(self):
        tool = SandboxTool()
        assert tool.read_only is True

    def test_init_with_default_values(self):
        tool = SandboxTool()
        assert tool._sandbox is not None

    def test_init_with_custom_allow_warnings(self):
        tool = SandboxTool(allow_warnings=True)
        assert tool._sandbox.allow_warnings is True

    def test_init_with_custom_max_code_length(self):
        tool = SandboxTool(max_code_length=5000)
        assert tool._sandbox.max_code_length == 5000

    @pytest.mark.asyncio
    async def test_execute_returns_validation_result(self):
        tool = SandboxTool()
        mock_result = MagicMock()
        mock_result.is_safe = True
        mock_result.errors = []
        mock_result.warnings = []
        mock_result.warnings_only = False
        tool._sandbox.validate = MagicMock(return_value=mock_result)

        result = await tool.execute(code="print('hello')")

        assert result["is_safe"] is True
        assert result["errors"] == []
        assert result["warnings"] == []

    @pytest.mark.asyncio
    async def test_execute_with_unsafe_code(self):
        tool = SandboxTool()
        mock_result = MagicMock()
        mock_result.is_safe = False
        mock_result.errors = ["Dangerous import: os"]
        mock_result.warnings = []
        mock_result.warnings_only = False
        tool._sandbox.validate = MagicMock(return_value=mock_result)

        result = await tool.execute(code="import os\nos.system('ls')")

        assert result["is_safe"] is False
        assert "Dangerous import" in result["errors"][0]
