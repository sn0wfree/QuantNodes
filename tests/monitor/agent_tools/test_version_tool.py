# -*- coding: utf-8 -*-
"""QuantNodes.monitor.agent_tools.version_tool 单元测试"""
import pytest
from unittest.mock import MagicMock, patch

from QuantNodes.monitor.agent_tools.version_tool import VersionTool


class TestVersionTool:
    def test_init(self):
        tool = VersionTool(db_path="~/.quantnodes/test_version.db")
        assert tool._db_path == "~/.quantnodes/test_version.db"
        assert tool._version_manager is None

    def test_name_property(self):
        tool = VersionTool()
        assert tool.name == "strategy_version"

    def test_description_property(self):
        tool = VersionTool()
        assert "版本" in tool.description
        assert "回滚" in tool.description

    def test_parameters_property(self):
        tool = VersionTool()
        params = tool.parameters
        assert params["type"] == "object"
        assert "action" in params["properties"]
        assert params["properties"]["action"]["type"] == "string"
        assert "save" in params["properties"]["action"]["enum"]

    def test_read_only_false(self):
        tool = VersionTool()
        assert tool.read_only is False

    def test_to_openai_schema(self):
        tool = VersionTool()
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert "function" in schema
        assert schema["function"]["name"] == "strategy_version"

    @pytest.mark.asyncio
    async def test_execute_save_without_config_path(self):
        tool = VersionTool()
        result = await tool.execute(action="save", strategy_name="test_strategy")
        assert "error" in result
        assert "config_path required" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_save(self):
        tool = VersionTool()

        mock_vm = MagicMock()
        mock_sv = MagicMock()
        mock_sv.version = "v2"
        mock_sv.commit_hash = "abc123"
        mock_vm.save_version.return_value = mock_sv

        with patch.object(tool, "_get_version_manager", return_value=mock_vm):
            result = await tool.execute(
                action="save",
                strategy_name="test_strategy",
                config_path="/path/to/config.yaml",
                description="Initial version",
            )

        assert result["status"] == "saved"
        assert result["version"] == "v2"
        assert result["commit_hash"] == "abc123"
        mock_vm.save_version.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_list(self):
        tool = VersionTool()

        mock_vm = MagicMock()
        v1 = MagicMock()
        v1.version = "v1"
        v1.commit_hash = "aaa111"
        v1.description = "First version"
        v1.created_at = None
        v2 = MagicMock()
        v2.version = "v2"
        v2.commit_hash = "bbb222"
        v2.description = "Second version"
        v2.created_at = None
        mock_vm.list_versions.return_value = [v2, v1]

        with patch.object(tool, "_get_version_manager", return_value=mock_vm):
            result = await tool.execute(action="list", strategy_name="test_strategy")

        assert "versions" in result
        assert len(result["versions"]) == 2
        assert result["versions"][0]["version"] == "v2"
        assert result["versions"][1]["version"] == "v1"

    @pytest.mark.asyncio
    async def test_execute_diff_without_version(self):
        tool = VersionTool()
        result = await tool.execute(action="diff", strategy_name="test_strategy")
        assert "error" in result
        assert "version required" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_diff_need_more_versions(self):
        tool = VersionTool()

        mock_vm = MagicMock()
        mock_vm.list_versions.return_value = [MagicMock(version="v1")]

        with patch.object(tool, "_get_version_manager", return_value=mock_vm):
            result = await tool.execute(action="diff", strategy_name="test_strategy", version="v1")

        assert "error" in result
        assert "at least 2 versions" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_diff(self):
        tool = VersionTool()

        mock_vm = MagicMock()
        v1 = MagicMock(version="v1")
        v2 = MagicMock(version="v2")
        mock_vm.list_versions.return_value = [v2, v1]
        mock_vm.diff_versions.return_value = "--- v2\n+++ v1\n-old line\n+new line"

        with patch.object(tool, "_get_version_manager", return_value=mock_vm):
            result = await tool.execute(action="diff", strategy_name="test_strategy", version="v2")

        assert "diff" in result
        mock_vm.diff_versions.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_rollback_without_version(self):
        tool = VersionTool()
        result = await tool.execute(action="rollback", strategy_name="test_strategy")
        assert "error" in result
        assert "version required" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_rollback_not_found(self):
        tool = VersionTool()

        mock_vm = MagicMock()
        mock_vm.rollback.return_value = None

        with patch.object(tool, "_get_version_manager", return_value=mock_vm):
            result = await tool.execute(action="rollback", strategy_name="test_strategy", version="v99")

        assert "error" in result
        assert "v99 not found" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_rollback_success(self):
        tool = VersionTool()

        mock_vm = MagicMock()
        new_sv = MagicMock()
        new_sv.version = "v1"
        new_sv.commit_hash = "new_hash_123"
        mock_vm.rollback.return_value = new_sv

        with patch.object(tool, "_get_version_manager", return_value=mock_vm):
            result = await tool.execute(action="rollback", strategy_name="test_strategy", version="v2")

        assert result["status"] == "rolled_back"
        assert result["new_version"] == "v1"
        assert result["commit_hash"] == "new_hash_123"

    @pytest.mark.asyncio
    async def test_execute_current(self):
        tool = VersionTool()

        mock_vm = MagicMock()
        mock_vm.get_current_version.return_value = "v3"

        with patch.object(tool, "_get_version_manager", return_value=mock_vm):
            result = await tool.execute(action="current", strategy_name="test_strategy")

        assert result["current_version"] == "v3"

    @pytest.mark.asyncio
    async def test_execute_unknown_action(self):
        tool = VersionTool()
        result = await tool.execute(action="unknown", strategy_name="test_strategy")
        assert "error" in result
        assert "Unknown action" in result["error"]
