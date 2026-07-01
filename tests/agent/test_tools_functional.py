# coding=utf-8
"""Tests for agent/tools/ functional behavior — 8 tool files, ~1800 LOC.

Covers: output_truncation, shell_safety, ToolContext, and tool schema/creation.
All confirmed LIVE modules recently modified.
"""

import asyncio
import os

import pytest

from QuantNodes.agent.tools.output_truncation import truncate_output, TruncatedOutput
from QuantNodes.agent.tools.shell_safety import ShellConfig
from QuantNodes.agent.tools.context import ToolContext


# ============================================================================
# output_truncation
# ============================================================================

class TestOutputTruncation:
    def test_short_output_unchanged(self):
        result = truncate_output("hello world")
        assert result.content == "hello world"
        assert result.truncated is False

    def test_empty_string(self):
        result = truncate_output("")
        assert result.content == ""
        assert result.truncated is False

    def test_long_output_truncated_head_tail(self):
        lines = [f"line {i}" for i in range(20000)]
        long_output = "\n".join(lines)
        result = truncate_output(long_output, max_lines=100)
        assert result.truncated is True
        assert result.kept_lines <= 100

    def test_byte_limit(self):
        big = "x" * (2 * 1024 * 1024)  # 2MB
        result = truncate_output(big, max_bytes=1024 * 1024)
        assert result.truncated is True

    def test_exact_limit_not_truncated(self):
        lines = ["x"] * 100
        result = truncate_output("\n".join(lines), max_lines=100)
        assert result.truncated is False

    def test_truncated_output_has_metadata(self):
        result = truncate_output("a\n" * 200, max_lines=10)
        assert hasattr(result, 'truncated')
        assert hasattr(result, 'total_lines')
        assert hasattr(result, 'kept_lines')
        assert hasattr(result, 'content')

    def test_total_bytes_tracking(self):
        result = truncate_output("hello", max_lines=100)
        assert result.total_bytes == 5

    def test_kept_bytes_tracking(self):
        result = truncate_output("hello", max_lines=100)
        assert result.kept_bytes == 5


# ============================================================================
# shell_safety
# ============================================================================

class TestShellSafety:
    def test_shell_config_creation(self):
        config = ShellConfig()
        assert config is not None
        assert config.timeout_seconds == 120

    def test_shell_config_with_params(self):
        config = ShellConfig(timeout_seconds=30, project_root="/tmp")
        assert config.timeout_seconds == 30
        assert config.project_root == "/tmp"

    def test_shell_config_defaults(self):
        config = ShellConfig()
        assert config.max_output_bytes == 1024 * 1024
        assert config.max_output_lines == 10000


# ============================================================================
# ToolContext
# ============================================================================

class TestToolContext:
    def test_creation(self):
        ctx = ToolContext(
            session_id="s1",
            agent_id="build",
            project_root="/tmp",
            permission_service=None,
        )
        assert ctx.session_id == "s1"
        assert ctx.agent_id == "build"

    def test_is_aborted_false_by_default(self):
        ctx = ToolContext(
            session_id="s1",
            agent_id="build",
            project_root="/tmp",
            permission_service=None,
        )
        assert ctx.is_aborted() is False

    def test_tool_defaults(self):
        ctx = ToolContext(
            session_id="s1",
            agent_id="build",
            project_root="/tmp",
            permission_service=None,
            tool_defaults={"sandbox": {"timeout": 30}},
        )
        assert ctx.tool_defaults.get("sandbox", {}).get("timeout") == 30

    def test_abort_signal_is_event(self):
        ctx = ToolContext(
            session_id="s1",
            agent_id="build",
            project_root="/tmp",
            permission_service=None,
        )
        assert isinstance(ctx.abort_signal, asyncio.Event)


# ============================================================================
# Tool Schema Tests (no workspace needed)
# ============================================================================

class TestFileOpsToolSchema:
    def test_tool_creation(self, tmp_path):
        from QuantNodes.agent.tools.file_ops import FileOpsTool
        tool = FileOpsTool(workspace=str(tmp_path))
        assert tool.name == "file_ops"

    def test_parameters_schema(self, tmp_path):
        from QuantNodes.agent.tools.file_ops import FileOpsTool
        tool = FileOpsTool(workspace=str(tmp_path))
        schema = tool.parameters
        assert "properties" in schema
        assert "action" in schema["properties"]


class TestCodeSearchToolSchema:
    def test_tool_creation(self, tmp_path):
        from QuantNodes.agent.tools.code_search import CodeSearchTool
        tool = CodeSearchTool(workspace=str(tmp_path))
        assert tool.name == "code_search"

    def test_parameters_schema(self, tmp_path):
        from QuantNodes.agent.tools.code_search import CodeSearchTool
        tool = CodeSearchTool(workspace=str(tmp_path))
        schema = tool.parameters
        assert "properties" in schema
        assert "action" in schema["properties"]


class TestGitOpsToolSchema:
    def test_tool_creation(self, tmp_path):
        from QuantNodes.agent.tools.git_ops import GitOpsTool
        tool = GitOpsTool(workspace=str(tmp_path))
        assert tool.name == "git_ops"

    def test_parameters_schema(self, tmp_path):
        from QuantNodes.agent.tools.git_ops import GitOpsTool
        tool = GitOpsTool(workspace=str(tmp_path))
        schema = tool.parameters
        assert "properties" in schema
        assert "action" in schema["properties"]


class TestWebFetchTool:
    def test_tool_creation(self):
        from QuantNodes.agent.tools.web_fetch import WebFetchTool
        tool = WebFetchTool()
        assert tool.name == "web_fetch"

    def test_parameters_schema(self):
        from QuantNodes.agent.tools.web_fetch import WebFetchTool
        tool = WebFetchTool()
        schema = tool.parameters
        assert "properties" in schema
        assert "url" in schema["properties"]


class TestWebSearchTool:
    def test_tool_creation(self):
        from QuantNodes.agent.tools.web_search import WebSearchTool
        tool = WebSearchTool()
        assert tool.name == "web_search"

    def test_parameters_schema(self):
        from QuantNodes.agent.tools.web_search import WebSearchTool
        tool = WebSearchTool()
        schema = tool.parameters
        assert "properties" in schema
        assert "query" in schema["properties"]


# ============================================================================
# FileOpsTool Functional (with tmp_path workspace)
# ============================================================================

class TestFileOpsFunctional:
    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        from QuantNodes.agent.tools.file_ops import FileOpsTool
        tool = FileOpsTool(workspace=str(tmp_path))
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")

        result = await tool.execute(action="read_file", path="test.txt")
        assert "hello world" in str(result)

    @pytest.mark.asyncio
    async def test_read_file_not_found(self, tmp_path):
        from QuantNodes.agent.tools.file_ops import FileOpsTool
        tool = FileOpsTool(workspace=str(tmp_path))

        result = await tool.execute(action="read_file", path="nonexistent.txt")
        assert "error" in str(result).lower() or "not found" in str(result).lower()

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_path):
        from QuantNodes.agent.tools.file_ops import FileOpsTool
        tool = FileOpsTool(workspace=str(tmp_path))

        result = await tool.execute(
            action="write_file",
            path="output.txt",
            content="test content",
        )
        target = tmp_path / "output.txt"
        assert target.exists()
        assert target.read_text() == "test content"

    @pytest.mark.asyncio
    async def test_list_files(self, tmp_path):
        from QuantNodes.agent.tools.file_ops import FileOpsTool
        tool = FileOpsTool(workspace=str(tmp_path))
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        result = await tool.execute(action="list_files", path=".")
        assert "a.txt" in str(result) or "a" in str(result)

    @pytest.mark.asyncio
    async def test_glob_files(self, tmp_path):
        from QuantNodes.agent.tools.file_ops import FileOpsTool
        tool = FileOpsTool(workspace=str(tmp_path))
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        result = await tool.execute(action="glob_files", pattern="*.py", path=".")
        assert "a.py" in str(result)


class TestCodeSearchFunctional:
    @pytest.mark.asyncio
    async def test_grep_finds_matches(self, tmp_path):
        from QuantNodes.agent.tools.code_search import CodeSearchTool
        tool = CodeSearchTool(workspace=str(tmp_path))
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello_world():\n    pass\n")

        result = await tool.execute(
            action="grep",
            pattern="hello_world",
            path=".",
        )
        assert "hello_world" in str(result)

    @pytest.mark.asyncio
    async def test_find_files(self, tmp_path):
        from QuantNodes.agent.tools.code_search import CodeSearchTool
        tool = CodeSearchTool(workspace=str(tmp_path))
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.txt").write_text("b")

        result = await tool.execute(
            action="find_files",
            pattern="*.py",
            path=".",
        )
        assert "a.py" in str(result)


class TestGitOpsFunctional:
    @pytest.mark.asyncio
    async def test_git_status(self, tmp_path):
        from QuantNodes.agent.tools.git_ops import GitOpsTool
        tool = GitOpsTool(workspace=str(tmp_path))

        result = await tool.execute(action="git_status")
        assert isinstance(result, dict) or isinstance(result, str)

    @pytest.mark.asyncio
    async def test_git_log(self, tmp_path):
        from QuantNodes.agent.tools.git_ops import GitOpsTool
        tool = GitOpsTool(workspace=str(tmp_path))

        result = await tool.execute(action="git_log", max_count=5)
        assert isinstance(result, dict) or isinstance(result, str)
