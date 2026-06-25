# coding=utf-8
"""Tests for v3.0.0 Stage 7 ``quantnodes agent {status,chat,restart}`` subcommand.

AgentCommand 是一个 sub-subparsers 命令组:
    quantnodes agent status     → GET  /api/agent/status
    quantnodes agent chat MSG   → POST /api/agent/chat/send
    quantnodes agent restart    → POST /api/agent/restart

设计原则:
    - 用 unittest.mock.patch httpx.get/post 避免真实 HTTP 调用
    - 验证 stdout 输出 + exit code + sub-subparser 解析
    - 覆盖 ConnectError / 非 200 状态码 / 错误响应体
"""
from __future__ import annotations

import argparse
import json
from unittest.mock import MagicMock, patch

import pytest

from QuantNodes.cli.commands.agent import (
    AgentCommand,
    cmd_agent_chat,
    cmd_agent_restart,
    cmd_agent_status,
)


# ============================================================================
# 1. argparse parsing (3 tests)
# ============================================================================

class TestAgentArgumentParsing:

    def _make_parser(self):
        parser = argparse.ArgumentParser(prog="test")
        sub = parser.add_subparsers(dest="command")
        AgentCommand().add_arguments(sub)
        return parser

    def test_agent_status_parses(self):
        parser = self._make_parser()
        args = parser.parse_args(["agent", "status"])
        assert args.agent_action == "status"
        assert args.api_url == "http://127.0.0.1:19380"  # default

    def test_agent_chat_parses_message_and_session(self):
        parser = self._make_parser()
        args = parser.parse_args([
            "agent", "chat", "hello world", "--session", "test-sess",
            "--api-url", "http://example.com:1234",
        ])
        assert args.agent_action == "chat"
        assert args.message == "hello world"
        assert args.session == "test-sess"
        assert args.api_url == "http://example.com:1234"

    def test_agent_restart_parses(self):
        parser = self._make_parser()
        args = parser.parse_args(["agent", "restart"])
        assert args.agent_action == "restart"

    def test_agent_requires_subcommand(self):
        """Without status/chat/restart, argparse should error."""
        parser = self._make_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["agent"])


# ============================================================================
# 2. agent status (3 tests)
# ============================================================================

class TestAgentStatus:

    def test_status_running_returns_0_and_prints_json(self, capsys):
        fake = MagicMock()
        fake.status_code = 200
        fake.json.return_value = {
            "state": "running", "components": {"agent": True, "bus": True},
        }
        fake.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=fake):
            rc = cmd_agent_status(MagicMock(api_url="http://x"))
        assert rc == 0
        out = capsys.readouterr().out
        assert '"state": "running"' in out
        assert '"agent": true' in out

    def test_status_not_running_returns_1(self, capsys):
        fake = MagicMock()
        fake.status_code = 200
        fake.json.return_value = {"state": "error"}
        fake.raise_for_status = MagicMock()
        with patch("httpx.get", return_value=fake):
            rc = cmd_agent_status(MagicMock(api_url="http://x"))
        assert rc == 1

    def test_status_connection_error_returns_1(self, capsys):
        import httpx
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            rc = cmd_agent_status(MagicMock(api_url="http://127.0.0.1:9999"))
        assert rc == 1
        captured = capsys.readouterr()
        # error message goes to stderr (so it doesn't pollute stdout JSON)
        assert "无法连接" in captured.err


# ============================================================================
# 3. agent chat (4 tests)
# ============================================================================

class TestAgentChat:

    def test_chat_sends_message_and_prints_content(self, capsys):
        fake = MagicMock()
        fake.status_code = 200
        fake.json.return_value = {
            "message_id": "msg-1",
            "content": "Hello response",
            "tools_used": [],
            "session_id": "default",
            "stop_reason": "stop",
            "error": None,
        }
        fake.raise_for_status = MagicMock()
        with patch("httpx.post", return_value=fake) as mock_post:
            args = MagicMock(
                api_url="http://x", message="hi", session="s1",
            )
            rc = cmd_agent_chat(args)
        assert rc == 0
        # Verify HTTP call payload
        mock_post.assert_called_once()
        args_post, kwargs_post = mock_post.call_args
        assert args_post[0] == "http://x/api/agent/chat/send"
        assert kwargs_post["json"]["message"] == "hi"
        assert kwargs_post["json"]["session_id"] == "s1"
        # Verify stdout
        out = capsys.readouterr().out
        assert "Hello response" in out

    def test_chat_error_field_returns_1(self, capsys):
        fake = MagicMock()
        fake.status_code = 200
        fake.json.return_value = {"error": "LLM timeout", "content": ""}
        fake.raise_for_status = MagicMock()
        with patch("httpx.post", return_value=fake):
            rc = cmd_agent_chat(MagicMock(api_url="http://x", message="x"))
        assert rc == 1
        captured = capsys.readouterr()
        # [Error] goes to stderr
        assert "LLM timeout" in captured.err

    def test_chat_empty_content_returns_0(self, capsys):
        """If content is empty but no error, still return 0 (don't fail)."""
        fake = MagicMock()
        fake.status_code = 200
        fake.json.return_value = {"content": "", "error": None}
        fake.raise_for_status = MagicMock()
        with patch("httpx.post", return_value=fake):
            rc = cmd_agent_chat(MagicMock(api_url="http://x", message="x"))
        assert rc == 0

    def test_chat_connection_error_returns_1(self, capsys):
        import httpx
        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            rc = cmd_agent_chat(MagicMock(api_url="http://x", message="x"))
        assert rc == 1
        captured = capsys.readouterr()
        assert "无法连接" in captured.err


# ============================================================================
# 4. agent restart (3 tests)
# ============================================================================

class TestAgentRestart:

    def test_restart_success_returns_0(self, capsys):
        fake = MagicMock()
        fake.status_code = 200
        fake.json.return_value = {"success": True, "state": "running"}
        with patch("httpx.post", return_value=fake):
            rc = cmd_agent_restart(MagicMock(api_url="http://x"))
        assert rc == 0
        out = capsys.readouterr().out
        assert '"success": true' in out

    def test_restart_500_returns_1(self, capsys):
        fake = MagicMock()
        fake.status_code = 500
        fake.json.return_value = {"detail": "restart failed: boom"}
        with patch("httpx.post", return_value=fake):
            rc = cmd_agent_restart(MagicMock(api_url="http://x"))
        assert rc == 1

    def test_restart_connection_error_returns_1(self, capsys):
        import httpx
        with patch("httpx.post", side_effect=httpx.ConnectError("refused")):
            rc = cmd_agent_restart(MagicMock(api_url="http://x"))
        assert rc == 1


# ============================================================================
# 5. Command dispatch via run() (1 test)
# ============================================================================

class TestAgentCommandDispatch:

    def test_run_dispatches_to_status(self):
        """AgentCommand.run() should delegate based on args.agent_action."""
        cmd = AgentCommand()
        args = MagicMock(agent_action="status", api_url="http://x")
        with patch("httpx.get") as mock_get:
            fake = MagicMock()
            fake.status_code = 200
            fake.json.return_value = {"state": "running"}
            fake.raise_for_status = MagicMock()
            mock_get.return_value = fake
            rc = cmd.run(args)
        assert rc == 0
        mock_get.assert_called_once()

    def test_run_dispatches_to_chat(self):
        cmd = AgentCommand()
        args = MagicMock(agent_action="chat", api_url="http://x",
                         message="hi", session="default")
        with patch("httpx.post") as mock_post:
            fake = MagicMock()
            fake.status_code = 200
            fake.json.return_value = {"content": "reply", "error": None}
            fake.raise_for_status = MagicMock()
            mock_post.return_value = fake
            rc = cmd.run(args)
        assert rc == 0
        mock_post.assert_called_once()

    def test_run_dispatches_to_restart(self):
        cmd = AgentCommand()
        args = MagicMock(agent_action="restart", api_url="http://x")
        with patch("httpx.post") as mock_post:
            fake = MagicMock()
            fake.status_code = 200
            fake.json.return_value = {"success": True}
            mock_post.return_value = fake
            rc = cmd.run(args)
        assert rc == 0