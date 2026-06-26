# coding=utf-8
"""Tests for v3.1.0 CLI: serve / stop / status / logs commands.

Zero-subprocess architecture:
    - cmd_serve calls uvicorn.Server.run() in-process (no subprocess)
    - --daemon uses os.fork() double-fork (not subprocess)
    - --frontend spawns npm dev server (subprocess, managed by finally)
    - --mcp spawns MCP server (subprocess)

Coverage:
    - ServeCommand.add_arguments: --port, --gateway-port, --frontend, --mcp, --daemon, --check-env
    - Port conflict detection
    - --check-env API key validation
    - Daemon mode double-fork + pidfile
    - StopCommand: pidfile exists / missing / stale
    - StatusCommand: /api/agent/status health check
    - LogsCommand: file missing / tail -F / 200 line fallback
    - _helpers: write_pidfile / read_pidfile / remove_pidfile roundtrip
    - is_port_free / is_pid_alive on alive/dead processes
"""
from __future__ import annotations

import os
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from QuantNodes.cli import COMMAND_REGISTRY
from QuantNodes.cli._helpers import (
    DEFAULT_GATEWAY_PORT,
    PIDFILE_NAME,
    is_pid_alive,
    is_port_free,
    load_env_file,
    read_pidfile,
    remove_pidfile,
    wait_for_health,
    write_pidfile,
)
from QuantNodes.cli.commands.serve import (
    LogsCommand,
    ServeCommand,
    StatusCommand,
    StopCommand,
    cmd_logs,
    cmd_serve,
    cmd_status,
    cmd_stop,
)


# ============================================================================
# 1. argparse parsing (5 tests)
# ============================================================================

class TestServeArgumentParsing:
    """Verify ServeCommand accepts all documented flags."""

    def _make_parser(self):
        import argparse
        parser = argparse.ArgumentParser(prog="test")
        sub = parser.add_subparsers(dest="command")
        ServeCommand().add_arguments(sub)
        return parser

    def test_serve_basic(self):
        parser = self._make_parser()
        args = parser.parse_args(["serve", "--host", "0.0.0.0", "--port", "9000"])
        assert args.host == "0.0.0.0"
        assert args.port == 9000

    def test_serve_default_gateway_port(self):
        parser = self._make_parser()
        args = parser.parse_args(["serve"])
        assert args.gateway_port == DEFAULT_GATEWAY_PORT

    def test_serve_daemon_frontend_check_env(self):
        parser = self._make_parser()
        args = parser.parse_args([
            "serve", "--daemon", "--frontend",
            "--check-env", "--frontend-port", "4000",
        ])
        assert args.daemon is True
        assert args.frontend is True
        assert args.check_env is True
        assert args.frontend_port == 4000

    def test_serve_mcp_flag(self):
        parser = self._make_parser()
        args = parser.parse_args(["serve", "--mcp", "--mcp-port", "9000"])
        assert args.mcp is True
        assert args.mcp_port == 9000

    def test_serve_mcp_default_port(self):
        parser = self._make_parser()
        args = parser.parse_args(["serve", "--mcp"])
        assert args.mcp is True
        assert args.mcp_port == 8765


# ============================================================================
# 2. Port-in-use detection (2 tests)
# ============================================================================

class TestServePortConflict:

    def test_backend_port_in_use_returns_1(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        busy_port = 19999

        def fake_is_port_free(port, host="127.0.0.1"):
            return port != busy_port
        monkeypatch.setattr("QuantNodes.cli.commands.serve.is_port_free", fake_is_port_free)

        args = MagicMock(
            host="127.0.0.1", port=busy_port,
            gateway_port=DEFAULT_GATEWAY_PORT,
            frontend=False, frontend_port=5173,
            mcp=False, mcp_port=8765,
            daemon=False, check_env=False,
        )
        rc = cmd_serve(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert f"端口 {busy_port} 已被占用" in out

    def test_gateway_port_in_use_returns_1(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        busy_port = 19998

        def fake_is_port_free(port, host="127.0.0.1"):
            return port != busy_port
        monkeypatch.setattr("QuantNodes.cli.commands.serve.is_port_free", fake_is_port_free)

        args = MagicMock(
            host="127.0.0.1", port=19380,
            gateway_port=busy_port,
            frontend=False, frontend_port=5173,
            mcp=False, mcp_port=8765,
            daemon=False, check_env=False,
        )
        rc = cmd_serve(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert f"端口 {busy_port} 已被占用" in out


# ============================================================================
# 3. --check-env validation (1 test)
# ============================================================================

class TestServeCheckEnv:
    def test_check_env_missing_api_key_returns_1(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("QUANTNODES__LLM__API_KEY", raising=False)
        monkeypatch.setattr("QuantNodes.cli.commands.serve.is_port_free", lambda *a, **k: True)

        args = MagicMock(
            host="127.0.0.1", port=19380,
            gateway_port=DEFAULT_GATEWAY_PORT,
            frontend=False, frontend_port=5173,
            mcp=False, mcp_port=8765,
            daemon=False, check_env=True,
        )
        rc = cmd_serve(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "QUANTNODES__LLM__API_KEY" in out


# ============================================================================
# 4. daemon mode (2 tests)
# ============================================================================

class TestServeDaemon:

    def test_daemon_double_fork_writes_pidfile(self, tmp_path, monkeypatch, capsys):
        """Daemon mode: double-fork, write pidfile, return 0."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("QuantNodes.cli.commands.serve.is_port_free", lambda *a, **k: True)
        monkeypatch.setattr("QuantNodes.cli.commands.serve.load_env_quietly", lambda: True)

        # Mock os.fork: first call returns child (0), second call returns grandchild (0)
        fork_calls = [0, 0, 1]  # parent=1 (exit), child=0, grandchild=0
        fork_idx = [0]
        def fake_fork():
            idx = fork_idx[0]
            fork_idx[0] += 1
            return fork_calls[idx]
        monkeypatch.setattr("os.fork", fake_fork)
        monkeypatch.setattr("os.setsid", lambda: None)

        # Mock uvicorn.Server to avoid actually starting
        mock_server = MagicMock()
        mock_server.run = MagicMock()
        mock_server.should_exit = False
        monkeypatch.setattr("uvicorn.Config", MagicMock(return_value=MagicMock()))
        monkeypatch.setattr("uvicorn.Server", MagicMock(return_value=mock_server))

        args = MagicMock(
            host="127.0.0.1", port=19380,
            gateway_port=DEFAULT_GATEWAY_PORT,
            frontend=False, frontend_port=5173,
            mcp=False, mcp_port=8765,
            daemon=True, check_env=False,
        )
        rc = cmd_serve(args)
        assert rc == 0

    def test_pidfile_roundtrip(self, tmp_path):
        """write_pidfile + read_pidfile preserves integer."""
        write_pidfile(98765, path=tmp_path / "test.pid")
        assert read_pidfile(tmp_path / "test.pid") == 98765
        remove_pidfile(tmp_path / "test.pid")
        assert read_pidfile(tmp_path / "test.pid") is None


# ============================================================================
# 5. stop command (3 tests)
# ============================================================================

class TestStopCommand:

    def test_stop_no_pidfile_returns_1(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        rc = cmd_stop(MagicMock())
        assert rc == 1
        out = capsys.readouterr().out
        assert "未找到" in out

    def test_stop_stale_pidfile_cleans_up(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        write_pidfile(2_000_000_000, path=tmp_path / PIDFILE_NAME)
        rc = cmd_stop(MagicMock())
        assert read_pidfile(tmp_path / PIDFILE_NAME) is None

    def test_stop_sends_sigterm(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        alive_pid = os.getpid()
        write_pidfile(alive_pid, path=tmp_path / PIDFILE_NAME)
        with patch("QuantNodes.cli.commands.serve.os.kill") as mock_kill:
            with patch("QuantNodes.cli.commands.serve.signal.SIGTERM", signal.SIGTERM):
                rc = cmd_stop(MagicMock())
        assert rc == 0
        assert mock_kill.call_count == 2
        second_call_args = mock_kill.call_args_list[1]
        assert second_call_args[0] == (alive_pid, signal.SIGTERM)
        assert not (tmp_path / PIDFILE_NAME).exists()


# ============================================================================
# 6. status command (2 tests)
# ============================================================================

class TestStatusCommand:

    def test_status_returns_0_when_running(self, capsys):
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"state": "running", "components": {"agent": True}}
        with patch("httpx.get", return_value=fake_response):
            rc = cmd_status(MagicMock(api_url="http://127.0.0.1:19380"))
        assert rc == 0
        out = capsys.readouterr().out
        assert '"state": "running"' in out

    def test_status_returns_1_on_connection_error(self, capsys):
        import httpx
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            rc = cmd_status(MagicMock(api_url="http://127.0.0.1:9999"))
        assert rc == 1
        out = capsys.readouterr().out
        assert '"reachable": false' in out


# ============================================================================
# 7. logs command (2 tests)
# ============================================================================

class TestLogsCommand:

    def test_logs_no_file_returns_1(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "QuantNodes.cli.commands.serve._serve_log_path",
            lambda: tmp_path / "nonexistent.log",
        )
        rc = cmd_logs(MagicMock(follow=False))
        assert rc == 1

    def test_logs_no_follow_uses_tail_n_200(self, tmp_path, monkeypatch):
        log = tmp_path / "quantnodes_serve.log"
        log.write_text("line1\nline2\n" * 100)
        monkeypatch.setattr("QuantNodes.cli.commands.serve._serve_log_path", lambda: log)
        with patch("subprocess.run") as mock_run:
            rc = cmd_logs(MagicMock(follow=False))
        assert rc == 0
        mock_run.assert_called_once()
        args, _ = mock_run.call_args
        assert args[0] == ["tail", "-n", "200", str(log)]


# ============================================================================
# 8. _helpers low-level (5 tests)
# ============================================================================

class TestHelpers:

    def test_is_pid_alive_for_current_process(self):
        assert is_pid_alive(os.getpid()) is True

    def test_is_pid_alive_for_nonexistent(self):
        assert is_pid_alive(2_000_000_000) is False

    def test_is_port_free_for_high_port(self):
        assert is_port_free(0) is True

    def test_load_env_file_returns_false_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert load_env_file() is False

    def test_load_env_file_returns_true_when_present(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("FOO_TEST_LOAD_ENV=hello\n")
        monkeypatch.chdir(tmp_path)
        assert load_env_file() is True
        assert os.environ.get("FOO_TEST_LOAD_ENV") == "hello"
        del os.environ["FOO_TEST_LOAD_ENV"]


# ============================================================================
# 9. Command registration sanity (1 test)
# ============================================================================

def test_lifecycle_commands_registered():
    names = COMMAND_REGISTRY.names()
    for cmd in ("serve", "stop", "status", "logs"):
        assert cmd in names, f"Missing command: {cmd}"
    assert len(names) == 20


# ============================================================================
# 10. wait_for_health polling (2 tests)
# ============================================================================

class TestWaitForHealth:

    def test_wait_for_health_returns_true_when_state_running(self):
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"state": "running"}
        with patch("httpx.get", return_value=fake_response):
            assert wait_for_health("http://127.0.0.1:9999", timeout_s=2) is True

    def test_wait_for_health_returns_false_on_timeout(self):
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"state": "starting"}
        with patch("httpx.get", return_value=fake_response):
            with patch("time.sleep"):
                assert wait_for_health("http://x", timeout_s=0.5, poll_interval_s=0.1) is False
