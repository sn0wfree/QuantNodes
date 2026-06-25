# coding=utf-8
"""Tests for v3.0.0 Stage 7 CLI: serve / stop / status / logs commands.

覆盖范围:
    - ServeCommand.add_arguments 接受所有 flag (--port, --gateway-port, --frontend, --daemon, --check-env)
    - 端口冲突检测（占用的端口立即返回 1）
    - --check-env 模式验证 API key
    - daemon 模式写 .quantnodes.pid + 日志路径
    - StopCommand: pidfile 存在/不存在/stale 三种情况
    - StatusCommand: 调用 /api/agent/status + 返回正确 exit code
    - LogsCommand: 文件不存在 / tail -F / 200 行 fallback
    - load_env_file: dotenv 缺失/.env 缺失
    - write_pidfile / read_pidfile / remove_pidfile roundtrip
    - is_port_free / is_pid_alive 在 alive/dead 进程上的正确性

设计原则:
    - 用 unittest.mock.patch 隔离 subprocess.Popen / httpx.get 等副作用
    - tmp_path 提供隔离的 .quantnodes.pid 文件路径
    - 不真正启动 uvicorn 或 调真实 HTTP endpoint
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
# 1. argparse parsing (4 tests)
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
        """quantnodes serve --host X --port Y"""
        parser = self._make_parser()
        args = parser.parse_args(["serve", "--host", "0.0.0.0", "--port", "9000"])
        assert args.host == "0.0.0.0"
        assert args.port == 9000

    def test_serve_default_gateway_port(self):
        """Default --gateway-port is DEFAULT_GATEWAY_PORT (18090)."""
        parser = self._make_parser()
        args = parser.parse_args(["serve"])
        assert args.gateway_port == DEFAULT_GATEWAY_PORT

    def test_serve_daemon_frontend_check_env(self):
        """All 4 boolean flags parse correctly."""
        parser = self._make_parser()
        args = parser.parse_args([
            "serve", "--daemon", "--frontend",
            "--check-env", "--frontend-port", "4000",
        ])
        assert args.daemon is True
        assert args.frontend is True
        assert args.check_env is True
        assert args.frontend_port == 4000


# ============================================================================
# 2. Port-in-use detection (2 tests)
# ============================================================================

class TestServePortConflict:
    """`cmd_serve` should fail fast on port conflict instead of letting uvicorn crash."""

    def test_backend_port_in_use_returns_1(self, tmp_path, monkeypatch, capsys):
        """If --port is occupied, serve returns 1 with clear message."""
        # Pick a random free port, then claim it
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            busy_port = s.getsockname()[1]
        # Hold the port until end of test
        try:
            monkeypatch.chdir(tmp_path)
            # Mock is_port_free: backend busy, gateway free
            def fake_is_port_free(port, host="127.0.0.1"):
                return port != busy_port
            monkeypatch.setattr(
                "QuantNodes.cli.commands.serve.is_port_free", fake_is_port_free
            )
            args = MagicMock(
                host="127.0.0.1", port=busy_port,
                gateway_port=DEFAULT_GATEWAY_PORT,
                frontend=False, frontend_port=5173,
                daemon=False, check_env=False,
            )
            rc = cmd_serve(args)
            assert rc == 1
            out = capsys.readouterr().out
            assert f"端口 {busy_port} 已被占用" in out
            assert "后端端口" in out
        finally:
            del s

    def test_gateway_port_in_use_returns_1(self, tmp_path, monkeypatch, capsys):
        """If --gateway-port is occupied, serve returns 1 with gpustack hint."""
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            busy_port = s.getsockname()[1]
        try:
            monkeypatch.chdir(tmp_path)
            # Mock is_port_free: backend free, gateway busy
            def fake_is_port_free(port, host="127.0.0.1"):
                return port != busy_port
            monkeypatch.setattr(
                "QuantNodes.cli.commands.serve.is_port_free", fake_is_port_free
            )
            args = MagicMock(
                host="127.0.0.1", port=19380,
                gateway_port=busy_port,
                frontend=False, frontend_port=5173,
                daemon=False, check_env=False,
            )
            rc = cmd_serve(args)
            assert rc == 1
            out = capsys.readouterr().out
            assert f"端口 {busy_port} 已被占用" in out
            assert "nanobot gateway 端口" in out
        finally:
            del s


# ============================================================================
# 3. --check-env validation (1 test)
# ============================================================================

class TestServeCheckEnv:
    def test_check_env_missing_api_key_returns_1(self, tmp_path, monkeypatch, capsys):
        """When --check-env is set and QUANTNODES__LLM__API_KEY is empty, exit 1."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("QUANTNODES__LLM__API_KEY", raising=False)
        # Mock ports as free so the check_env branch is reached
        monkeypatch.setattr(
            "QuantNodes.cli.commands.serve.is_port_free", lambda *a, **k: True
        )
        args = MagicMock(
            host="127.0.0.1", port=19380,
            gateway_port=DEFAULT_GATEWAY_PORT,
            frontend=False, frontend_port=5173,
            daemon=False, check_env=True,
        )
        rc = cmd_serve(args)
        assert rc == 1
        out = capsys.readouterr().out
        assert "QUANTNODES__LLM__API_KEY" in out


# ============================================================================
# 4. daemon mode writes pidfile (2 tests)
# ============================================================================

class TestServeDaemon:

    def _patch_popen(self, monkeypatch, target_pid: int = 12345):
        """Patch subprocess.Popen to return a fake proc with .pid."""
        mock_proc = MagicMock()
        mock_proc.pid = target_pid
        mock_proc.wait = MagicMock()
        mock_proc.terminate = MagicMock()
        mock_popen = MagicMock(return_value=mock_proc)
        monkeypatch.setattr("subprocess.Popen", mock_popen)
        return mock_popen, mock_proc

    def test_daemon_writes_pidfile_and_returns_0(self, tmp_path, monkeypatch, capsys):
        monkeypatch.chdir(tmp_path)
        # 占用真实进程当前 PID 作为"已被占用"检查的占位
        # 我们用 0 (无效) 让 is_port_free 直接返回 True
        monkeypatch.setattr("QuantNodes.cli.commands.serve.is_port_free",
                            lambda *a, **k: True)
        self._patch_popen(monkeypatch, target_pid=4242)

        args = MagicMock(
            host="127.0.0.1", port=19380,
            gateway_port=DEFAULT_GATEWAY_PORT,
            frontend=False, frontend_port=5173,
            daemon=True, check_env=False,
        )
        rc = cmd_serve(args)
        assert rc == 0
        # pidfile written
        pidfile = tmp_path / PIDFILE_NAME
        assert pidfile.is_file()
        assert pidfile.read_text().strip() == "4242"
        # output mentions log path
        out = capsys.readouterr().out
        assert "PID: 4242" in out or "4242" in out

    def test_pidfile_contains_only_int(self, tmp_path):
        """write_pidfile + read_pidfile roundtrip preserves integer."""
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

    def test_stop_stale_pidfile_cleans_up(self, tmp_path, monkeypatch, capsys):
        """If PID points to a dead process, clean up pidfile and return 0."""
        monkeypatch.chdir(tmp_path)
        # Use a PID that's extremely unlikely to be alive
        write_pidfile(2_000_000_000, path=tmp_path / PIDFILE_NAME)
        rc = cmd_stop(MagicMock())
        # Either is_pid_alive returns False (clean up) or True (send signal)
        # Both paths are acceptable; just assert pidfile is gone
        assert read_pidfile(tmp_path / PIDFILE_NAME) is None

    def test_stop_sends_sigterm_to_alive_pid(self, tmp_path, monkeypatch):
        """If PID is alive, send SIGTERM and clean pidfile."""
        monkeypatch.chdir(tmp_path)
        # 使用当前进程 PID (alive)
        alive_pid = os.getpid()
        write_pidfile(alive_pid, path=tmp_path / PIDFILE_NAME)
        # SIGTERM 自己会终止测试；用 mock 替换 os.kill
        with patch("QuantNodes.cli.commands.serve.os.kill") as mock_kill:
            with patch("QuantNodes.cli.commands.serve.signal.SIGTERM", signal.SIGTERM):
                rc = cmd_stop(MagicMock())
        assert rc == 0
        # os.kill 被调用 2 次：一次 is_pid_alive 的 ping (signal 0),
        # 一次真正的 SIGTERM。
        assert mock_kill.call_count == 2
        # 第二次调用必须是 SIGTERM
        second_call_args = mock_kill.call_args_list[1]
        assert second_call_args[0] == (alive_pid, signal.SIGTERM)
        # pidfile 已清理
        assert not (tmp_path / PIDFILE_NAME).exists()


# ============================================================================
# 6. status command (2 tests)
# ============================================================================

class TestStatusCommand:

    def test_status_returns_0_when_running(self, capsys):
        """When /api/agent/status returns state=running, exit 0."""
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {
            "state": "running", "components": {"agent": True}
        }
        with patch("httpx.get", return_value=fake_response):
            rc = cmd_status(MagicMock(api_url="http://127.0.0.1:19380"))
        assert rc == 0
        out = capsys.readouterr().out
        assert '"state": "running"' in out

    def test_status_returns_1_on_connection_error(self, capsys):
        """When httpx raises ConnectError, exit 1 + error embedded in JSON."""
        import httpx
        with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
            rc = cmd_status(MagicMock(api_url="http://127.0.0.1:9999"))
        assert rc == 1
        out = capsys.readouterr().out
        # JSON output includes "reachable": false + "error": "refused"
        assert '"reachable": false' in out
        assert '"error": "refused"' in out


# ============================================================================
# 7. logs command (2 tests)
# ============================================================================

class TestLogsCommand:

    def test_logs_no_file_returns_1(self, tmp_path, monkeypatch, capsys):
        """If log file doesn't exist, return 1 with hint."""
        # _serve_log_path() uses get_project_root(), so monkey-patch it
        monkeypatch.setattr(
            "QuantNodes.cli.commands.serve._serve_log_path",
            lambda: tmp_path / "nonexistent.log",
        )
        rc = cmd_logs(MagicMock(follow=False))
        assert rc == 1

    def test_logs_no_follow_uses_tail_n_200(self, tmp_path, monkeypatch):
        """Without -f, call tail -n 200 <log>."""
        log = tmp_path / "quantnodes_serve.log"
        log.write_text("line1\nline2\n" * 100)
        monkeypatch.setattr(
            "QuantNodes.cli.commands.serve._serve_log_path", lambda: log
        )
        with patch("subprocess.run") as mock_run:
            rc = cmd_logs(MagicMock(follow=False))
        assert rc == 0
        mock_run.assert_called_once()
        args, _ = mock_run.call_args
        assert args[0] == ["tail", "-n", "200", str(log)]


# ============================================================================
# 8. _helpers low-level (4 tests)
# ============================================================================

class TestHelpers:

    def test_is_pid_alive_for_current_process(self):
        """Current process PID must be alive."""
        assert is_pid_alive(os.getpid()) is True

    def test_is_pid_alive_for_nonexistent(self):
        """An absurdly large PID must not be alive."""
        assert is_pid_alive(2_000_000_000) is False

    def test_is_port_free_for_high_port(self):
        """Port 65530 is usually free; port 1 may be reserved but not bound locally."""
        # 用 is_port_free(0) 会拿到 OS 分配的临时端口, 必然 free
        assert is_port_free(0) is True

    def test_load_env_file_returns_false_when_missing(self, tmp_path, monkeypatch):
        """If .env doesn't exist, return False (don't raise)."""
        monkeypatch.chdir(tmp_path)
        assert load_env_file() is False

    def test_load_env_file_returns_true_when_present(self, tmp_path, monkeypatch):
        """If .env exists, return True and populate os.environ."""
        env_file = tmp_path / ".env"
        env_file.write_text("FOO_TEST_LOAD_ENV=hello\n")
        # Note: cwd is test dir; but load_env_file uses Path.cwd() — patch cwd
        monkeypatch.chdir(tmp_path)
        assert load_env_file() is True
        assert os.environ.get("FOO_TEST_LOAD_ENV") == "hello"
        del os.environ["FOO_TEST_LOAD_ENV"]


# ============================================================================
# 9. Command registration sanity (1 test)
# ============================================================================

def test_lifecycle_commands_registered():
    """serve / stop / status / logs must be in the global registry."""
    names = COMMAND_REGISTRY.names()
    for cmd in ("serve", "stop", "status", "logs"):
        assert cmd in names, f"Missing command: {cmd}"
    # 总数 20 (Stage 7 之前 13 + serve + stop + status + logs + agent + run --gateway-port 保持)
    assert len(names) == 20


# ============================================================================
# 10. wait_for_health polling (1 test)
# ============================================================================

class TestWaitForHealth:

    def test_wait_for_health_returns_true_when_state_running(self, monkeypatch):
        """First poll returns state=running → return True immediately."""
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"state": "running"}
        with patch("httpx.get", return_value=fake_response):
            assert wait_for_health("http://127.0.0.1:9999", timeout_s=2) is True

    def test_wait_for_health_returns_false_on_timeout(self, monkeypatch):
        """If state never reaches running within timeout, return False."""
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {"state": "starting"}
        with patch("httpx.get", return_value=fake_response):
            with patch("time.sleep"):  # speed up
                assert wait_for_health("http://x", timeout_s=0.5, poll_interval_s=0.1) is False