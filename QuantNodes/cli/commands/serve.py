# coding=utf-8
"""``quantnodes serve / stop / status / logs`` — backend lifecycle.

v3.0.0 Stage 7: 一键启动 / 停止 / 状态查询 / 日志查看。
对标 llmwikify 的 ``llmwikify <cmd>`` 风格。

Commands:
    serve   — 启动 FastAPI (+ nanobot gateway), 可选前端 (--frontend)
    stop    — 通过 pidfile 停止 serve
    status  — 打印 health + /api/agent/status JSON
    logs    — tail -f logs/quantnodes_serve.log

Design:
    - serve 用 subprocess.Popen 启动 uvicorn (与 run.py 一致)
    - --daemon 时写 .quantnodes.pid 并立即返回 0
    - 前台时 wait_for_health 轮询直到 state=running, 然后等 Ctrl+C
    - 端口冲突检测在启动前 (避免 nanobot 报 OSError 才知道 18080 被 gpustack 占用)
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

from QuantNodes.cli.command import Command

from .._helpers import (
    DEFAULT_API_PORT,
    DEFAULT_FRONTEND_PORT,
    DEFAULT_GATEWAY_PORT,
    DEFAULT_HOST,
    PIDFILE_NAME,
    get_project_root,
    is_pid_alive,
    is_port_free,
    print_nanobot_install_hint,
    read_pidfile,
    remove_pidfile,
    wait_for_health,
    write_pidfile,
)


def _log_dir() -> Path:
    """Return logs/ dir at project root, creating it on first call."""
    log_dir = get_project_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _serve_log_path() -> Path:
    """Path to the rolling serve log file."""
    return _log_dir() / "quantnodes_serve.log"


def _start_uvicorn(host: str, port: int, gateway_port: int,
                   log_path: Path) -> subprocess.Popen:
    """Spawn uvicorn with NANOBOT_GATEWAY_PORT injected into the subprocess env.

    Returns the Popen handle (caller is responsible for .wait() / .terminate()).
    Stdout+stderr are appended to ``log_path`` so the daemon case leaves an
    audit trail that ``quantnodes logs`` can tail.
    """
    env = os.environ.copy()
    env["NANOBOT_GATEWAY_HOST"] = host
    env["NANOBOT_GATEWAY_PORT"] = str(gateway_port)
    cmd = [
        sys.executable, "-m", "uvicorn", "api.main:app",
        "--host", host, "--port", str(port),
    ]
    log_fd = open(log_path, "ab", buffering=0)
    return subprocess.Popen(
        cmd, cwd=str(get_project_root()), env=env,
        stdout=log_fd, stderr=subprocess.STDOUT,
    )


def _start_frontend(host: str, frontend_port: int, api_port: int,
                    gateway_port: int = DEFAULT_GATEWAY_PORT) -> subprocess.Popen:
    """Spawn Vite dev server (npm run dev) with HOST/PORT/API_PORT injected.

    v3.0.0 Stage 7: also inject ``VITE_NANOBOT_GATEWAY_URL`` and ``GATEWAY_PORT``
    so the Vue frontend's AgentChat.vue can reach the nanobot gateway
    for WebSocket chat and HTTP APIs (sessions/settings/mcp).
    Stdout/stderr are inherited (frontend's own dev output is useful for the user).
    """
    env = os.environ.copy()
    env["HOST"] = host
    env["PORT"] = str(frontend_port)
    env["API_PORT"] = str(api_port)
    # v3.0.0: inject gateway port so AgentChat.vue resolves the correct
    # WebSocket + HTTP API endpoint (avoids hardcoded 18080).
    env["GATEWAY_PORT"] = str(gateway_port)
    env["VITE_NANOBOT_GATEWAY_URL"] = f"http://{host}:{gateway_port}/"
    return subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(get_project_root() / "frontend"),
        env=env,
    )


def cmd_serve(args) -> int:
    """Foreground by default; --daemon writes .quantnodes.pid and returns 0."""
    if not load_env_quietly():
        print("⚠ .env 未找到（继续运行，部分功能可能受限）")
        print("  建议运行: quantnodes init  生成 .env")

    # 1. 端口冲突检测（明确报错，比 nanobot 内部 OSError 更友好）
    if not is_port_free(args.port):
        print(f"✗ 后端端口 {args.port} 已被占用")
        print(f"  提示: 用 --port 换其他端口 (默认 {DEFAULT_API_PORT})")
        return 1
    if not is_port_free(args.gateway_port):
        print(f"✗ nanobot gateway 端口 {args.gateway_port} 已被占用")
        if args.gateway_port == 18080:
            print("  提示: 18080 常被 gpustack 等占用，建议换 --gateway-port 18090")
        print(f"  当前默认: {DEFAULT_GATEWAY_PORT}")
        return 1

    # 2. --check-env 时验证 .env 中 QUANTNODES__LLM__API_KEY
    if args.check_env and not os.environ.get("QUANTNODES__LLM__API_KEY"):
        print("✗ .env 中缺少 QUANTNODES__LLM__API_KEY")
        print("  请运行: quantnodes init  或手动编辑 .env")
        return 1

    # 3. 启动后端
    log_path = _serve_log_path()
    api_proc = _start_uvicorn(args.host, args.port, args.gateway_port, log_path)

    # 4. 可选前端
    frontend_proc: Optional[subprocess.Popen] = None
    if args.frontend:
        frontend_proc = _start_frontend(args.host, args.frontend_port, args.port,
                                             gateway_port=args.gateway_port)

    # 5. daemon / 前台 分支
    if args.daemon:
        write_pidfile(api_proc.pid)
        api_url = f"http://{args.host}:{args.port}"
        gateway_url = f"http://{args.host}:{args.gateway_port}"
        print("=" * 50)
        print("✓ QuantNodes 已后台启动")
        print("=" * 50)
        print(f"  API:       {api_url}")
        print(f"  WebUI:     {gateway_url}/")
        print(f"  PID:       {api_proc.pid}    日志: {log_path}")
        print(f"  停止:      quantnodes stop")
        print(f"  状态:      quantnodes status")
        print()
        print_nanobot_install_hint()
        return 0

    # 前台模式：等 health 成功后打印 ready 信息
    api_url = f"http://{args.host}:{args.port}"
    print(f"⏳ 等待后端就绪 (timeout 30s)...")
    if wait_for_health(api_url, timeout_s=30):
        print()
        print("=" * 50)
        print("✓ QuantNodes 后端已就绪")
        print("=" * 50)
        print(f"  API:       {api_url}")
        print(f"  WebUI:     http://{args.host}:{args.gateway_port}/")
        if args.frontend:
            print(f"  Frontend:  http://{args.host}:{args.frontend_port}/")
        print(f"  日志:      tail -f {log_path}")
        print()
        print("按 Ctrl+C 停止")
        print()
        print_nanobot_install_hint()
    else:
        print("⚠ 后端 30s 内未就绪，请检查:")
        print(f"  tail -f {log_path}")

    try:
        api_proc.wait()
    except KeyboardInterrupt:
        print("\n正在停止...")
        api_proc.terminate()
        if frontend_proc:
            try:
                frontend_proc.terminate()
            except Exception:
                pass
        try:
            api_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            api_proc.kill()
    return 0


def load_env_quietly() -> bool:
    """Call load_env_file without printing on missing optional dep."""
    try:
        from .._helpers import load_env_file
        return load_env_file()
    except Exception:
        return False


def cmd_stop(args) -> int:
    """Send SIGTERM to the PID stored in .quantnodes.pid."""
    pid = read_pidfile()
    if pid is None:
        print(f"未找到 {PIDFILE_NAME}，服务可能未启动或已手动停止")
        return 1
    if not is_pid_alive(pid):
        print(f"PID {pid} 不存在（stale pidfile，已清理）")
        remove_pidfile()
        return 0
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"✓ 已发送 SIGTERM 到 PID {pid}")
    except ProcessLookupError:
        print(f"PID {pid} 已退出")
    except PermissionError:
        print(f"✗ 无权终止 PID {pid}（可能属于其他用户）")
        return 1
    # 清理 pidfile（即使进程还没完全退出，下一次 stop 不会重复 kill）
    remove_pidfile()
    print(f"  pidfile 已清理: {PIDFILE_NAME}")
    return 0


def cmd_status(args) -> int:
    """Print pidfile + health endpoint + /api/agent/status as one JSON summary."""
    api_url = args.api_url.rstrip("/")
    pid = read_pidfile()
    pid_alive = is_pid_alive(pid) if pid else False

    summary: dict = {
        "pidfile": {
            "path": str(Path.cwd() / PIDFILE_NAME),
            "pid": pid,
            "alive": pid_alive,
        },
        "api_url": api_url,
    }

    try:
        import httpx
        r = httpx.get(f"{api_url}/api/agent/status", timeout=3.0)
        if r.status_code == 200:
            summary["agent_status"] = r.json()
            summary["reachable"] = True
        else:
            summary["reachable"] = False
            summary["http_status"] = r.status_code
    except Exception as e:
        summary["reachable"] = False
        summary["error"] = str(e)

    import json
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary.get("agent_status", {}).get("state") == "running" else 1


def cmd_logs(args) -> int:
    """tail -f logs/quantnodes_serve.log (or cat if --no-follow)."""
    log_path = _serve_log_path()
    if not log_path.is_file():
        print(f"日志文件不存在: {log_path}")
        print("（需要先运行 quantnodes serve）")
        return 1
    if args.follow:
        # Use `tail -F` so logrotate / truncation triggers reopen
        try:
            subprocess.run(["tail", "-F", "-n", "+1", str(log_path)])
        except KeyboardInterrupt:
            pass
    else:
        # 打印最后 200 行后退出
        try:
            subprocess.run(["tail", "-n", "200", str(log_path)])
        except FileNotFoundError:
            # tail 不存在（Windows）；fallback 到 Python read
            text = log_path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()[-200:]
            print("\n".join(lines))
    return 0


# ============================================================================
# Command classes (registry pattern)
# ============================================================================

class ServeCommand(Command):
    """``quantnodes serve`` — 启动后端（+ 可选前端）."""

    name = "serve"
    description = "启动 QuantNodes 后端（FastAPI + nanobot gateway）"

    def add_arguments(self, subparsers) -> None:
        p = subparsers.add_parser(self.name, help=self.description)
        p.add_argument("--host", default=DEFAULT_HOST, help=f"绑定主机 (默认 {DEFAULT_HOST})")
        p.add_argument("--port", type=int, default=DEFAULT_API_PORT,
                       help=f"后端端口 (默认 {DEFAULT_API_PORT})")
        p.add_argument("--gateway-port", type=int, default=DEFAULT_GATEWAY_PORT,
                       help=f"nanobot WebSocket gateway 端口 (默认 {DEFAULT_GATEWAY_PORT})")
        p.add_argument("--frontend", action="store_true",
                       help="同时启动 Vite dev server (默认仅启动后端)")
        p.add_argument("--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT,
                       help=f"前端端口 (默认 {DEFAULT_FRONTEND_PORT})")
        p.add_argument("--daemon", action="store_true",
                       help="后台运行，写入 .quantnodes.pid")
        p.add_argument("--check-env", action="store_true",
                       help="启动前校验 .env 中 QUANTNODES__LLM__API_KEY")

    def run(self, args) -> int:
        return cmd_serve(args)


class StopCommand(Command):
    """``quantnodes stop`` — 停止 serve（通过 .quantnodes.pid）."""

    name = "stop"
    description = "通过 pidfile 停止 quantnodes serve"

    def add_arguments(self, subparsers) -> None:
        subparsers.add_parser(self.name, help=self.description)

    def run(self, args) -> int:
        return cmd_stop(args)


class StatusCommand(Command):
    """``quantnodes status`` — 综合服务状态（pidfile + /api/agent/status）."""

    name = "status"
    description = "查服务运行状态（pidfile + /api/agent/status）"

    def add_arguments(self, subparsers) -> None:
        p = subparsers.add_parser(self.name, help=self.description)
        p.add_argument("--api-url", default=f"http://{DEFAULT_HOST}:{DEFAULT_API_PORT}",
                       help=f"FastAPI base URL (默认 http://{DEFAULT_HOST}:{DEFAULT_API_PORT})")

    def run(self, args) -> int:
        return cmd_status(args)


class LogsCommand(Command):
    """``quantnodes logs`` — 查看 / tail serve 日志."""

    name = "logs"
    description = "查看 serve 日志 (logs/quantnodes_serve.log)"

    def add_arguments(self, subparsers) -> None:
        p = subparsers.add_parser(self.name, help=self.description)
        p.add_argument("-f", "--follow", action="store_true",
                       help="实时滚动 (默认只打印最后 200 行)")

    def run(self, args) -> int:
        return cmd_logs(args)