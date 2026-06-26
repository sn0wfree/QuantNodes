# coding=utf-8
"""``quantnodes serve / stop / status / logs`` — backend lifecycle.

v3.1.0: Zero-subprocess architecture.

Commands:
    serve   — 启动 FastAPI (+ nanobot gateway), 可选前端/ MCP
    stop    — 通过 pidfile 停止 serve
    status  — 打印 health + /api/agent/status JSON
    logs    — tail -f logs/quantnodes_serve.log

Design:
    - serve 直接调用 uvicorn.Server.run() (同进程, 无子进程)
    - --daemon 时双 fork 脱离终端, 写 .quantnodes.pid
    - --frontend 时 spawn npm dev server (subprocess, lifespan 管理)
    - --mcp 时 spawn MCP server (subprocess, 供外部客户端)
    - 前台时 Ctrl+C 直接终止 (所有组件 in-process)
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional

from QuantNodes.cli.command import Command

from QuantNodes.constants import DEFAULT_WEBSOCKET_PORT

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


def _read_gateway_token() -> str:
    """Read the gateway tokenIssueSecret from .agent/nanobot_config.json."""
    try:
        cfg_path = get_project_root() / ".agent" / "nanobot_config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            return cfg.get("channels", {}).get("websocket", {}).get("token", "")
    except Exception:
        pass
    return ""


def _log_dir() -> Path:
    """Return logs/ dir at project root, creating it on first call."""
    log_dir = get_project_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _serve_log_path() -> Path:
    """Path to the rolling serve log file."""
    return _log_dir() / "quantnodes_serve.log"


def _start_frontend(host: str, frontend_port: int, api_port: int,
                    gateway_port: int = DEFAULT_GATEWAY_PORT) -> subprocess.Popen:
    """Spawn Vite dev server (npm run dev) with HOST/PORT/API_PORT injected.

    The frontend subprocess lifecycle is managed by the caller's finally
    block (terminate on shutdown).
    """
    env = os.environ.copy()
    env["HOST"] = host
    env["PORT"] = str(frontend_port)
    env["API_PORT"] = str(api_port)
    env["GATEWAY_PORT"] = str(gateway_port)
    env["VITE_NANOBOT_GATEWAY_URL"] = f"http://{host}:{gateway_port}/"
    return subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(get_project_root() / "frontend"),
        env=env,
    )


def _start_mcp_server(host: str, port: int) -> subprocess.Popen:
    """Spawn MCP server as HTTP subprocess for external clients.

    The MCP server exposes quant tools over MCP protocol (HTTP transport)
    so that Claude Desktop, Cursor, etc. can connect.
    """
    cmd = [
        sys.executable, "-m", "QuantNodes.mcp_server",
        "--transport", "http",
        "--host", host,
        "--port", str(port),
    ]
    return subprocess.Popen(
        cmd, cwd=str(get_project_root()),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _print_startup_info(args, pid: int) -> None:
    """Print startup info (used in daemon mode)."""
    api_url = f"http://{args.host}:{args.port}"
    gateway_url = f"http://{args.host}:{args.gateway_port}"
    log_path = _serve_log_path()
    print("=" * 50)
    print("✓ QuantNodes 已后台启动")
    print("=" * 50)
    print(f"  API:       {api_url}")
    print(f"  WebUI:     {gateway_url}/")
    _token = _read_gateway_token()
    if _token:
        print(f"  Token:     {_token}")
    if getattr(args, "mcp", False):
        print(f"  MCP:       http://{args.host}:{getattr(args, 'mcp_port', DEFAULT_WEBSOCKET_PORT)}/")
    print(f"  PID:       {pid}    日志: {log_path}")
    print(f"  停止:      quantnodes stop")
    print(f"  状态:      quantnodes status")
    print()
    print_nanobot_install_hint()


def _kill_procs(*procs: Optional[subprocess.Popen]) -> None:
    """Terminate a list of subprocesses gracefully, with SIGKILL fallback."""
    for proc in procs:
        if proc is None or proc.poll() is not None:
            continue
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def cmd_serve(args) -> int:
    """Start the QuantNodes backend.

    Foreground by default; --daemon double-forks and writes .quantnodes.pid.
    uvicorn runs in-process (no subprocess) for clean signal handling.
    """
    if not load_env_quietly():
        print("⚠ .env 未找到（继续运行，部分功能可能受限）")
        print("  建议运行: quantnodes init  生成 .env")

    # 1. Port conflict detection
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
    if getattr(args, "mcp", False) and not is_port_free(getattr(args, "mcp_port", DEFAULT_WEBSOCKET_PORT)):
        print(f"✗ MCP server 端口 {args.mcp_port} 已被占用")
        return 1

    # 2. --check-env
    if args.check_env and not os.environ.get("QUANTNODES__LLM__API_KEY"):
        print("✗ .env 中缺少 QUANTNODES__LLM__API_KEY")
        print("  请运行: quantnodes init  或手动编辑 .env")
        return 1

    # 3. Set environment variables (for lifespan / nanobot)
    os.environ["NANOBOT_GATEWAY_HOST"] = args.host
    os.environ["NANOBOT_GATEWAY_PORT"] = str(args.gateway_port)

    # 4. Optional frontend subprocess
    frontend_proc: Optional[subprocess.Popen] = None
    if args.frontend:
        frontend_proc = _start_frontend(args.host, args.frontend_port, args.port,
                                             gateway_port=args.gateway_port)

    # 5. Optional MCP server subprocess
    mcp_proc: Optional[subprocess.Popen] = None
    if getattr(args, "mcp", False):
        mcp_proc = _start_mcp_server(args.host, getattr(args, "mcp_port", DEFAULT_WEBSOCKET_PORT))

    # 6. Daemon mode: double-fork to detach from terminal
    if args.daemon:
        # Fork 1: exit parent, child becomes session leader
        if os.fork() > 0:
            _kill_procs(frontend_proc, mcp_proc)
            sys.exit(0)
        os.setsid()
        # Fork 2: prevent re-acquiring controlling terminal
        if os.fork() > 0:
            sys.exit(0)
        # Redirect stdio to log file
        sys.stdin = open(os.devnull)
        log_path = _serve_log_path()
        log_fd = open(log_path, "a")
        sys.stdout = sys.stderr = log_fd
        write_pidfile(os.getpid())
        _print_startup_info(args, os.getpid())

    # 7. Run uvicorn in-process (blocking)
    import uvicorn
    config = uvicorn.Config(
        "api.main:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )
    server = uvicorn.Server(config)

    # Signal: SIGTERM → graceful shutdown
    def _shutdown(sig, frame):
        server.should_exit = True
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        _kill_procs(frontend_proc, mcp_proc)
        if not args.daemon:
            remove_pidfile()

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
        try:
            subprocess.run(["tail", "-F", "-n", "+1", str(log_path)])
        except KeyboardInterrupt:
            pass
    else:
        try:
            subprocess.run(["tail", "-n", "200", str(log_path)])
        except FileNotFoundError:
            text = log_path.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()[-200:]
            print("\n".join(lines))
    return 0


# ============================================================================
# Command classes (registry pattern)
# ============================================================================

class ServeCommand(Command):
    """``quantnodes serve`` — 启动后端（+ 可选前端/MCP）."""

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
                       help="同时启动 Vite dev server (开发模式)")
        p.add_argument("--frontend-port", type=int, default=DEFAULT_FRONTEND_PORT,
                       help=f"前端端口 (默认 {DEFAULT_FRONTEND_PORT})")
        p.add_argument("--mcp", action="store_true",
                       help="同时启动 MCP server (供 Claude Desktop / Cursor 等外部客户端)")
        p.add_argument("--mcp-port", type=int, default=DEFAULT_WEBSOCKET_PORT,
                       help=f"MCP server HTTP 端口 (默认 {DEFAULT_WEBSOCKET_PORT})")
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
