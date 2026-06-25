# coding=utf-8
"""``quantnodes run`` command + server start helpers."""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Any, List

from QuantNodes.core.path_utils import ensure_parent
from QuantNodes.cli.command import Command

from .._helpers import (
    DEFAULT_API_PORT,
    DEFAULT_FRONTEND_PORT,
    DEFAULT_GATEWAY_PORT,
    DEFAULT_HOST,
    get_project_root,
    is_initialized,
)


def start_api_server(
    host: str,
    port: int,
    log_file: Optional[Path] = None,
    gateway_port: Optional[int] = None,
) -> Tuple[subprocess.Popen, Optional[Any]]:
    """Start the API server. Returns (process, log_file_handle).

    v3.0.0 Stage 7: ``gateway_port`` (default ``DEFAULT_GATEWAY_PORT=18090``)
    is injected into the subprocess env as ``NANOBOT_GATEWAY_PORT`` so the
    nanobot WebSocket gateway binds to a port not occupied by gpustack
    (which defaults to 18080).
    """
    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", host,
        "--port", str(port),
        "--reload"
    ]

    env = os.environ.copy()
    env["NANOBOT_GATEWAY_HOST"] = host
    env["NANOBOT_GATEWAY_PORT"] = str(gateway_port or DEFAULT_GATEWAY_PORT)

    if log_file:
        ensure_parent(log_file)
        log_fd = open(log_file, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            cwd=get_project_root(),
            env=env,
        )
        return proc, log_fd
    else:
        proc = subprocess.Popen(
            cmd,
            cwd=get_project_root(),
            env=env,
        )
        return proc, None


def start_frontend_server(
    host: str,
    port: int,
    api_port: int = 8000,
    log_file: Optional[Path] = None,
) -> Tuple[subprocess.Popen, Optional[Any]]:
    """Start the frontend server. Returns (process, log_file_handle)."""
    cmd = ["npm", "run", "dev"]

    env = os.environ.copy()
    env["HOST"] = host
    env["PORT"] = str(port)
    env["API_PORT"] = str(api_port)

    if log_file:
        ensure_parent(log_file)
        log_fd = open(log_file, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            cwd=str(get_project_root() / "frontend"),
            env=env
        )
        return proc, log_fd
    else:
        proc = subprocess.Popen(
            cmd,
            cwd=str(get_project_root() / "frontend"),
            env=env
        )
        return proc, None


def cmd_run(args) -> int:
    """Start QuantNodes services."""
    if not is_initialized():
        print("错误: 当前目录未初始化")
        print("请先运行: quantnodes init")
        return 1

    host = args.host or DEFAULT_HOST
    frontend_port = args.port or DEFAULT_FRONTEND_PORT
    # 联动：如果只设置 --port，则 api_port = port + 1000
    if args.port and not args.api_port:
        api_port = args.port + 1000
    else:
        api_port = args.api_port or DEFAULT_API_PORT

    if args.daemon:
        if sys.platform != "linux":
            print("错误: daemon 模式仅支持 Linux")
            return 1

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        api_log = log_dir / f"quantnodes_api_{timestamp}.log"
        frontend_log = log_dir / f"quantnodes_frontend_{timestamp}.log"

        print("=" * 50)
        print("QuantNodes 服务 (后台运行)")
        print("=" * 50)
        print(f"  后端: http://{host}:{api_port}")
        print(f"  前端: http://{host}:{frontend_port}")
        print(f"  API 日志: {api_log}")
        print(f"  前端日志: {frontend_log}")
        print()

        api_proc, api_fd = start_api_server(host, api_port, api_log,
                                             gateway_port=args.gateway_port)
        frontend_proc, frontend_fd = start_frontend_server(
            host, frontend_port, api_port, frontend_log,
        )

        print("✓ 服务已后台启动")
        print(f"  API 进程: {api_proc.pid}")
        print(f"  前端进程: {frontend_proc.pid}")
        print(f"  nanobot gateway: ws://{host}:{args.gateway_port}")
        print()
        print("查看日志:")
        print(f"  tail -f {api_log}")
        print(f"  tail -f {frontend_log}")
        print()
        print("停止服务:")
        print(f"  kill {api_proc.pid} {frontend_proc.pid}")
        print(f"  (quantnodes stop 仅作用于 serve --daemon 启动的服务)")

        return 0

    print("=" * 50)
    print("QuantNodes 服务")
    print("=" * 50)

    processes: List[Tuple[str, subprocess.Popen]] = []
    log_fds: List[Any] = []

    try:
        if not args.frontend_only:
            print(f"\n启动后端: http://{host}:{api_port}")
            print(f"  nanobot gateway: ws://{host}:{args.gateway_port}")
            api_proc, api_fd = start_api_server(host, api_port,
                                                 gateway_port=args.gateway_port)
            processes.append(("API", api_proc))
            log_fds.append(api_fd)
            print(f"  进程 PID: {api_proc.pid}")

        if not args.api_only:
            print(f"\n启动前端: http://{host}:{frontend_port}")
            # Wait for backend to be ready before starting frontend
            import time
            import urllib.request
            import urllib.error
            print("  等待后端就绪...")
            for i in range(30):
                try:
                    urllib.request.urlopen(f"http://localhost:{api_port}/docs", timeout=2)
                    print("  ✓ 后端已就绪")
                    break
                except (urllib.error.URLError, OSError):
                    time.sleep(1)
            else:
                print("  ⚠ 后端未就绪，继续启动前端")
            frontend_proc, frontend_fd = start_frontend_server(host, frontend_port, api_port)
            processes.append(("Frontend", frontend_proc))
            log_fds.append(frontend_fd)
            print(f"  进程 PID: {frontend_proc.pid}")

        print()
        print("=" * 50)
        print("✓ 服务已启动")
        print("=" * 50)
        print()
        print("访问:")
        if not args.frontend_only:
            print(f"  后端: http://localhost:{api_port}/docs")
        if not args.api_only:
            print(f"  前端: http://localhost:{frontend_port}")
        print()
        print("按 Ctrl+C 停止服务")
        print()

        try:
            for name, proc in processes:
                proc.wait()
        except KeyboardInterrupt:
            print("\n\n正在停止服务...")
            for name, proc in processes:
                proc.terminate()
                proc.wait()
            for fd in log_fds:
                if fd:
                    fd.close()
            print("✓ 服务已停止")

    except Exception as e:
        print(f"错误: {e}")
        for name, proc in processes:
            proc.terminate()
        for fd in log_fds:
            if fd:
                fd.close()
        return 1

    return 0


class RunCommand(Command):
    """``quantnodes run`` subcommand."""

    name = "run"
    description = "启动服务"

    def add_arguments(self, subparsers) -> None:
        p = subparsers.add_parser(self.name, help=self.description)
        p.add_argument("--host", help="绑定主机")
        p.add_argument("--port", type=int, help="前端端口")
        p.add_argument("--api-port", type=int, dest="api_port", help="后端端口")
        # v3.0.0 Stage 7: nanobot WebSocket gateway 端口（注入到子进程 env）
        p.add_argument("--gateway-port", type=int, dest="gateway_port",
                       default=DEFAULT_GATEWAY_PORT,
                       help=f"nanobot WebSocket gateway 端口 (默认 {DEFAULT_GATEWAY_PORT})")
        p.add_argument("--daemon", action="store_true", help="后台运行 (仅 Linux)")
        p.add_argument("--api-only", action="store_true", dest="api_only", help="仅启动后端")
        p.add_argument(
            "--frontend-only", action="store_true", dest="frontend_only", help="仅启动前端"
        )

    def run(self, args) -> int:
        return cmd_run(args)
