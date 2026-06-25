# coding=utf-8
"""``quantnodes agent {status,chat,restart}`` — HTTP client for the nanobot runtime.

v3.0.0 Stage 7: 通过 HTTP 调用 ``/api/agent/*`` endpoints,
不需要 CLI 进程安装 ``nanobot-ai`` (相反, ``quantnodes chat``
走 CLI 直连, 需要 [agent] extra).

Subcommands (sub-subparsers):
    status   — GET  /api/agent/status
    chat     — POST /api/agent/chat/send  (单次问答)
    restart  — POST /api/agent/restart

对比 ``quantnodes chat``:
    chat       CLI 进程内 Agent.run()  →  需 [agent] extra
    agent chat HTTP 调用 FastAPI    →  仅需后端在跑
"""
from __future__ import annotations

import json
import sys
from typing import Optional

from QuantNodes.cli.command import Command

from .._helpers import DEFAULT_API_PORT, DEFAULT_HOST


def _api_client(api_url: str, timeout_s: float = 5.0):
    """Return httpx module (local import keeps CLI importable without httpx)."""
    import httpx
    return httpx


def cmd_agent_status(args) -> int:
    """GET /api/agent/status — pretty-print JSON; return 0 if running."""
    httpx = _api_client(args.api_url)
    try:
        r = httpx.get(f"{args.api_url}/api/agent/status", timeout=5.0)
        r.raise_for_status()
        data = r.json()
    except httpx.ConnectError:
        print(f"✗ 无法连接 {args.api_url}，请先运行: quantnodes serve",
              file=sys.stderr)
        return 1
    except Exception as e:
        print(f"✗ 请求失败: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(json.dumps(data, indent=2, ensure_ascii=False))
    state = data.get("state", "unknown")
    return 0 if state == "running" else 1


def cmd_agent_chat(args) -> int:
    """POST /api/agent/chat/send — print response content (or error)."""
    httpx = _api_client(args.api_url)
    try:
        r = httpx.post(
            f"{args.api_url}/api/agent/chat/send",
            json={"message": args.message, "session_id": args.session},
            timeout=120.0,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.ConnectError:
        print(f"✗ 无法连接 {args.api_url}，请先运行: quantnodes serve",
              file=sys.stderr)
        return 1
    except Exception as e:
        print(f"✗ 请求失败: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    if data.get("error"):
        print(f"[Error] {data['error']}", file=sys.stderr)
        return 1
    content = data.get("content", "")
    if content:
        print(content)
    return 0


def cmd_agent_restart(args) -> int:
    """POST /api/agent/restart — 重启 nanobot runtime."""
    httpx = _api_client(args.api_url)
    try:
        r = httpx.post(f"{args.api_url}/api/agent/restart", timeout=10.0)
    except httpx.ConnectError:
        print(f"✗ 无法连接 {args.api_url}，请先运行: quantnodes serve",
              file=sys.stderr)
        return 1
    except Exception as e:
        print(f"✗ 请求失败: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    try:
        data = r.json()
    except Exception:
        data = {"http_status": r.status_code, "text": r.text}

    print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0 if r.status_code == 200 else 1


# ============================================================================
# Command class — agent is a sub-subparser group (status / chat / restart)
# ============================================================================

class AgentCommand(Command):
    """``quantnodes agent {status,chat,restart}`` — HTTP 客户端调用后端."""

    name = "agent"
    description = "通过 HTTP 调用 nanobot runtime（不需 CLI 装 nanobot-ai）"

    def add_arguments(self, subparsers) -> None:
        p = subparsers.add_parser(self.name, help=self.description)
        # 第二层 subparsers: status / chat / restart
        sub = p.add_subparsers(
            dest="agent_action",
            required=True,
            metavar="{status,chat,restart}",
            help="agent 子命令",
        )

        sp_status = sub.add_parser("status", help="查 nanobot runtime 状态")
        sp_status.add_argument(
            "--api-url",
            default=f"http://{DEFAULT_HOST}:{DEFAULT_API_PORT}",
            help=f"FastAPI base URL (默认 http://{DEFAULT_HOST}:{DEFAULT_API_PORT})",
        )

        sp_chat = sub.add_parser("chat", help="通过 HTTP 发送一条 chat 消息")
        sp_chat.add_argument("message", help="消息内容")
        sp_chat.add_argument("--session", default="default",
                             help="session_id (默认 default)")
        sp_chat.add_argument(
            "--api-url",
            default=f"http://{DEFAULT_HOST}:{DEFAULT_API_PORT}",
            help=f"FastAPI base URL (默认 http://{DEFAULT_HOST}:{DEFAULT_API_PORT})",
        )

        sp_restart = sub.add_parser("restart", help="重启 nanobot runtime")
        sp_restart.add_argument(
            "--api-url",
            default=f"http://{DEFAULT_HOST}:{DEFAULT_API_PORT}",
            help=f"FastAPI base URL (默认 http://{DEFAULT_HOST}:{DEFAULT_API_PORT})",
        )

    def run(self, args) -> int:
        action = getattr(args, "agent_action", None)
        if action == "status":
            return cmd_agent_status(args)
        if action == "chat":
            return cmd_agent_chat(args)
        if action == "restart":
            return cmd_agent_restart(args)
        # argparse 的 required=True 不会让 action 为空；保留 fallback
        print(f"未知 agent 子命令: {action!r}", file=sys.stderr)
        return 1