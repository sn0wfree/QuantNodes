# coding=utf-8
"""``quantnodes chat`` command (thin wrapper around enhanced chat).

Phase 3.1 (2026-06-22): 改为 Command pattern — ChatCommand.
旧的 cmd_chat 函数保留作 backward compat.
"""

from QuantNodes.cli.command import Command


def cmd_chat(args):
    """启动 Agent 对话模式"""
    from .. import enhanced as _enhanced

    chat = _enhanced.chat
    chat_single = _enhanced.chat_single

    workspace = args.workspace
    if args.message:
        chat_single(args.message, workspace=workspace)
    else:
        chat(workspace=workspace)
    return 0


class ChatCommand(Command):
    """``quantnodes chat`` subcommand."""

    name = "chat"
    description = "启动 Agent 对话模式"

    def add_arguments(self, subparsers) -> None:
        p = subparsers.add_parser(self.name, help=self.description)
        p.add_argument("message", nargs="?", help="单次提问（不指定则进入交互模式）")
        p.add_argument("--workspace", default=".", help="工作目录")

    def run(self, args) -> int:
        return cmd_chat(args)
