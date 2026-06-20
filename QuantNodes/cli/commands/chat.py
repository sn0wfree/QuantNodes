# coding=utf-8
"""``quantnodes chat`` command (thin wrapper around enhanced chat)."""


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
