# coding=utf-8
"""
Agent命令行接口
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

from ..bus.queue import MessageBus
from ..providers.base import LLMProvider
from ..tools.echo import EchoTool
from ..core.loop import AgentLoop
from QuantNodes.core.path_utils import ensure_dir


def _get_default_provider() -> Optional[LLMProvider]:
    """获取默认LLM Provider"""
    try:
        from QuantNodes.ai.llm.openai import OpenAIClient
        from ..providers.quantnodes import QuantNodesLLMProvider

        client = OpenAIClient()
        return QuantNodesLLMProvider(client)
    except Exception as e:
        print(f"Warning: Could not initialize LLM provider: {e}", file=sys.stderr)
        return None


async def run_interactive(loop: AgentLoop) -> None:
    """交互式对话模式"""
    print("QuantNodes Agent 交互模式")
    print("输入 'quit' 或 'exit' 退出")
    print("-" * 50)

    session_id = "cli_interactive"

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("再见!")
            break

        print("思考中...", end="\r")
        response = await loop.chat(user_input, session_id=session_id)
        print(" " * 20, end="\r")
        print(response)


async def run_single(loop: AgentLoop, message: str) -> None:
    """单次执行模式"""
    response = await loop.chat(message, session_id="cli_single")
    print(response)


def main():
    """CLI主入口"""
    import argparse

    parser = argparse.ArgumentParser(description="QuantNodes Agent CLI")
    parser.add_argument(
        "message",
        nargs="*",
        help="要发送的消息（如果省略则进入交互模式）",
    )
    parser.add_argument(
        "--workspace",
        default="./.quant_agent",
        help="工作目录路径",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="使用的模型名称",
    )
    parser.add_argument(
        "--no-echo",
        action="store_true",
        help="不注册echo测试工具",
    )

    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    ensure_dir(workspace)

    bus = MessageBus()
    provider = _get_default_provider()

    if not provider:
        print("Error: No LLM provider available", file=sys.stderr)
        sys.exit(1)

    loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=workspace,
        model=args.model,
    )

    if not args.no_echo:
        loop.register_tool(EchoTool())

    message = " ".join(args.message).strip()

    if message:
        asyncio.run(run_single(loop, message))
    else:
        asyncio.run(run_interactive(loop))


if __name__ == "__main__":
    main()
