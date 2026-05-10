# coding=utf-8
"""
Enhanced CLI - Rich Agent 交互终端

提供流式输出、Markdown 渲染、命令历史的 Agent 对话界面。
"""

import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text


console = Console()


async def _stream_chat(agent, message: str) -> None:
    """流式输出 Agent 响应"""
    full_response = ""
    try:
        async for chunk in agent.chat(message):
            if chunk:
                full_response += chunk
                console.print(chunk, end="", highlight=False)
        console.print()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        return

    if full_response:
        console.print()
        try:
            console.print(Markdown(full_response))
        except Exception:
            pass


def _print_welcome() -> None:
    """打印欢迎信息"""
    welcome = Text()
    welcome.append("QuantNodes Agent", style="bold cyan")
    welcome.append(" - 量化研究智能助手\n", style="dim")
    welcome.append("输入问题开始对话，", style="dim")
    welcome.append("exit", style="bold")
    welcome.append(" 退出\n", style="dim")
    console.print(Panel(welcome, border_style="cyan"))


def _print_help() -> None:
    """打印帮助信息"""
    help_text = """
**可用命令：**
- `exit` / `quit` / `q` - 退出
- `help` - 显示此帮助
- `clear` - 清屏
- `history` - 查看对话历史
- 直接输入文本开始对话
"""
    console.print(Markdown(help_text))


def chat(workspace: str = ".", config: dict = None) -> None:
    """启动 Agent 对话模式

    Args:
        workspace: 工作目录
        config: Agent 配置
    """
    from QuantNodes.agent import Agent

    console.print("[bold cyan]正在初始化 Agent...[/bold cyan]")

    try:
        agent = Agent(workspace=workspace, config=config or {})
    except Exception as e:
        console.print(f"[red]Agent 初始化失败: {e}[/red]")
        return

    _print_welcome()

    history: list[str] = []

    while True:
        try:
            user_input = console.input("[bold green]> [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]再见！[/dim]")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("exit", "quit", "q"):
            console.print("[dim]再见！[/dim]")
            break
        elif cmd == "help":
            _print_help()
            continue
        elif cmd == "clear":
            console.clear()
            continue
        elif cmd == "history":
            if history:
                for i, msg in enumerate(history, 1):
                    console.print(f"[dim]{i}.[/dim] {msg[:80]}...")
            else:
                console.print("[dim]暂无历史记录[/dim]")
            continue

        history.append(user_input)

        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_stream_chat(agent, user_input))
            loop.close()
        except KeyboardInterrupt:
            console.print("\n[yellow]已中断[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


def chat_single(message: str, workspace: str = ".", config: dict = None) -> None:
    """单次问答模式

    Args:
        message: 用户消息
        workspace: 工作目录
        config: Agent 配置
    """
    from QuantNodes.agent import Agent

    try:
        agent = Agent(workspace=workspace, config=config or {})
    except Exception as e:
        console.print(f"[red]Agent 初始化失败: {e}[/red]")
        return

    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_stream_chat(agent, message))
        loop.close()
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
