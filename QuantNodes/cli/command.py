# coding=utf-8
"""CLI Command pattern base (Phase 3.1, 2026-06-22).

替代原 cli/__init__.py:159-192 的 34 行 if/elif ladder 派发到 cmd_*
函数。新增 subcommand 只需:
  1. 写一个 Command 子类 (name + add_arguments + run)
  2. 在 commands/__init__.py 注册到 REGISTRY
  3. cli/__init__.py:build_parser 自动加 parser, main() 自动 dispatch

向后兼容: 旧的 cmd_* 函数仍 export, ``from QuantNodes.cli import cmd_init`` 等
调用方式不变。

设计要点:
  - Command ABC: name (str) + description (str) + add_arguments(subparsers) + run(args)
  - CommandRegistry: register / get / all 三个方法, 支持同名注册报错
  - 不修改 ``argparse.Namespace`` 结构: 仍走 args.command / args.* 字段
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import argparse


class Command(ABC):
    """CLI 子命令抽象基类 (Command pattern).

    子类需实现:
      - name: 子命令名 (e.g. "init", "factor-info")
      - description: 简短描述 (用于 help)
      - add_arguments(subparsers): 注册该子命令的 argparse 参数
      - run(args) -> int: 执行子命令, 返回 exit code (0 = 成功)
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def add_arguments(self, subparsers: "argparse._SubParsersAction") -> None:
        """注册 argparse 子命令参数.

        Args:
            subparsers: parser.add_subparsers() 返回的 action,
                         调用 subparsers.add_parser(self.name, ...) 添加.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, args: "argparse.Namespace") -> int:
        """执行子命令. 返回 exit code (0 = 成功, 非 0 = 错误)."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


class CommandRegistry:
    """CLI 子命令注册表 (Phase 3.1).

    提供 register / get / all 三个方法. 重复注册同名 command 抛 ValueError,
    保证 1 个 name 对应 1 个 handler.
    """

    def __init__(self) -> None:
        self._cmds: dict[str, Command] = {}

    def register(self, cmd: Command) -> Command:
        """注册一个 Command. 重复同名抛 ValueError.

        Returns:
            传入的 cmd (便于 ``@REGISTRY.register class ...`` 链式调用)
        """
        if not cmd.name:
            raise ValueError(
                f"{cmd!r} has empty .name, refusing to register"
            )
        if cmd.name in self._cmds:
            existing = self._cmds[cmd.name]
            if existing is cmd:
                return cmd  # idempotent
            raise ValueError(
                f"Command '{cmd.name}' already registered to {existing!r}, "
                f"refusing to overwrite with {cmd!r}"
            )
        self._cmds[cmd.name] = cmd
        return cmd

    def get(self, name: str) -> Optional[Command]:
        """按 name 查询. 未找到返回 None."""
        return self._cmds.get(name)

    def all(self) -> List[Command]:
        """返回所有注册 command 的 list (顺序与注册顺序一致)."""
        return list(self._cmds.values())

    def names(self) -> List[str]:
        """返回所有注册 name (按注册顺序)."""
        return list(self._cmds.keys())

    def clear(self) -> None:
        """清空注册表 (仅供测试用)."""
        self._cmds.clear()

    def __len__(self) -> int:
        return len(self._cmds)

    def __contains__(self, name: str) -> bool:
        return name in self._cmds
