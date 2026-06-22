# coding=utf-8
"""Command modules for QuantNodes CLI.

Each module exposes the command handler(s) it owns. Public re-exports are
performed by QuantNodes.cli.__init__ so the entry point
``from QuantNodes.cli import main`` keeps working unchanged.

Phase 3.1 (2026-06-22): 新增 COMMAND_REGISTRY (Command pattern).
各 command module 末尾追加 *Command 子类, 此文件在 import 时统一注册到
REGISTRY. cli/__init__.py:build_parser / main 改为用 registry 派发.
"""
from QuantNodes.cli.command import CommandRegistry

# 模块级 registry (单例)
COMMAND_REGISTRY = CommandRegistry()


def _register_all() -> None:
    """注册所有 command. 顺序决定 argparse help 中子命令的显示顺序."""
    # 顺序: 初始化 / 启动 / 对话 / 演化 / factor-* (按字母) / version / help
    from QuantNodes.cli.commands.init import InitCommand
    from QuantNodes.cli.commands.run import RunCommand
    from QuantNodes.cli.commands.chat import ChatCommand
    from QuantNodes.cli.commands.evolve import EvolveCommand
    from QuantNodes.cli.commands.factor import (
        FactorInfoCommand,
        FactorBestCommand,
        FactorVisualCommand,
        FactorDashboardCommand,
        FactorDataFetchCommand,
        FactorRagEvalCommand,
        FactorRagShowCommand,
    )
    from QuantNodes.cli.commands.version import VersionCommand, HelpCommand

    for cmd in [
        InitCommand(),
        RunCommand(),
        ChatCommand(),
        EvolveCommand(),
        FactorInfoCommand(),
        FactorBestCommand(),
        FactorVisualCommand(),
        FactorRagShowCommand(),
        FactorRagEvalCommand(),
        FactorDataFetchCommand(),
        FactorDashboardCommand(),
        VersionCommand(),
        HelpCommand(),
    ]:
        COMMAND_REGISTRY.register(cmd)


_register_all()
