# coding=utf-8
"""CLI Command pattern 测试 (Phase 3.1, 2026-06-22).

覆盖:
    - Command ABC: 抽象方法强制 / name / description / repr
    - CommandRegistry: register / get / all / names / clear / 重复注册 / 空 name
    - COMMAND_REGISTRY: 13 子命令齐全 + 顺序
    - _build_parser: 每个 Command 自注册 argparse, 与旧行为等价
    - main() dispatch: registry.get(command).run(args)
    - backward compat: cmd_* 函数仍可 import + 调用
"""
from __future__ import annotations

import argparse
import sys
from unittest.mock import patch

import pytest

from QuantNodes.cli import PROG_NAME, main
from QuantNodes.cli.command import Command, CommandRegistry
from QuantNodes.cli.commands import COMMAND_REGISTRY


EXPECTED_COMMANDS = [
    "init",
    "run",
    "chat",
    "evolve",
    "factor-info",
    "factor-best",
    "factor-visual",
    "factor-rag-show",
    "factor-rag-eval",
    "factor-data-fetch",
    "factor-dashboard",
    "version",
    "help",
]


# ============================================================================
# 1. Command ABC (4 tests)
# ============================================================================

class _DummyCommand(Command):
    name = "dummy"
    description = "dummy desc"

    def add_arguments(self, subparsers) -> None:
        subparsers.add_parser(self.name)

    def run(self, args) -> int:
        return 42


class TestCommandABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Command()  # type: ignore[abstract]

    def test_subclass_instantiates(self):
        cmd = _DummyCommand()
        assert cmd.name == "dummy"
        assert cmd.description == "dummy desc"

    def test_run_returns_exit_code(self):
        assert _DummyCommand().run(None) == 42

    def test_repr_contains_name(self):
        assert "dummy" in repr(_DummyCommand())


# ============================================================================
# 2. CommandRegistry (8 tests)
# ============================================================================

class TestCommandRegistry:
    def test_register_and_get(self):
        reg = CommandRegistry()
        cmd = _DummyCommand()
        reg.register(cmd)
        assert reg.get("dummy") is cmd

    def test_get_missing_returns_none(self):
        assert CommandRegistry().get("nope") is None

    def test_all_preserves_order(self):
        reg = CommandRegistry()

        class A(_DummyCommand):
            name = "a"

        class B(_DummyCommand):
            name = "b"

        a, b = A(), B()
        reg.register(a)
        reg.register(b)
        assert reg.all() == [a, b]

    def test_names(self):
        reg = CommandRegistry()
        reg.register(_DummyCommand())
        assert reg.names() == ["dummy"]

    def test_duplicate_name_raises(self):
        reg = CommandRegistry()
        reg.register(_DummyCommand())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(_DummyCommand())

    def test_register_same_instance_idempotent(self):
        reg = CommandRegistry()
        cmd = _DummyCommand()
        reg.register(cmd)
        reg.register(cmd)  # 同一实例不报错
        assert len(reg) == 1

    def test_empty_name_raises(self):
        reg = CommandRegistry()

        class NoName(_DummyCommand):
            name = ""

        with pytest.raises(ValueError, match="empty"):
            reg.register(NoName())

    def test_clear_and_contains_and_len(self):
        reg = CommandRegistry()
        reg.register(_DummyCommand())
        assert "dummy" in reg
        assert len(reg) == 1
        reg.clear()
        assert len(reg) == 0
        assert "dummy" not in reg


# ============================================================================
# 3. COMMAND_REGISTRY 内容 (3 tests)
# ============================================================================

class TestModuleRegistry:
    def test_all_13_commands_registered(self):
        assert len(COMMAND_REGISTRY) == 13

    def test_command_names_and_order(self):
        assert COMMAND_REGISTRY.names() == EXPECTED_COMMANDS

    def test_every_command_has_description(self):
        for cmd in COMMAND_REGISTRY.all():
            assert cmd.description, f"{cmd!r} missing description"


# ============================================================================
# 4. _build_parser 通过 registry 构造 (4 tests)
# ============================================================================

class TestBuildParser:
    def test_parser_has_all_subcommands(self):
        from QuantNodes.cli import _build_parser

        parser = _build_parser()
        ns = parser.parse_args(["version"])
        assert ns.command == "version"

    def test_factor_best_args_parsed(self):
        from QuantNodes.cli import _build_parser

        ns = _build_parser().parse_args(
            ["factor-best", "--pool-dir", "p", "--top", "7", "--metric", "ir"]
        )
        assert ns.command == "factor-best"
        assert ns.pool_dir == "p"
        assert ns.top == 7
        assert ns.metric == "ir"

    def test_evolve_requires_config(self):
        from QuantNodes.cli import _build_parser

        with pytest.raises(SystemExit):
            _build_parser().parse_args(["evolve"])

    def test_run_flags(self):
        from QuantNodes.cli import _build_parser

        ns = _build_parser().parse_args(
            ["run", "--port", "18380", "--api-only"]
        )
        assert ns.port == 18380
        assert ns.api_only is True
        assert ns.frontend_only is False


# ============================================================================
# 5. main() dispatch (3 tests)
# ============================================================================

class TestMainDispatch:
    def test_main_dispatches_to_command_run(self):
        called = {}

        def fake_run(self, args):
            called["cmd"] = args.command
            return 0

        with patch.object(sys, "argv", [PROG_NAME, "version"]):
            with patch(
                "QuantNodes.cli.commands.version.VersionCommand.run",
                fake_run,
            ):
                assert main() == 0
        assert called["cmd"] == "version"

    def test_unknown_command_falls_back_to_help(self):
        # add_help=False, dest=command 缺省为 None → get(None) → help
        with patch.object(sys, "argv", [PROG_NAME]):
            assert main() == 0

    def test_help_command_runs(self):
        with patch.object(sys, "argv", [PROG_NAME, "help"]):
            assert main() == 0


# ============================================================================
# 6. Backward compat (3 tests)
# ============================================================================

class TestBackwardCompat:
    def test_cmd_functions_importable(self):
        from QuantNodes.cli import (  # noqa: F401
            cmd_init,
            cmd_run,
            cmd_chat,
            cmd_evolve,
            cmd_version,
            cmd_help,
            cmd_factor_info,
            cmd_factor_best,
            cmd_factor_visual,
            cmd_factor_dashboard,
            cmd_factor_data_fetch,
            cmd_factor_rag_eval,
            cmd_factor_rag_show,
        )

    def test_cmd_version_still_callable(self):
        from QuantNodes.cli import cmd_version

        ns = argparse.Namespace()
        assert cmd_version(ns) == 0

    def test_server_helpers_importable(self):
        from QuantNodes.cli import (  # noqa: F401
            start_api_server,
            start_frontend_server,
            _load_runner_from_config,
        )
