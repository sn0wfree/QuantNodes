"""quantnodes research — 策略研究工作区管理。

薄包装：调用 quantnodes-strategy-research 包。
"""
from __future__ import annotations

import argparse
import sys

from QuantNodes.cli.command import Command


class ResearchCommand(Command):
    """quantnodes research — 策略研究工作区管理。"""

    name = "research"
    description = "策略研究工作区管理"

    def add_arguments(self, subparsers) -> None:
        p = subparsers.add_parser(self.name, help=self.description)
        p.add_argument(
            "args",
            nargs="*",
            help="传递给 quantnodes-research 的参数",
        )

    def run(self, args: argparse.Namespace) -> int:
        try:
            from strategy_research.cli import main as research_main
            # 将参数传递给子命令
            sys.argv = ["quantnodes-research"] + args.args
            return research_main()
        except ImportError:
            print("❌ 未安装 quantnodes-strategy-research")
            print("   安装: pip install -e ~/Public/QuantNodes/research/strategy-research")
            return 1
