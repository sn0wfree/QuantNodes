# coding=utf-8
"""``quantnodes version`` and ``quantnodes help`` commands.

Phase 3.1 (2026-06-22): 改为 Command pattern — VersionCommand + HelpCommand.
旧的 cmd_version / cmd_help 函数保留作 backward compat.
"""

from QuantNodes import __version__
from QuantNodes.cli.command import Command


def cmd_version(args) -> int:
    """Show version information."""
    print(f"QuantNodes version {__version__}")
    return 0


def cmd_help(args) -> int:
    """Show help message."""
    print("""
QuantNodes CLI - 量化研究节点架构命令行工具

用法:
    quantnodes <命令> [选项]

命令:
    init        初始化当前目录
    run         启动服务
    chat        启动 Agent 对话模式
    evolve      多轮演化主入口 (Week 5)
    factor-info 显示 TrajectoryPool 统计 (Week 5)
    factor-best 显示 Top-N 最佳 entry (Week 5)
    factor-visual 生成可视化 HTML 报告 (Week 6)
    factor-rag-show RAG 检索演示 (Week 7)
    factor-rag-eval 批量评估 RAG 质量 (Week 10)
    factor-data-fetch 从 iFinD 拉取数据 + 写 H5 (Week 12)
    factor-dashboard 生成 3 类指标 dashboard (Week 13)
    version     显示版本
    help        显示帮助

evolve 选项:
    --config PATH          YAML 配置文件路径 (必填)
    --directions LIST      逗号分隔的研究方向
    --initial-json JSON    初始 candidates JSON
    --max-rounds N         覆盖 config.evolution.max_rounds
    --early-stop N         覆盖 config.evolution.early_stop_patience

factor-info / factor-best / factor-visual / factor-rag-show 选项:
    --pool-dir PATH        TrajectoryPool 目录
    --top N                Top-N (默认 5, 仅 factor-best/rag-show)
    --metric NAME          排序指标 (默认 sharpe, 仅 factor-best/visual)
    --output PATH          HTML 输出路径 (仅 factor-visual)
    --title TITLE          报告标题 (仅 factor-visual)
    --query TEXT           查询文本 (仅 factor-rag-show)
    --compress             启用谱系压缩 (Week 9, 仅 factor-rag-show)
    --ancestor-depth N     祖先深度 (默认 2, 仅 --compress)
    --descendant-depth N   后裔深度 (默认 2, 仅 --compress)
    --max-tokens N         压缩最大字符数 (默认 200, 仅 --compress)

init 选项:
    --force           强制重新初始化 (覆盖现有配置)

run 选项:
    --host HOST         绑定主机 (默认: localhost)
    --port PORT         前端端口 (默认: 5173)，设置后后端端口自动设为 PORT+1000
    --api-port PORT     后端端口 (默认: 19380)，优先级高于 --port 联动
    --daemon           后台运行 (仅 Linux)
    --api-only         仅启动后端
    --frontend-only    仅启动前端

示例:
    quantnodes init
    quantnodes run                          # 前端:5173, 后端:19380
    quantnodes run --port 18380             # 前端:18380, 后端:19380 (联动)
    quantnodes run --port 18380 --api-port 9000  # 前端:18380, 后端:9000 (指定后端)
    quantnodes run --daemon
    quantnodes run --api-only
    quantnodes evolve --config configs/evolve.yaml \\
        --directions momentum,reversal --max-rounds 3 --workers 4
    quantnodes factor-info --pool-dir output/trajectory
    quantnodes factor-best --pool-dir output/trajectory --top 10 --metric sharpe
    quantnodes factor-visual --pool-dir output/trajectory --output report.html
    quantnodes factor-rag-show --pool-dir output/trajectory \\
        --query "momentum effect" --top 5
    quantnodes factor-rag-eval --pool-dir output/trajectory \\
        --queries "momentum,reversal" --top 5
    quantnodes factor-data-fetch --output-dir /tmp/real_data/ \\
        --date-beg 20260101 --factors momentum_20d,reversal_5d
    quantnodes factor-dashboard --pool-dir output/trajectory --output dashboard.html
    quantnodes version

详细文档: docs/QuickStart.md
""")
    return 0


class VersionCommand(Command):
    """``quantnodes version`` subcommand."""

    name = "version"
    description = "显示版本"

    def add_arguments(self, subparsers) -> None:
        subparsers.add_parser(self.name, help=self.description)

    def run(self, args) -> int:
        return cmd_version(args)


class HelpCommand(Command):
    """``quantnodes help`` subcommand."""

    name = "help"
    description = "显示帮助"

    def add_arguments(self, subparsers) -> None:
        subparsers.add_parser(self.name, help=self.description)

    def run(self, args) -> int:
        return cmd_help(args)
