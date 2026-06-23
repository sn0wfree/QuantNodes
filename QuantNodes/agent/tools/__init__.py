# coding=utf-8
"""
工具系统模块

Tool基类（继承自 HKUDS nanobot upstream Tool）/ 注册表 / 具体工具实现
"""

from .base import Tool, ToolExecutionResult
from .context import ToolContext, ToolContextFactory
from .registry import ToolRegistry
from .echo import EchoTool
from .sandbox import SandboxTool
from .pipeline import PipelineTool
from .strategy import StrategyTool
from .backtest import BacktestTool
from .factor import FactorTool
from .config_backtest import ConfigBacktestTool
from .wiki import WikiTool
from .file_ops import FileOpsTool
from .code_search import CodeSearchTool
from .git_ops import GitOpsTool
from .web_fetch import WebFetchTool
from .web_search import WebSearchTool
from .task import TaskTool


_QUANT_TOOL_FACTORIES = [
    EchoTool,
    SandboxTool,
    PipelineTool,
    StrategyTool,
    BacktestTool,
    FactorTool,
    ConfigBacktestTool,
    WikiTool,
    FileOpsTool,
    CodeSearchTool,
    GitOpsTool,
    WebFetchTool,
    WebSearchTool,
    TaskTool,
]


def register_all_quant_tools(registry: ToolRegistry) -> int:
    """Register all 14 quant tool classes (echo/sandbox/.../task) into a registry.

    Returns the number of quant tools registered. Idempotent: re-registration
    of an existing tool name is skipped (the upstream registry overwrites by
    name, but we prefer explicit skip to surface duplicate-name bugs early).

    Usage from Agent.__init__::

        from nanobot.agent.tools.registry import ToolRegistry
        from QuantNodes.agent.tools import register_all_quant_tools
        register_all_quant_tools(self._loop.tool_registry)
    """
    registered = 0
    for factory in _QUANT_TOOL_FACTORIES:
        try:
            tool = factory()
        except TypeError:
            tool = factory
        if registry.has(tool.name):
            continue
        registry.register(tool)
        registered += 1
    return registered


__all__ = [
    "Tool",
    "ToolExecutionResult",
    "ToolContext",
    "ToolContextFactory",
    "ToolRegistry",
    "register_all_quant_tools",
    "EchoTool",
    "SandboxTool",
    "PipelineTool",
    "StrategyTool",
    "BacktestTool",
    "FactorTool",
    "ConfigBacktestTool",
    "WikiTool",
    "FileOpsTool",
    "CodeSearchTool",
    "GitOpsTool",
    "WebFetchTool",
    "WebSearchTool",
    "TaskTool",
]
