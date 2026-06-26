# coding=utf-8
"""
工具系统模块

Tool基类（继承自 HKUDS nanobot upstream Tool）/ 注册表 / 具体工具实现
"""

import logging
from pathlib import Path
from typing import Any

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
from .alpha_evaluate import AlphaEvaluateTool
from .alpha_backtest import AlphaBacktestTool
from .operator_lookup import OperatorLookupTool

logger = logging.getLogger(__name__)


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
    AlphaEvaluateTool,
    AlphaBacktestTool,
    OperatorLookupTool,
]


def register_all_quant_tools(
    registry: ToolRegistry,
    workspace: Path | None = None,
    llm_client: Any = None,
    model: str | None = None,
) -> int:
    """Register all 14 quant tool classes (echo/sandbox/.../task) into a registry.

    Returns the number of quant tools registered. Idempotent: re-registration
    of an existing tool name is skipped (the upstream registry overwrites by
    name, but we prefer explicit skip to surface duplicate-name bugs early).

    ``workspace`` is forwarded to tools that need it (``WikiTool``,
    ``FileOpsTool``, ``CodeSearchTool``, ``GitOpsTool``, ``TaskTool``).
    Other tools take no constructor args.

    ``llm_client`` and ``model`` are forwarded to ``WorkflowTool`` (if
    nanobot is available). When ``llm_client`` is provided, the
    ``run_workflow`` tool is registered, enabling LLM-driven pipeline
    execution via ``WorkflowRegistry``.

    Usage from Agent.__init__::

        from nanobot.agent.tools.registry import ToolRegistry
        from QuantNodes.agent.tools import register_all_quant_tools
        register_all_quant_tools(self._loop.tools, workspace=self.workspace)
    """
    if workspace is None:
        workspace = Path(".agent")
    workspace = Path(workspace)

    workspace_dep = {
        "WikiTool": {"wiki_path": str(workspace / "wiki")},
        "FileOpsTool": {"workspace": workspace},
        "CodeSearchTool": {"workspace": workspace},
        "GitOpsTool": {"workspace": workspace},
        "TaskTool": {"workspace": workspace},
    }

    registered = 0
    for factory in _QUANT_TOOL_FACTORIES:
        kwargs = workspace_dep.get(factory.__name__, {})
        try:
            tool = factory(**kwargs) if kwargs else factory()
        except (TypeError, Exception) as exc:
            logger.warning("Failed to register quant tool %s: %s", factory.__name__, exc)
            continue
        if registry.has(tool.name):
            continue
        registry.register(tool)
        registered += 1

    # Register WorkflowTool (run_workflow) if llm_client is provided
    if llm_client is not None:
        try:
            from ..workflows.tool import WorkflowTool

            wt = WorkflowTool(
                llm_client=llm_client,
                model=model,
                results_dir=workspace / "results",
            )
            if not registry.has(wt.name):
                registry.register(wt)
                registered += 1
        except Exception as exc:
            logger.warning("Failed to register WorkflowTool: %s", exc)

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
    "AlphaEvaluateTool",
    "AlphaBacktestTool",
    "OperatorLookupTool",
]
