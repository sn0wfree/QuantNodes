# coding=utf-8
"""
QuantNodes Agent 系统

基于 HKUDS/nanobot 0.2.1 上游 (PyPI: ``nanobot-ai>=0.2.1,<0.3.0``) 的量化研究智能体。

v3.0.0 架构变更（Path A: 直接消费上游）：
- 核心运行时由本地复刻 → 改为包装 ``Nanobot.from_config``（见 ``nanobot_bridge.py``）
- 量化专属 Dream 钩子见 ``QuantNodes.agent.core.quant_dream``
- 15 个量化工具父类改为 ``nanobot.agent.tools.base.Tool``（见 ``tools/base.py``）
- workspace 由 ``.quant_agent/`` → ``.agent/``（上游默认约定）

**v3.0.0 重要变更（Stage 5.3）**：``nanobot-ai`` 改为 **可选依赖**
（``pip install 'quantnodes[agent]'``）。未装时：
- ``NANOBOT_AVAILABLE = False``
- ``Agent`` 类的属性访问抛 ``NanobotNotInstalled``
- 量化工具库（Wiki / Factor / Backtest / Strategy）完全可用
- MCP server、API、CLI 等不依赖 nanobot 的部分正常启动

Usage (向后兼容 v2.x 签名):
    from QuantNodes.agent import Agent, NANOBOT_AVAILABLE

    if NANOBOT_AVAILABLE:
        agent = Agent(workspace=".agent", config={"model": "gpt-4o"})
        response = await agent.run("帮我生成一个动量策略")
    else:
        # 量化工具库仍可用
        from QuantNodes.research.wiki.proxy import WikiFactorProxy
        wiki = WikiFactorProxy()
        factor = await wiki.get("momentum_20")
"""

from __future__ import annotations

import warnings
from typing import Any

from QuantNodes.core.path_utils import ensure_dir

__version__ = "3.0.0"

# ----------------------------------------------------------------------------
# Optional-dependency guard
# ----------------------------------------------------------------------------
#
# v3.0.0 之前，``nanobot-ai`` 是强制依赖。从 Stage 5.3 起改为 ``[agent]`` extras：
# - ``pip install quantnodes``            → 纯量化工具库
# - ``pip install 'quantnodes[agent]'``   → + nanobot agent / WebUI / MCP
# - ``pip install 'quantnodes[all]'``     → 装齐所有 extras
#
# 顶层 import 任何 nanobot 符号必须先检查 ``NANOBOT_AVAILABLE``，否则
# ``from QuantNodes.agent import Agent`` 在未装 extras 时会 ImportError。
NANOBOT_AVAILABLE: bool = True
NANOBOT_IMPORT_ERROR: str | None = None
_NANOBOT_PROBE: Any = None

# Probe: try to import a real nanobot submodule, not just the top-level
# namespace. The top-level `nanobot` may exist as a namespace package
# (e.g. with just a `bridge/` subpackage) but the `nanobot.agent` submodule
# could be missing — in which case the agent runtime is not usable.
# We probe a known-deep symbol: ``nanobot.agent.tools.base.Tool``.
try:
    from nanobot.agent.tools.base import Tool as _ProbeTool  # noqa: F401
    NANOBOT_AVAILABLE = True
    del _ProbeTool
except ImportError as _e:  # pragma: no cover - exercised by the import
    NANOBOT_AVAILABLE = False
    NANOBOT_IMPORT_ERROR = str(_e)


class NanobotNotInstalled(ImportError):
    """Raised when user code touches a nanobot-only symbol without [agent] extra.

    The error message is tailored to be friendly: it tells the user exactly
    which pip extra to install.
    """

    def __init__(self, symbol: str = "Agent"):
        msg = (
            f"{symbol} requires the optional 'agent' extra. "
            "Install it with:  pip install 'quantnodes[agent]'  (or  'quantnodes[all]')"
        )
        if NANOBOT_IMPORT_ERROR:
            msg += f"\nUnderlying error: {NANOBOT_IMPORT_ERROR}"
        super().__init__(msg)


# ----------------------------------------------------------------------------
# Re-exports (lazy where nanobot is required)
# ----------------------------------------------------------------------------
if NANOBOT_AVAILABLE:
    from .nanobot_bridge import Agent  # noqa: E402
    from .core.quant_dream import (  # noqa: E402
        QuantDreamHook,
        QuantDreamInsight,
        DreamEngine,
    )
    from .tools import register_all_quant_tools  # noqa: E402
else:
    # Provide a stub for ``Agent`` so ``from QuantNodes.agent import Agent`` does
    # not raise at import time. The stub raises ``NanobotNotInstalled`` only when
    # the user actually tries to *instantiate* or *attribute-access* it.
    class _NanobotUnavailableProxy:
        """Proxy that raises ``NanobotNotInstalled`` on any access.

        Implemented via ``__getattr__`` (PEP 562) so that the import itself
        succeeds — only attribute access triggers the error. This keeps
        ``from QuantNodes.agent import Agent`` working everywhere; the user's
        first real call (``Agent(...)`` or ``Agent.some_attr``) gets the
        friendly error message.
        """

        _PROXY_SYMBOLS = {
            "Agent",
            "QuantDreamHook",
            "QuantDreamInsight",
            "DreamEngine",
            "register_all_quant_tools",
        }

        def __getattr__(self, name: str) -> Any:
            if name in self._PROXY_SYMBOLS:
                raise NanobotNotInstalled(name)
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r} (nanobot-ai not installed)"
            )

        def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise NanobotNotInstalled("Agent")

    _proxy = _NanobotUnavailableProxy()
    Agent = _proxy  # type: ignore[assignment]
    QuantDreamHook = _proxy  # type: ignore[assignment]
    QuantDreamInsight = _proxy  # type: ignore[assignment]
    DreamEngine = _proxy  # type: ignore[assignment]
    register_all_quant_tools = _proxy  # type: ignore[assignment]

    # Emit a one-time DeprecationWarning-ish info on import
    warnings.warn(
        "QuantNodes.agent loaded without nanobot-ai (NANOBOT_AVAILABLE=False). "
        "Agent / WebUI / MCP features are disabled. "
        "Install with:  pip install 'quantnodes[agent]'",
        ImportWarning,
        stacklevel=2,
    )


__all__ = [
    "__version__",
    "NANOBOT_AVAILABLE",
    "NANOBOT_IMPORT_ERROR",
    "NanobotNotInstalled",
    "Agent",
    "QuantDreamHook",
    "QuantDreamInsight",
    "DreamEngine",
    "register_all_quant_tools",
]
