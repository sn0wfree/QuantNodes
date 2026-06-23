# coding=utf-8
"""
QuantNodes Agent 系统

基于 HKUDS/nanobot 0.2.1 上游 (PyPI: nanobot-ai>=0.2.1,<0.3.0) 的量化研究智能体。

v3.0.0 架构变更（Path A: 直接消费上游）：
- 核心运行时由本地复刻 → 改为包装 ``Nanobot.from_config``（见 ``nanobot_bridge.py``）
- 量化专属 Dream 钩子见 ``QuantNodes.agent.core.quant_dream``
- 15 个量化工具父类改为 ``nanobot.agent.tools.base.Tool``（见 ``tools/base.py``）
- workspace 由 ``.quant_agent/`` → ``.agent/``（上游默认约定）

Usage (向后兼容 v2.x 签名):
    from QuantNodes.agent import Agent

    agent = Agent(workspace=".agent", config={"model": "gpt-4o"})
    response = await agent.run("帮我生成一个动量策略")
"""

from __future__ import annotations

import warnings

from QuantNodes.core.path_utils import ensure_dir

from .nanobot_bridge import Agent
from .core.quant_dream import QuantDreamHook, QuantDreamInsight, DreamEngine
from .tools import register_all_quant_tools

__version__ = "3.0.0"

__all__ = [
    "__version__",
    "Agent",
    "QuantDreamHook",
    "QuantDreamInsight",
    "DreamEngine",
    "register_all_quant_tools",
]
