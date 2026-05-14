# coding=utf-8
"""
核心引擎模块

消息循环 / 执行引擎 / 上下文构建 / 记忆系统
"""

from .context import ContextBuilder
from .hook import AgentHook, CompositeHook
from .runner import AgentRunner, AgentRunSpec, AgentRunResult
from .loop import AgentLoop
from .memory import MemoryStore
from .compaction import ContextCompactor, CompactionConfig, CompactionResult, compact_messages

__all__ = [
    "ContextBuilder",
    "AgentHook",
    "CompositeHook",
    "AgentRunner",
    "AgentRunSpec",
    "AgentRunResult",
    "AgentLoop",
    "MemoryStore",
    "ContextCompactor",
    "CompactionConfig",
    "CompactionResult",
    "compact_messages",
]
