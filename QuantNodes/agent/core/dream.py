# coding=utf-8
"""
向后兼容 shim — 量化专属 Dream 引擎。

v3.0.0 起，Dream 相关实现已迁移到 ``QuantNodes.agent.core.quant_dream``。
本模块保留旧导入路径 ``from QuantNodes.agent.core.dream import ...`` 的可用性。

替代映射：
- ``DreamEngine``        -> ``QuantNodes.agent.core.quant_dream.DreamEngine``
- ``DreamStore``         -> 已删除（由 nanobot 上游 ``MemoryStore`` 替代，见 .agent/memory/）
- ``MemoryStore``        -> 已删除（由 nanobot 上游 ``nanobot.agent.memory.MemoryStore`` 替代）
- ``MemoryManager``      -> 已删除（迁移到 nanobot 的 MEMORY.md/SOUL.md 文件系统约定）
"""

from __future__ import annotations

import warnings

from .quant_dream import DreamEngine, QuantDreamHook, QuantDreamInsight

warnings.warn(
    "QuantNodes.agent.core.dream is a backward-compatibility shim. "
    "Import from QuantNodes.agent.core.quant_dream instead. "
    "DreamStore/MemoryStore/MemoryManager have been replaced by the "
    "upstream nanobot memory subsystem.",
    DeprecationWarning,
    stacklevel=2,
)


class _RemovedSymbol:
    """Sentinel for symbols removed in v3.0.0 (nanobot upstream replacement)."""


DreamStore = _RemovedSymbol
MemoryStore = _RemovedSymbol
MemoryManager = _RemovedSymbol


__all__ = [
    "DreamEngine",
    "DreamStore",
    "MemoryStore",
    "MemoryManager",
    "QuantDreamHook",
    "QuantDreamInsight",
]
