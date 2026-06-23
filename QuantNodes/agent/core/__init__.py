# coding=utf-8
"""
核心引擎模块 (v3.0.0 精简版)

v3.0.0 之前本目录包含自写的 loop/runner/memory/dream/...，已全部由
HKUDS/nanobot 0.2.1 上游替代。

当前保留：
- quant_dream.py - 量化专属 Dream 钩子（QuantDreamHook）
- dream.py - 向后兼容 shim
"""

from .quant_dream import DreamEngine, QuantDreamHook, QuantDreamInsight

__all__ = [
    "DreamEngine",
    "QuantDreamHook",
    "QuantDreamInsight",
]
