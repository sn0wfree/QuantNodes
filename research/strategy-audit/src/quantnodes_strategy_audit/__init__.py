"""quantnodes-strategy-audit.

量化策略审计工具 — 双引擎 + 教训驱动.

源自 QuantNodes 项目 17 天研发周期的教训总结（48 条 L-NNN 教训）。

Public API:
    - Lesson: 教训数据类
    - LessonLoader: 教训加载器
    - StaticEngine: 静态规则引擎 (Engine A)
    - ContextEngine: 上下文提供器 (Engine B)
    - CodeContextExtractor: AST 上下文提取
    - Warning / Severity: 检测结果
    - Report: 报告生成器
"""
from __future__ import annotations

__version__ = "0.2.0"

from quantnodes_strategy_audit.core.base import BaseDetector, BaseValidator
from quantnodes_strategy_audit.core.code_context import CodeContext, CodeContextExtractor
from quantnodes_strategy_audit.core.lesson import Lesson, LessonLoader
from quantnodes_strategy_audit.core.registry import DetectorRegistry
from quantnodes_strategy_audit.core.report import Report
from quantnodes_strategy_audit.core.warning import Severity, Warning
from quantnodes_strategy_audit.engines.context_engine import ContextEngine
from quantnodes_strategy_audit.engines.static_engine import StaticEngine

__all__ = [
    "BaseDetector",
    "BaseValidator",
    "CodeContext",
    "CodeContextExtractor",
    "ContextEngine",
    "DetectorRegistry",
    "Lesson",
    "LessonLoader",
    "Report",
    "Severity",
    "StaticEngine",
    "Warning",
]
