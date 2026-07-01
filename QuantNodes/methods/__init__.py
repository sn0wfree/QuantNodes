# coding=utf-8
"""
QuantNodes Methods Package

Pure method implementations for external agents.
These methods can be called via API without LLM dependencies.

核心入口:
    - validate_code / execute_code: 代码沙箱执行 (api/routers/code.py)
    - validate_pipeline: pipeline 校验 (api/routers/code.py)

历史:
    早期版本曾导出 run_backtest / analyze_factor / query_wiki / FileOperations /
    CodeSearch / GitOperations 等，均无生产调用方，已在 v3.0.0 清理。
    对应 Agent 工具实现见 `agent/tools/`。
"""

from .sandbox import execute_code, validate_code, ExecutionResult, ValidationResult
from .pipeline import validate_pipeline, PipelineValidationResult

__all__ = [
    "validate_code",
    "execute_code",
    "ValidationResult",
    "ExecutionResult",
    "validate_pipeline",
    "PipelineValidationResult",
]