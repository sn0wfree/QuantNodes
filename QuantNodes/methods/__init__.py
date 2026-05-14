# coding=utf-8
"""
QuantNodes Methods Package

Pure method implementations for external agents.
These methods can be called via API without LLM dependencies.
"""

from .backtest import run_backtest, BacktestResult
from .sandbox import validate_code, execute_code, ValidationResult, ExecutionResult
from .pipeline import validate_pipeline, PipelineValidationResult
from .factor import analyze_factor, FactorAnalysisResult
from .wiki import query_wiki, WikiResult
from .file_ops import FileOperations
from .code_search import CodeSearch
from .git_ops import GitOperations

__all__ = [
    "run_backtest",
    "BacktestResult",
    "validate_code",
    "execute_code",
    "ValidationResult",
    "ExecutionResult",
    "validate_pipeline",
    "PipelineValidationResult",
    "analyze_factor",
    "FactorAnalysisResult",
    "query_wiki",
    "WikiResult",
    "FileOperations",
    "CodeSearch",
    "GitOperations",
]