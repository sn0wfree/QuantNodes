# coding=utf-8
"""Sandbox Pandas Bridge — auto-detect engine + inject context (PR-QN-4, 2026-06-22)

Provides engine detection for LLM-generated code and automatic context
injection into CodeSandbox.validate_and_execute().

Usage:
    from QuantNodes.ai.sandbox_pandas_bridge import detect_and_inject_context

    # Auto-detect engine from code, inject appropriate df + lib
    ctx = detect_and_inject_context(code, df=polars_df)
    # ctx now contains: df (polars or pandas), pl or pd, __version__, etc.
"""
from __future__ import annotations

import ast
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def detect_engine_from_code(code: str) -> str:
    """Scan code for import statements to detect engine.

    Heuristics (same as _engine.detect_engine):
      - import polars → polars
      - import pandas → pandas
      - both → polars (default, faster)
      - neither → polars (safe default)

    Returns:
        "polars" or "pandas"
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "polars"

    has_pl = False
    has_pd = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "polars" or alias.name.startswith("polars."):
                    has_pl = True
                elif alias.name == "pandas" or alias.name.startswith("pandas."):
                    has_pd = True
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "polars" or node.module.startswith("polars.")):
                has_pl = True
            elif node.module and (node.module == "pandas" or node.module.startswith("pandas.")):
                has_pd = True

    if has_pd and not has_pl:
        return "pandas"
    return "polars"


def detect_and_inject_context(
    code: str,
    df: Any = None,
    default_engine: str = "polars",
    **extra_context: Any,
) -> Dict[str, Any]:
    """Detect engine from code and build context dict for sandbox execution.

    Args:
        code: LLM-generated code to analyze
        df: Optional DataFrame to inject (auto-converted if needed)
        default_engine: Fallback engine when detection is ambiguous
        **extra_context: Additional context variables to inject

    Returns:
        Context dict ready for sandbox.validate_and_execute()
    """
    engine = detect_engine_from_code(code)
    ctx: Dict[str, Any] = dict(extra_context)

    if df is not None:
        if engine == "pandas":
            # Inject as pandas DataFrame
            if hasattr(df, "to_pandas"):
                ctx["df"] = df.to_pandas()
            else:
                ctx["df"] = df
            try:
                import pandas as pd
                ctx["pd"] = pd
            except ImportError:
                pass
        else:
            # Inject as polars DataFrame (default)
            ctx["df"] = df
            try:
                import polars as pl
                ctx["pl"] = pl
            except ImportError:
                pass

    # Always inject both libs for user convenience (LLM may mix)
    try:
        import polars as pl
        ctx.setdefault("pl", pl)
    except ImportError:
        pass
    try:
        import pandas as pd
        ctx.setdefault("pd", pd)
    except ImportError:
        pass

    # Inject QuantNodes operators for LLM access
    try:
        import QuantNodes
        ctx.setdefault("QuantNodes", QuantNodes)
    except ImportError:
        pass

    ctx["__engine__"] = engine
    return ctx


def patch_sandbox_with_bridge(sandbox: Any, df: Any = None) -> None:
    """Monkey-patch a CodeSandbox instance with auto-detect bridge.

    After patching, sandbox.validate_and_execute() will auto-detect
    the engine from code and inject appropriate context.

    Args:
        sandbox: CodeSandbox instance
        df: Optional DataFrame to inject
    """
    _original_validate = sandbox.validate_and_execute

    def _patched_validate(code: str, context: Optional[Dict[str, Any]] = None) -> Any:
        ctx = detect_and_inject_context(code, df=df)
        if context:
            ctx.update(context)
        return _original_validate(code, context=ctx)

    sandbox.validate_and_execute = _patched_validate  # type: ignore[attr-defined]
    logger.debug("Sandbox patched with pandas bridge (auto-detect)")

__all__ = [
    "detect_engine_from_code",
    "detect_and_inject_context",
    "patch_sandbox_with_bridge",
]
