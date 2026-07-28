"""Tool 3: audit_get_code_context — extract AST-based code context."""
from __future__ import annotations

from quantnodes_strategy_audit.engines.context_engine import ContextEngine


def get_code_context_tool(
    context_engine: ContextEngine,
    file: str,
    focus_lines: list[int],
    depth: int = 2,
) -> dict:
    """Extract structured code context around focus lines.

    Args:
        context_engine: ContextEngine instance
        file: Path to Python file
        focus_lines: List of 1-indexed line numbers of interest
        depth: Context depth (lines of surrounding context)

    Returns:
        Code context dict with imports, enclosing_function, variables, etc.
    """
    return context_engine.get_code_context(
        file=file, focus_lines=focus_lines, depth=depth
    )
