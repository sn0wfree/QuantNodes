"""Tool 5: audit_static_precheck — Engine A precheck."""
from __future__ import annotations

from quantnodes_strategy_audit.engines.context_engine import ContextEngine


def static_precheck_tool(
    context_engine: ContextEngine,
    file: str,
    lesson_ids: list[str] | None = None,
) -> dict:
    """Run Engine A static precheck for specific lessons.

    Args:
        context_engine: ContextEngine instance
        file: Path to Python file
        lesson_ids: Optional list of L-NNN to filter (None = all)

    Returns:
        Dict with total_warnings, by_lesson grouping, precheck_violated list.
    """
    return context_engine.static_precheck(file=file, lesson_ids=lesson_ids)
