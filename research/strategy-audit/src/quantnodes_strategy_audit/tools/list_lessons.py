"""Tool 2: audit_list_lessons — list lessons with filters."""
from __future__ import annotations

from quantnodes_strategy_audit.engines.context_engine import ContextEngine


def list_lessons_tool(
    context_engine: ContextEngine,
    category: str | None = None,
    severity: str | None = None,
    auto_checkable: str | None = None,
) -> dict:
    """List lessons with optional filters.

    Args:
        context_engine: ContextEngine instance
        category: Filter by category (e.g., "lookahead", "nan_safe")
        severity: Filter by severity ("CRITICAL", "HIGH", "MEDIUM", "LOW")
        auto_checkable: Filter by routing ("static", "agent", "partial", "manual")

    Returns:
        Dict with count and list of lesson dicts.
    """
    return context_engine.list_lessons(
        category=category, severity=severity, auto_checkable=auto_checkable
    )
