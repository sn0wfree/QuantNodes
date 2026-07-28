"""Tool 4: audit_search_lessons — keyword search."""
from __future__ import annotations

from quantnodes_strategy_audit.engines.context_engine import ContextEngine


def search_lessons_tool(
    context_engine: ContextEngine, query: str, top_k: int = 5
) -> dict:
    """Search lessons by keyword relevance.

    Args:
        context_engine: ContextEngine instance
        query: Search query string
        top_k: Max results to return

    Returns:
        Dict with query and ranked results.
    """
    return context_engine.search_lessons(query=query, top_k=top_k)
