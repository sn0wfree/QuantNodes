"""Tool 1: audit_get_lesson — load lesson markdown."""
from __future__ import annotations

from quantnodes_strategy_audit.engines.context_engine import ContextEngine


def get_lesson_tool(context_engine: ContextEngine, lesson_id: str) -> dict:
    """Load complete lesson document.

    Args:
        context_engine: ContextEngine instance
        lesson_id: L-NNN identifier (e.g., "L-202")

    Returns:
        Lesson dict with id, title, severity, check_prompt, content_markdown, etc.
        Does NOT include file_path (Q3 = A: don't expose paths).
    """
    return context_engine.get_lesson(lesson_id)
