"""Tool 6: audit_submit_finding — Agent submits finding."""
from __future__ import annotations

from quantnodes_strategy_audit.engines.context_engine import ContextEngine


def submit_finding_tool(context_engine: ContextEngine, finding_data: dict) -> dict:
    """Submit a finding (from Agent's semantic analysis).

    Args:
        context_engine: ContextEngine instance
        finding_data: Dict with keys: file, line, lesson_id, status, severity,
                      evidence, fix_suggestion, confidence

    Returns:
        Dict with finding_id and stored=True.
    """
    return context_engine.submit_finding(finding_data)
