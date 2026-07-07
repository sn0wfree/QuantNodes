"""WikiReproduction dataclass (PR6.6 / M4.3 split).

WikiReproduction: 研报复现结果持久化模型 (8 字段).

向后兼容: `from QuantNodes.research.wiki import WikiReproduction` 仍可用.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class WikiReproduction:
    """Wiki 研报复现结果."""

    report_title: str
    pdf_path: str = ""
    verified_count: int = 0
    failed_count: int = 0
    report_markdown: str = ""
    created_at: Optional[str] = None
    wiki_page_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = ["WikiReproduction"]