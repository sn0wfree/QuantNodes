"""WikiStrategy dataclass (PR6.6 / M4.3 split).

WikiStrategy: 策略持久化模型 (9 字段).

向后兼容: `from QuantNodes.research.wiki import WikiStrategy` 仍可用.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class WikiStrategy:
    """Wiki 策略定义."""

    name: str
    strategy_yaml: str
    description: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    backtest_result: Optional[Dict] = None
    created_at: Optional[str] = None
    wiki_page_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


__all__ = ["WikiStrategy"]