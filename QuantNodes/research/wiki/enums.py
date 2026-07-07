"""Wiki enums + relation types (PR6.6 / M4.3 split).

M4.3 wiki.py 拆分:
  原 `QuantNodes.research.wiki` (1218 行单文件) 拆成子包.
  本文件包含所有 enum 和常量:
    - FactorSource (5 成员: RESEARCH_REPORT, AUTO_RESEARCH, MANUAL, DERIVED, IMPORTED)
    - FactorCategory (7 成员: MOMENTUM, VALUE, QUALITY, VOLATILITY, SIZE, GROWTH, OTHER)
    - LogicSource (2 成员: RESEARCH_REPORT, MANUAL)
    - QUANT_RELATION_TYPES (9 关系类型)

向后兼容: `from QuantNodes.research.wiki import FactorSource` 仍可用
(由 `wiki/__init__.py` re-export).
"""
from __future__ import annotations

from enum import Enum


class FactorSource(Enum):
    """因子来源枚举."""

    RESEARCH_REPORT = "research_report"
    AUTO_RESEARCH = "auto_research"
    MANUAL = "manual"
    DERIVED = "derived"
    IMPORTED = "imported"


class FactorCategory(Enum):
    """因子分类枚举."""

    MOMENTUM = "momentum"
    VALUE = "value"
    QUALITY = "quality"
    VOLATILITY = "volatility"
    SIZE = "size"
    GROWTH = "growth"
    OTHER = "other"


class LogicSource(Enum):
    """逻辑来源枚举."""

    RESEARCH_REPORT = "research_report"
    MANUAL = "manual"


QUANT_RELATION_TYPES = {
    "uses",
    "correlates_with",
    "derived_from",
    "outperforms",
    "underperforms",
    "similar_to",
    "contradicts",
    "supports",
    "related_to",
}


__all__ = [
    "FactorSource",
    "FactorCategory",
    "LogicSource",
    "QUANT_RELATION_TYPES",
]