"""WikiLogic dataclass (PR6.6 / M4.3 split).

WikiLogic: 逻辑（factor extraction logic）持久化模型.
- 11 基础字段 (name, content, source, etc.)
- + 4 结构化字段 (PR-1/4 向后兼容, 全部 Optional)

向后兼容: `from QuantNodes.research.wiki import WikiLogic` 仍可用
(由 `wiki/__init__.py` re-export).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .enums import LogicSource


@dataclass
class WikiLogic:
    """Wiki 逻辑定义 (research report 中提取的逻辑)."""

    name: str
    content: str
    source: LogicSource
    extracted_formula: Optional[str] = None
    source_detail: Dict[str, str] = field(default_factory=dict)
    related_strategies: List[str] = field(default_factory=list)
    related_factors: List[str] = field(default_factory=list)
    validation_status: str = "pending"
    wiki_page_name: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # === 新增:结构化字段(PR-1/4 向后兼容,全部 Optional) ===
    structured: Optional[Any] = None  # WikiLogicStructured (避免循环导入)
    performance_evidence: Optional[Any] = None  # LogicPerformanceEvidence
    parent_logic: Optional[str] = None  # 衍生自的逻辑名(用于追溯重构链)
    refinement_round: int = 0  # 第几轮外层优化生成/重构

    def to_structured_dict(self) -> Dict[str, Any]:
        """序列化为字典(便于 JSON 持久化)"""
        return {
            "name": self.name,
            "content": self.content,
            "source": self.source.value if hasattr(self.source, "value") else str(self.source),
            "extracted_formula": self.extracted_formula,
            "validation_status": self.validation_status,
            "parent_logic": self.parent_logic,
            "refinement_round": self.refinement_round,
            "structured": self.structured.to_dict() if self.structured else None,
            "performance_evidence": (
                self.performance_evidence.to_dict()
                if self.performance_evidence and hasattr(self.performance_evidence, "to_dict")
                else self.performance_evidence
            ),
        }

    @classmethod
    def from_structured_dict(cls, data: Dict[str, Any]) -> "WikiLogic":
        """从字典创建(用于反序列化)"""
        # Lazy import to avoid circular dep with quant_alpha.logic_mining.models
        from QuantNodes.research.quant_alpha.logic_mining.models import (
            LogicBehavior,
            LogicCondition,
            LogicPerformanceEvidence,
            WikiLogicStructured,
        )

        structured = None
        if data.get("structured"):
            try:
                s = data["structured"]
                structured = WikiLogicStructured.from_dict(s)
            except Exception:
                structured = None

        evidence = None
        if data.get("performance_evidence"):
            try:
                evidence = LogicPerformanceEvidence.from_dict(data["performance_evidence"])
            except Exception:
                evidence = None

        try:
            source = LogicSource(data.get("source", "research_report"))
        except ValueError:
            source = LogicSource.RESEARCH_REPORT

        return cls(
            name=data["name"],
            content=data.get("content", ""),
            source=source,
            extracted_formula=data.get("extracted_formula"),
            validation_status=data.get("validation_status", "pending"),
            structured=structured,
            performance_evidence=evidence,
            parent_logic=data.get("parent_logic"),
            refinement_round=data.get("refinement_round", 0),
        )


__all__ = ["WikiLogic"]