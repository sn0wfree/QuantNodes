# coding=utf-8
"""
state.py - Alpha-GPT workflow 状态管理

5 类记录的 dataclass + AlphaGptState 容器。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IdeaRecord:
    """第 1 阶段产物：alpha 想法"""

    id: str
    name: str
    category: str
    description: str
    expected_direction: str = "long"
    suggested_lookback: int = 20
    a_share_compatible: bool = True
    orthogonal_to: List[str] = field(default_factory=list)
    complexity_hint: str = "simple"
    round_idx: int = 1

    @classmethod
    def from_dict(cls, d: Dict[str, Any], round_idx: int) -> "IdeaRecord":
        return cls(
            id=d.get("id", ""),
            name=d.get("name", ""),
            category=d.get("category", "unknown"),
            description=d.get("description", ""),
            expected_direction=d.get("expected_direction", "long"),
            suggested_lookback=int(d.get("suggested_lookback", 20)),
            a_share_compatible=bool(d.get("a_share_compatible", True)),
            orthogonal_to=list(d.get("orthogonal_to", [])),
            complexity_hint=d.get("complexity_hint", "simple"),
            round_idx=round_idx,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "expected_direction": self.expected_direction,
            "suggested_lookback": self.suggested_lookback,
            "a_share_compatible": self.a_share_compatible,
            "orthogonal_to": self.orthogonal_to,
            "complexity_hint": self.complexity_hint,
        }


@dataclass
class FormulaRecord:
    """第 2 阶段产物：polars 公式"""

    formula_id: str
    idea_id: str
    formula: str
    round_discovered: int
    complexity: int = 0
    a_share_compatible: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formula_id": self.formula_id,
            "idea_id": self.idea_id,
            "formula": self.formula,
            "round_discovered": self.round_discovered,
            "complexity": self.complexity,
            "a_share_compatible": self.a_share_compatible,
        }


@dataclass
class EvaluationRecord:
    """第 3 阶段产物：公式评估"""

    formula_id: str
    formula: str
    status: str  # "success" | "failed"
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ir: float = 0.0
    ic_decay: Dict[int, float] = field(default_factory=dict)
    error_msg: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formula_id": self.formula_id,
            "formula": self.formula,
            "status": self.status,
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "ir": self.ir,
            "ic_decay": {str(k): v for k, v in self.ic_decay.items()},
            "error_msg": self.error_msg,
        }


@dataclass
class ReflectionRecord:
    """第 4 阶段产物：反思 verdicts + 改进建议"""

    round_idx: int
    verdicts: List[Dict[str, Any]] = field(default_factory=list)
    suggestions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round": self.round_idx,
            "formula_feedback": self.verdicts,
            "next_round_suggestions": self.suggestions,
        }


@dataclass
class FinalFormulaRecord:
    """第 5 阶段产物 / 最终输出：top-K 公式"""

    rank: int
    formula_id: str
    formula: str
    ic_mean: float = 0.0
    ir: float = 0.0
    category: Optional[str] = None
    round_discovered: int = 0
    selection_reason: str = ""
    risk_notes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], rank: int) -> "FinalFormulaRecord":
        return cls(
            rank=rank,
            formula_id=d.get("formula_id", ""),
            formula=d.get("formula", ""),
            ic_mean=float(d.get("metrics", {}).get("ic_mean", 0.0)),
            ir=float(d.get("metrics", {}).get("ir", 0.0)),
            category=d.get("category"),
            round_discovered=int(d.get("round_discovered", 0)),
            selection_reason=d.get("selection_reason", ""),
            risk_notes=list(d.get("risk_notes", [])),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "formula_id": self.formula_id,
            "formula": self.formula,
            "ic_mean": self.ic_mean,
            "ir": self.ir,
            "category": self.category,
            "round_discovered": self.round_discovered,
            "selection_reason": self.selection_reason,
            "risk_notes": self.risk_notes,
        }


@dataclass
class AlphaGptState:
    """Alpha-GPT 工作流状态容器"""

    objective: str
    iterations_total: int = 5
    round_idx_hint: int = 1

    all_ideas: List[IdeaRecord] = field(default_factory=list)
    all_formulas: List[FormulaRecord] = field(default_factory=list)
    all_evaluations: List[EvaluationRecord] = field(default_factory=list)
    all_reflections: List[ReflectionRecord] = field(default_factory=list)
    critic_output: Optional[Dict[str, Any]] = None


__all__ = [
    "IdeaRecord",
    "FormulaRecord",
    "EvaluationRecord",
    "ReflectionRecord",
    "FinalFormulaRecord",
    "AlphaGptState",
]
