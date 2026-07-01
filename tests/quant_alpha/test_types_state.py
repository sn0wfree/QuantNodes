# coding=utf-8
"""Tests for research/quant_alpha/types/state.py and constants.py.

Covers: IdeaRecord, FormulaRecord, EvaluationRecord, ReflectionRecord,
FinalFormulaRecord, AlphaGptState — from_dict, to_dict, thinking chain fields.
"""

import pytest

from QuantNodes.research.quant_alpha.types.state import (
    IdeaRecord,
    FormulaRecord,
    EvaluationRecord,
    ReflectionRecord,
    FinalFormulaRecord,
    AlphaGptState,
)
from QuantNodes.research.quant_alpha.types.constants import ALLOWED_OPERATORS


# ============================================================================
# Constants
# ============================================================================

class TestAllowedOperators:
    def test_constants_not_empty(self):
        assert len(ALLOWED_OPERATORS) > 0

    def test_common_operators_present(self):
        assert "rank" in ALLOWED_OPERATORS
        assert "zscore" in ALLOWED_OPERATORS
        assert "ts_mean" in ALLOWED_OPERATORS
        assert "add" in ALLOWED_OPERATORS

    def test_no_duplicates(self):
        assert len(ALLOWED_OPERATORS) == len(set(ALLOWED_OPERATORS))


# ============================================================================
# IdeaRecord
# ============================================================================

class TestIdeaRecord:
    def test_creation(self):
        idea = IdeaRecord(
            id="idea-1",
            name="momentum_20",
            category="momentum",
            description="20-day momentum factor",
        )
        assert idea.id == "idea-1"
        assert idea.name == "momentum_20"
        assert idea.category == "momentum"
        assert idea.description == "20-day momentum factor"

    def test_defaults(self):
        idea = IdeaRecord(id="1", name="x", category="y", description="z")
        assert idea.expected_direction == "long"
        assert idea.suggested_lookback == 20
        assert idea.a_share_compatible is True
        assert idea.orthogonal_to == []
        assert idea.complexity_hint == "simple"
        assert idea.round_idx == 1

    def test_thinking_chain_fields(self):
        idea = IdeaRecord(
            id="1",
            name="x",
            category="y",
            description="z",
            thinking="thinking content",
            hypothesis="my hypothesis",
            mechanism="my mechanism",
            mentioned_ops=["rank", "ts_mean"],
        )
        assert idea.thinking == "thinking content"
        assert idea.hypothesis == "my hypothesis"
        assert idea.mechanism == "my mechanism"
        assert idea.mentioned_ops == ["rank", "ts_mean"]

    def test_from_dict_basic(self):
        d = {"id": "1", "name": "x", "category": "y", "description": "z"}
        idea = IdeaRecord.from_dict(d, round_idx=3)
        assert idea.round_idx == 3

    def test_from_dict_with_rationale(self):
        """P3: 'rationale' (new) takes precedence over 'description' (old)."""
        d = {
            "id": "1",
            "name": "x",
            "category": "y",
            "rationale": "new rationale",
            "description": "old description",
        }
        idea = IdeaRecord.from_dict(d, round_idx=1)
        assert idea.description == "new rationale"

    def test_from_dict_fallback_to_description(self):
        d = {
            "id": "1",
            "name": "x",
            "category": "y",
            "description": "old description",
        }
        idea = IdeaRecord.from_dict(d, round_idx=1)
        assert idea.description == "old description"

    def test_from_dict_defaults(self):
        idea = IdeaRecord.from_dict({}, round_idx=1)
        assert idea.id == ""
        assert idea.name == ""

    def test_to_dict_includes_thinking(self):
        idea = IdeaRecord(
            id="1", name="x", category="y", description="z",
            hypothesis="h", mentioned_ops=["rank"],
        )
        d = idea.to_dict()
        assert "hypothesis" in d
        assert "mentioned_ops" in d

    def test_to_dict_roundtrip(self):
        original = IdeaRecord(
            id="1", name="x", category="y", description="z",
            expected_direction="short", complexity_hint="medium",
        )
        d = original.to_dict()
        restored = IdeaRecord.from_dict(d, round_idx=1)
        assert restored.id == original.id
        assert restored.expected_direction == "short"


# ============================================================================
# FormulaRecord
# ============================================================================

class TestFormulaRecord:
    def test_creation(self):
        f = FormulaRecord(
            formula_id="f-1",
            idea_id="idea-1",
            formula="rank(close)",
            round_discovered=1,
        )
        assert f.formula_id == "f-1"
        assert f.formula == "rank(close)"
        assert f.round_discovered == 1

    def test_defaults(self):
        f = FormulaRecord(formula_id="f-1", idea_id="i-1", formula="x", round_discovered=0)
        assert f.complexity == 0
        assert f.a_share_compatible is True

    def test_thinking_chain_fields(self):
        f = FormulaRecord(
            formula_id="f-1",
            idea_id="i-1",
            formula="x",
            round_discovered=1,
            thinking="t",
            hypothesis="h",
            mentioned_ops=["rank"],
        )
        assert f.thinking == "t"
        assert f.mentioned_ops == ["rank"]

    def test_to_dict(self):
        f = FormulaRecord(
            formula_id="f-1", idea_id="i-1", formula="x",
            round_discovered=2, complexity=5, a_share_compatible=False,
        )
        d = f.to_dict()
        assert d["formula_id"] == "f-1"
        assert d["complexity"] == 5
        assert d["a_share_compatible"] is False


# ============================================================================
# EvaluationRecord
# ============================================================================

class TestEvaluationRecord:
    def test_creation_success(self):
        e = EvaluationRecord(
            formula_id="f-1",
            formula="x",
            status="success",
            ic_mean=0.05,
            ir=0.5,
        )
        assert e.status == "success"
        assert e.ic_mean == 0.05

    def test_creation_failed(self):
        e = EvaluationRecord(
            formula_id="f-1", formula="x", status="failed",
            error_msg="syntax error",
        )
        assert e.status == "failed"
        assert e.error_msg == "syntax error"

    def test_defaults(self):
        e = EvaluationRecord(formula_id="f-1", formula="x", status="success")
        assert e.ic_mean == 0.0
        assert e.ic_std == 0.0
        assert e.ir == 0.0
        assert e.ic_decay == {}

    def test_to_dict(self):
        e = EvaluationRecord(
            formula_id="f-1", formula="x", status="success",
            ic_mean=0.05, ir=0.5, ic_decay={1: 0.05, 5: 0.04},
        )
        d = e.to_dict()
        assert d["ic_mean"] == 0.05
        assert "1" in d["ic_decay"]
        assert "5" in d["ic_decay"]


# ============================================================================
# ReflectionRecord
# ============================================================================

class TestReflectionRecord:
    def test_creation(self):
        r = ReflectionRecord(round_idx=1)
        assert r.round_idx == 1
        assert r.verdicts == []
        assert r.suggestions == {}

    def test_creation_with_verdicts(self):
        r = ReflectionRecord(
            round_idx=2,
            verdicts=[{"formula": "x", "verdict": "keep"}],
            suggestions={"next_round": "try more momentum"},
        )
        assert len(r.verdicts) == 1
        assert r.suggestions["next_round"] == "try more momentum"

    def test_thinking_fields(self):
        r = ReflectionRecord(
            round_idx=1,
            thinking="my thinking",
            key_insights=["insight1", "insight2"],
        )
        assert r.thinking == "my thinking"
        assert r.key_insights == ["insight1", "insight2"]

    def test_to_dict(self):
        r = ReflectionRecord(
            round_idx=3,
            verdicts=[{"x": 1}],
            suggestions={"y": 2},
            key_insights=["a"],
        )
        d = r.to_dict()
        assert d["round"] == 3
        assert d["formula_feedback"] == [{"x": 1}]
        assert d["next_round_suggestions"] == {"y": 2}


# ============================================================================
# FinalFormulaRecord
# ============================================================================

class TestFinalFormulaRecord:
    def test_creation(self):
        f = FinalFormulaRecord(
            rank=1,
            formula_id="f-1",
            formula="rank(close)",
        )
        assert f.rank == 1
        assert f.formula_id == "f-1"

    def test_from_dict_basic(self):
        d = {
            "formula_id": "f-1",
            "formula": "x",
            "metrics": {"ic_mean": 0.05, "ir": 0.5},
        }
        f = FinalFormulaRecord.from_dict(d, rank=1)
        assert f.rank == 1
        assert f.ic_mean == 0.05
        assert f.ir == 0.5

    def test_from_dict_no_metrics(self):
        d = {"formula_id": "f-1", "formula": "x"}
        f = FinalFormulaRecord.from_dict(d, rank=2)
        assert f.ic_mean == 0.0
        assert f.rank == 2

    def test_from_dict_with_category(self):
        d = {"formula_id": "f-1", "formula": "x", "category": "momentum"}
        f = FinalFormulaRecord.from_dict(d, rank=1)
        assert f.category == "momentum"

    def test_to_dict(self):
        f = FinalFormulaRecord(
            rank=1, formula_id="f-1", formula="x",
            ic_mean=0.05, ir=0.5, category="momentum",
            selection_reason="top IR", risk_notes=["high turnover"],
        )
        d = f.to_dict()
        assert d["rank"] == 1
        assert d["selection_reason"] == "top IR"
        assert "high turnover" in d["risk_notes"]


# ============================================================================
# AlphaGptState
# ============================================================================

class TestAlphaGptState:
    def test_creation(self):
        s = AlphaGptState(objective="find momentum")
        assert s.objective == "find momentum"
        assert s.iterations_total == 5
        assert s.round_idx_hint == 1

    def test_default_collections(self):
        s = AlphaGptState(objective="x")
        assert s.all_ideas == []
        assert s.all_formulas == []
        assert s.all_evaluations == []
        assert s.all_reflections == []
        assert s.critic_output is None

    def test_aggregating_records(self):
        s = AlphaGptState(objective="x")
        s.all_ideas.append(IdeaRecord(id="1", name="x", category="y", description="z"))
        s.all_formulas.append(FormulaRecord(formula_id="f-1", idea_id="1", formula="x", round_discovered=1))
        assert len(s.all_ideas) == 1
        assert len(s.all_formulas) == 1

    def test_critic_output_assignment(self):
        s = AlphaGptState(objective="x")
        s.critic_output = {"top_k": ["f-1"]}
        assert s.critic_output["top_k"] == ["f-1"]


# ============================================================================
# Thinking Chain Integration
# ============================================================================

class TestThinkingChainIntegration:
    def test_idea_with_full_thinking(self):
        idea = IdeaRecord(
            id="1", name="x", category="y", description="z",
            thinking="<thinking>deep reasoning</thinking>",
            hypothesis="the hypothesis",
            mechanism="the mechanism",
            mentioned_ops=["rank", "ts_mean", "delta"],
        )
        d = idea.to_dict()
        assert "rank" in d["mentioned_ops"]
        assert "thinking" not in d  # thinking not exported

    def test_formula_with_thinking(self):
        f = FormulaRecord(
            formula_id="f-1", idea_id="i-1", formula="x", round_discovered=1,
            thinking="thinking", hypothesis="h", mentioned_ops=["add"],
        )
        d = f.to_dict()
        assert d["mentioned_ops"] == ["add"]