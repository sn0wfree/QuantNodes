# coding=utf-8
"""Tests for core/evolution/ — settings, operators (Hypothesizer/Mutator/Crosser), loop.

Covers: OperatorSetting, EvolutionSetting, FactorCandidate, mock operator
fallbacks, EvolutionLoop configuration, EvolutionResult dataclass.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from QuantNodes.core.evolution.settings import OperatorSetting, EvolutionSetting
from QuantNodes.core.evolution.operators import (
    FactorCandidate,
    BaseOperator,
    Hypothesizer,
    Mutator,
    Crosser,
    _mock_variant,
    _HYPOTHESIZE_PROMPT,
    _MUTATE_PROMPT,
    _CROSSOVER_PROMPT,
)


# ============================================================================
# OperatorSetting
# ============================================================================

class TestOperatorSetting:
    def test_defaults(self):
        s = OperatorSetting()
        assert s.enabled is True
        assert s.model == "mock"
        assert s.max_correction_attempts == 3
        assert s.seed == 42

    def test_custom(self):
        s = OperatorSetting(
            enabled=False,
            model="deepseek-v3",
            max_correction_attempts=5,
            seed=0,
        )
        assert s.enabled is False
        assert s.model == "deepseek-v3"


# ============================================================================
# EvolutionSetting
# ============================================================================

class TestEvolutionSetting:
    def test_defaults(self):
        s = EvolutionSetting()
        assert s.enabled is False
        assert s.max_rounds == 3
        assert s.parents_per_round == 1
        assert s.metric == "sharpe"
        assert s.top_n == 10

    def test_any_operator_enabled_default(self):
        s = EvolutionSetting()
        # Default: all enabled
        assert s.any_operator_enabled() is True

    def test_any_operator_enabled_all_disabled(self):
        s = EvolutionSetting(
            hypothesizer=OperatorSetting(enabled=False),
            mutator=OperatorSetting(enabled=False),
            crosser=OperatorSetting(enabled=False),
        )
        assert s.any_operator_enabled() is False

    def test_custom_subsections(self):
        s = EvolutionSetting(
            hypothesizer=OperatorSetting(model="gpt-4"),
            max_rounds=10,
            top_n=20,
        )
        assert s.hypothesizer.model == "gpt-4"
        assert s.max_rounds == 10
        assert s.top_n == 20

    def test_selection_strategy(self):
        s = EvolutionSetting(parent_selection_strategy="best")
        assert s.parent_selection_strategy == "best"


# ============================================================================
# FactorCandidate
# ============================================================================

class TestFactorCandidate:
    def test_creation(self):
        c = FactorCandidate(
            factor_id="f-1",
            name="test",
            expression="rank(close)",
        )
        assert c.factor_id == "f-1"
        assert c.name == "test"
        assert c.expression == "rank(close)"
        assert c.hypothesis == ""
        assert c.description == ""

    def test_with_all_fields(self):
        c = FactorCandidate(
            factor_id="f-1",
            name="test",
            expression="x",
            hypothesis="momentum",
            description="20-day momentum",
        )
        assert c.hypothesis == "momentum"
        assert c.description == "20-day momentum"


# ============================================================================
# BaseOperator
# ============================================================================

class TestBaseOperator:
    def test_creation_mock(self):
        op = BaseOperator(model="mock")
        assert op.model == "mock"
        assert op._llm_callable is None

    def test_creation_non_mock_requires_callable(self):
        with pytest.raises(ValueError, match="llm_callable"):
            BaseOperator(model="gpt-4")

    def test_creation_with_callable(self):
        def my_llm(prompt):
            return '{"name": "x", "expression": "y"}'

        op = BaseOperator(model="gpt-4", llm_callable=my_llm)
        assert op._llm_callable is my_llm

    def test_call_uses_llm_callable(self):
        def my_llm(prompt):
            return "custom response"

        op = BaseOperator(model="mock", llm_callable=my_llm)
        result = op._call("test prompt")
        assert result == "custom response"

    def test_call_uses_mock_when_no_callable(self):
        op = BaseOperator(model="mock", seed=42)
        result = op._call(_HYPOTHESIZE_PROMPT)
        # Mock returns valid JSON
        import json
        data = json.loads(result)
        assert "name" in data
        assert "expression" in data


# ============================================================================
# _mock_variant
# ============================================================================

class TestMockVariant:
    def test_hypothesize_mock(self):
        prompt = _HYPOTHESIZE_PROMPT.format(
            hypothesis="momentum",
            description="20-day",
        )
        result = _mock_variant(prompt)
        assert "name" in result
        assert "expression" in result
        assert "description" in result

    def test_mutate_mock(self):
        prompt = _MUTATE_PROMPT.format(
            parent_expression="rank(close)",
            parent_hypothesis="momentum",
            parent_description="20-day",
        )
        result = _mock_variant(prompt)
        assert result["name"].startswith("m_")
        assert "rank(close)" in result["expression"] or "shift" in result["expression"]

    def test_crossover_mock(self):
        prompt = _CROSSOVER_PROMPT.format(
            p1_expression="rank(close)",
            p1_description="d1",
            p2_expression="zscore(open)",
            p2_description="d2",
        )
        result = _mock_variant(prompt)
        assert result["name"].startswith("x_")
        # Should reference parent names (extracted up to first paren)
        assert "rank" in result["expression"] or "zscore" in result["expression"]

    def test_no_parent_no_hypothesis(self):
        """Empty prompt → default hypothesize fallback."""
        result = _mock_variant("empty prompt with no markers")
        assert "name" in result
        assert "expression" in result


# ============================================================================
# Hypothesizer
# ============================================================================

class TestHypothesizer:
    def test_creation_default(self):
        h = Hypothesizer()
        assert h.knowledge_base is None
        assert h.rag_top_k == 3

    def test_creation_with_kb(self, tmp_path):
        from QuantNodes.core.knowledge.knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        h = Hypothesizer(knowledge_base=kb)
        assert h.knowledge_base is kb

    def test_hypothesize_mock(self):
        h = Hypothesizer()
        cand = h.hypothesize(direction="momentum", description="20-day")
        assert isinstance(cand, FactorCandidate)
        assert cand.factor_id is not None
        assert cand.expression != ""

    def test_hypothesize_default_name(self):
        h = Hypothesizer()
        cand = h.hypothesize(direction="my_direction", description="")
        # Should use direction[:8] as fallback name
        assert cand.name != ""

    def test_hypothesize_with_kb(self, tmp_path):
        """Hypothesizer with KB should still produce valid candidate."""
        from QuantNodes.core.knowledge.knowledge_base import KnowledgeBase
        pool = __import__(
            "QuantNodes.core.trajectory.pool", fromlist=["TrajectoryPool"]
        ).TrajectoryPool
        from QuantNodes.core.trajectory.pool import TrajectoryPool
        from QuantNodes.core.trajectory.entry import TrajectoryEntry

        p = TrajectoryPool(base_dir=tmp_path / "pool")
        p.add(TrajectoryEntry(
            entry_id="e1",
            config_snapshot={"factor": {
                "name": "rank_close",
                "expression": "rank(close)",
                "hypothesis": "momentum hypothesis",
                "description": "rank test",
            }},
            metrics={"sharpe": 1.0},
        ))

        kb = KnowledgeBase(pool=p)
        kb.sync_from_pool()

        h = Hypothesizer(knowledge_base=kb)
        cand = h.hypothesize(direction="momentum", description="test")
        assert isinstance(cand, FactorCandidate)
        assert cand.expression != ""

    def test_hypothesize_with_invalid_llm_response(self):
        """LLM returns invalid JSON → fallback to mock."""
        def bad_llm(prompt):
            return "not valid json"

        h = Hypothesizer(model="gpt-4", llm_callable=bad_llm, max_correction_attempts=2)
        # After max_correction_attempts, falls back to mock
        cand = h.hypothesize(direction="test", description="test")
        # Should still return a FactorCandidate via mock fallback
        assert isinstance(cand, FactorCandidate)


# ============================================================================
# Mutator
# ============================================================================

class TestMutator:
    def test_creation(self):
        m = Mutator()
        assert m.model == "mock"

    def test_mutate_basic(self):
        m = Mutator()
        parent = FactorCandidate(
            factor_id="p-1",
            name="parent",
            expression="rank(close)",
            hypothesis="h",
            description="d",
        )
        child = m.mutate(parent)
        assert isinstance(child, FactorCandidate)
        assert child.factor_id is not None
        assert child.factor_id != parent.factor_id  # new ID
        assert child.name.startswith("m_")

    def test_mutate_inherits_hypothesis(self):
        m = Mutator()
        parent = FactorCandidate(
            factor_id="p", name="p", expression="rank(x)",
            hypothesis="original_hyp", description="d",
        )
        child = m.mutate(parent)
        assert child.hypothesis == "original_hyp"

    def test_mutate_with_invalid_response(self):
        def bad_llm(prompt):
            return "invalid"

        m = Mutator(model="gpt-4", llm_callable=bad_llm, max_correction_attempts=1)
        parent = FactorCandidate(factor_id="p", name="p", expression="x")
        child = m.mutate(parent)
        # Falls back to mock
        assert isinstance(child, FactorCandidate)


# ============================================================================
# Crosser
# ============================================================================

class TestCrosser:
    def test_creation(self):
        c = Crosser()
        assert c.model == "mock"

    def test_crossover_basic(self):
        c = Crosser()
        p1 = FactorCandidate(factor_id="p1", name="p1", expression="rank(close)")
        p2 = FactorCandidate(factor_id="p2", name="p2", expression="zscore(open)")
        child = c.crossover(p1, p2)
        assert isinstance(child, FactorCandidate)
        assert child.factor_id is not None
        assert child.factor_id not in ("p1", "p2")
        assert child.name.startswith("x_")

    def test_crossover_combines_hypotheses(self):
        c = Crosser()
        p1 = FactorCandidate(factor_id="p1", name="p1", expression="x", hypothesis="h1")
        p2 = FactorCandidate(factor_id="p2", name="p2", expression="y", hypothesis="h2")
        child = c.crossover(p1, p2)
        assert "combo" in child.hypothesis
        assert "h1" in child.hypothesis
        assert "h2" in child.hypothesis


# ============================================================================
# EvolutionLoop (without running)
# ============================================================================

class TestEvolutionLoopImports:
    """Just verify imports work."""

    def test_import(self):
        from QuantNodes.core.evolution.loop import EvolutionLoop, EvolutionResult
        assert EvolutionLoop is not None
        assert EvolutionResult is not None

    def test_evolution_result_creation(self):
        from QuantNodes.core.evolution.loop import EvolutionResult
        r = EvolutionResult()
        assert r.best_entries == []
        assert r.all_entries == []
        assert r.rounds_completed == 0
        assert r.rejected_count == 0
        assert r.total_count == 0


# ============================================================================
# Integration
# ============================================================================

class TestIntegration:
    def test_hypothesizer_mutator_chain(self):
        """Full hypothesize → mutate flow."""
        h = Hypothesizer()
        m = Mutator()

        # Hypothesize
        initial = h.hypothesize(direction="momentum", description="test")
        assert initial is not None

        # Mutate the initial
        child = m.mutate(initial)
        assert child is not None
        assert child.factor_id != initial.factor_id

    def test_crosser_chain(self):
        h = Hypothesizer()
        c = Crosser()

        p1 = h.hypothesize(direction="d1", description="d1")
        p2 = h.hypothesize(direction="d2", description="d2")
        child = c.crossover(p1, p2)
        assert child is not None


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_prompts_format_correctly(self):
        """All prompts should format with their placeholders."""
        # Hypothesize prompt
        p1 = _HYPOTHESIZE_PROMPT.format(hypothesis="x", description="y")
        assert "x" in p1
        assert "y" in p1

        # Mutate prompt
        p2 = _MUTATE_PROMPT.format(
            parent_expression="x",
            parent_hypothesis="h",
            parent_description="d",
        )
        assert "x" in p2

        # Crossover prompt
        p3 = _CROSSOVER_PROMPT.format(
            p1_expression="x",
            p1_description="d1",
            p2_expression="y",
            p2_description="d2",
        )
        assert "x" in p3
        assert "y" in p3

    def test_mock_variant_deterministic(self):
        """Same prompt should produce same mock output."""
        prompt = "父因子: rank(close)\n研究假设: test"
        r1 = _mock_variant(prompt)
        r2 = _mock_variant(prompt)
        assert r1 == r2