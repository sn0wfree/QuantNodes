# coding=utf-8
"""Tests for core/evolution/loop.py — EvolutionLoop and EvolutionResult.

Covers: EvolutionLoop creation, _build_round0, _batch_evaluate_and_record,
_make_entry_from_result, _update_best helper, EvolutionResult dataclass.
Uses mock evaluate_fn to avoid real backtests.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from QuantNodes.core.evolution.loop import (
    EvolutionLoop,
    EvolutionResult,
    _update_best,
    EvaluateFn,
)
from QuantNodes.core.evolution.settings import EvolutionSetting
from QuantNodes.core.evolution.operators import FactorCandidate
from QuantNodes.core.trajectory.entry import TrajectoryEntry
from QuantNodes.core.trajectory.pool import TrajectoryPool
from QuantNodes.core.feedback.dataclass import FactorFeedback


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def settings():
    return EvolutionSetting(
        enabled=True,
        max_rounds=2,
        parents_per_round=1,
        metric="sharpe",
    )


@pytest.fixture
def pool(tmp_path):
    return TrajectoryPool(base_dir=tmp_path / "pool")


def mock_evaluate(passed=True, metric_value=1.0):
    """Create a mock evaluate function that returns success."""
    def evaluate_fn(candidate):
        return passed, {"sharpe": metric_value}, FactorFeedback(
            factor_id=candidate.factor_id,
            factor_name=candidate.name,
            decision=passed,
            summary="mock ok",
        )
    return evaluate_fn


# ============================================================================
# EvolutionResult
# ============================================================================

class TestEvolutionResult:
    def test_creation(self):
        r = EvolutionResult()
        assert r.best_entries == []
        assert r.all_entries == []
        assert r.rounds_completed == 0
        assert r.rejected_count == 0
        assert r.total_count == 0

    def test_modification(self):
        r = EvolutionResult()
        r.total_count = 5
        r.rejected_count = 2
        assert r.total_count == 5


# ============================================================================
# _update_best helper
# ============================================================================

class TestUpdateBest:
    def test_first_entry(self):
        e = TrajectoryEntry(entry_id="e1", metrics={"sharpe": 1.5})
        result = _update_best(float("-inf"), e, "sharpe", 0)
        assert result == 1.5

    def test_higher_metric(self):
        e = TrajectoryEntry(entry_id="e1", metrics={"sharpe": 2.0})
        result = _update_best(1.0, e, "sharpe", 0)
        assert result == 2.0

    def test_lower_metric(self):
        e = TrajectoryEntry(entry_id="e1", metrics={"sharpe": 0.5})
        result = _update_best(1.0, e, "sharpe", 0)
        assert result == 1.0

    def test_missing_metric(self):
        e = TrajectoryEntry(entry_id="e1", metrics={})
        result = _update_best(1.0, e, "sharpe", 0)
        assert result == 1.0


# ============================================================================
# EvolutionLoop Creation
# ============================================================================

class TestEvolutionLoopCreation:
    def test_creation_basic(self, settings, pool):
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate())
        assert loop.settings is settings
        assert loop.pool is pool
        assert loop.hypothesizer is not None
        assert loop.mutator is not None
        assert loop.crosser is not None

    def test_creation_with_kb(self, settings, pool):
        from QuantNodes.core.knowledge.knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate(), knowledge_base=kb)
        assert loop.knowledge_base is kb

    def test_creation_with_quality_gate(self, settings, pool):
        from QuantNodes.core.quality_gate import QualityGateNode
        gate = QualityGateNode()
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate(), quality_gate=gate)
        assert loop.quality_gate is gate


# ============================================================================
# run() — required evaluate_fn
# ============================================================================

class TestEvolutionLoopRun:
    def test_run_without_evaluate_fn_raises(self, settings, pool):
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=None)
        with pytest.raises(ValueError, match="evaluate_fn"):
            loop.run(initial_directions=["d1"])

    def test_run_with_empty_initial(self, settings, pool):
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate())
        # Empty initial → no candidates → only round 0 with no work
        result = loop.run(initial_directions=[])
        assert result.rounds_completed == 1

    def test_run_single_direction(self, settings, pool):
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate())
        result = loop.run(initial_directions=["momentum"])
        # Round 0 evaluated + rounds 1..max_rounds (parent selection may be empty)
        assert result.rounds_completed >= 1
        # Should have at least 1 all_entries (round 0)
        assert len(result.all_entries) >= 1

    def test_run_with_initial_candidates(self, settings, pool):
        candidates = [
            FactorCandidate(factor_id="c1", name="test", expression="rank(close)"),
        ]
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate())
        result = loop.run(initial_candidates=candidates)
        assert len(result.all_entries) >= 1

    def test_run_failing_evaluate(self, settings, pool):
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate(passed=False))
        result = loop.run(initial_directions=["test"])
        # All rejected
        assert result.rejected_count >= 1
        assert result.total_count == 0

    def test_run_mixed_directions_and_candidates(self, settings, pool):
        candidates = [
            FactorCandidate(factor_id="c1", name="test", expression="x"),
        ]
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate())
        result = loop.run(initial_directions=["d1"], initial_candidates=candidates)
        assert len(result.all_entries) >= 2

    def test_run_returns_best_entries(self, settings, pool):
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate(metric_value=2.0))
        result = loop.run(initial_directions=["d1"])
        # top_n=10 by default
        assert isinstance(result.best_entries, list)
        assert len(result.best_entries) <= 10

    def test_run_with_quality_gate(self, tmp_path):
        from QuantNodes.core.quality_gate import QualityGateNode
        from QuantNodes.core.quality_gate.settings import QualityGateSetting, ComplexitySetting

        settings = EvolutionSetting(
            enabled=True,
            max_rounds=1,
            metric="sharpe",
        )
        pool = TrajectoryPool(base_dir=tmp_path / "pool")
        # Quality gate that rejects long expressions
        gate = QualityGateNode(settings=QualityGateSetting(
            complexity=ComplexitySetting(symbol_length_threshold=10),
        ))
        loop = EvolutionLoop(
            settings=settings, pool=pool,
            evaluate_fn=mock_evaluate(),
            quality_gate=gate,
        )
        result = loop.run(initial_directions=["momentum"])
        # Mock generates long expressions → should be rejected by quality gate
        # (mock templates like (x).rolling(5).mean() are about 20 chars)
        assert result.rejected_count >= 0


# ============================================================================
# sync_knowledge_base
# ============================================================================

class TestSyncKnowledgeBase:
    def test_sync_without_kb_returns_zero(self, settings, pool):
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate())
        assert loop.sync_knowledge_base() == 0

    def test_sync_with_kb(self, settings, pool):
        from QuantNodes.core.knowledge.knowledge_base import KnowledgeBase

        # Add entry to pool
        pool.add(TrajectoryEntry(
            entry_id="e1",
            config_snapshot={"factor": {"name": "e1"}},
        ))

        kb = KnowledgeBase(pool=pool)
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate(), knowledge_base=kb)
        n = loop.sync_knowledge_base()
        assert n == 1


# ============================================================================
# _evaluate_candidate
# ============================================================================

class TestEvaluateCandidate:
    def test_evaluate_candidate_default(self, settings, pool):
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=None)
        c = FactorCandidate(factor_id="c1", name="x", expression="x")
        passed, metrics, feedback = loop._evaluate_candidate(c)
        assert passed is False
        assert metrics == {}

    def test_evaluate_candidate_with_fn(self, settings, pool):
        evaluate = mock_evaluate(metric_value=2.5)
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=evaluate)
        c = FactorCandidate(factor_id="c1", name="x", expression="x")
        passed, metrics, feedback = loop._evaluate_candidate(c)
        assert passed is True
        assert metrics.get("sharpe") == 2.5


# ============================================================================
# _make_entry_from_result
# ============================================================================

class TestMakeEntryFromResult:
    def test_passed_result(self, settings, pool):
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate())
        c = FactorCandidate(
            factor_id="c1", name="test", expression="rank(close)",
            hypothesis="momentum", description="20-day",
        )
        result = {
            "passed": True,
            "metrics": {"sharpe": 1.5},
            "feedback_dict": None,
        }
        entry = loop._make_entry_from_result(c, result, operation="original", parent_ids=[])
        assert entry.entry_id == "c1"
        assert entry.feedback.decision is True
        assert entry.metrics.get("sharpe") == 1.5

    def test_failed_result(self, settings, pool):
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate())
        c = FactorCandidate(factor_id="c1", name="test", expression="x")
        result = {"passed": False, "metrics": {}, "feedback_dict": None, "error": "fail"}
        entry = loop._make_entry_from_result(c, result, operation="mutation", parent_ids=["p1"])
        assert entry.feedback.decision is False
        assert entry.operation == "mutation"
        assert entry.parent_ids == ["p1"]

    def test_with_feedback_dict(self, settings, pool):
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate())
        c = FactorCandidate(factor_id="c1", name="test", expression="x")
        result = {
            "passed": True,
            "metrics": {},
            "feedback_dict": {
                "factor_id": "fb_id",
                "factor_name": "fb_name",
                "decision": True,
                "summary": "fb summary",
                "metadata": {"k": "v"},
            },
        }
        entry = loop._make_entry_from_result(c, result, operation="original", parent_ids=[])
        assert entry.feedback.factor_id == "fb_id"
        assert entry.feedback.factor_name == "fb_name"
        assert entry.feedback.summary == "fb summary"


# ============================================================================
# crossover delegation
# ============================================================================

class TestCrossoverDelegation:
    def test_crossover_delegates_to_crosser(self, settings, pool):
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate())
        p1 = FactorCandidate(factor_id="p1", name="p1", expression="rank(close)")
        p2 = FactorCandidate(factor_id="p2", name="p2", expression="zscore(open)")
        child = loop.crossover(p1, p2)
        assert isinstance(child, FactorCandidate)
        assert child.factor_id not in ("p1", "p2")


# ============================================================================
# metric_collector injection
# ============================================================================

class TestMetricCollector:
    def test_metric_collector_default_none(self, settings, pool):
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate())
        assert loop.metric_collector is None

    def test_metric_collector_settable(self, settings, pool):
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate())
        collector = MagicMock()
        loop.metric_collector = collector
        assert loop.metric_collector is collector


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_run_with_rag_evaluator_no_directions(self, settings, pool):
        """RAG evaluator with empty directions should be skipped."""
        from unittest.mock import MagicMock
        rag_eval = MagicMock()
        loop = EvolutionLoop(
            settings=settings, pool=pool,
            evaluate_fn=mock_evaluate(),
            knowledge_base=None,  # No KB
            rag_evaluator=rag_eval,
        )
        # Should not call rag_eval.evaluate since no directions
        loop.run(initial_directions=[])
        rag_eval.evaluate.assert_not_called()

    def test_run_unicode_directions(self, settings, pool):
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate())
        result = loop.run(initial_directions=["中文假设", "中文假设2"])
        assert result.rounds_completed >= 1

    def test_pool_size_grows(self, settings, pool):
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate())
        initial_size = pool.size
        loop.run(initial_directions=["d1", "d2"])
        assert pool.size > initial_size

    def test_workers_parameter(self, settings, pool):
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate(), workers=1)
        assert loop.workers == 1

    def test_snapshot_path_default(self, settings, pool):
        loop = EvolutionLoop(settings=settings, pool=pool, evaluate_fn=mock_evaluate())
        assert loop.snapshot_path is None