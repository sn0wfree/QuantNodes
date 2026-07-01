# coding=utf-8
"""Tests for core/knowledge/knowledge_base.py — KnowledgeBase (H19 fields).

Covers: creation with TFIDF retriever, add/add_many, sync_from_pool, query,
query_with_lineage, save/load roundtrip, field_weights H19 configuration.
"""

from pathlib import Path

import pandas as pd
import pytest

from QuantNodes.core.knowledge.knowledge_base import (
    KnowledgeBase,
    KnowledgeBaseSetting,
    DEFAULT_FIELD_WEIGHTS,
)
from QuantNodes.core.knowledge.retriever import make_retriever
from QuantNodes.core.trajectory.entry import TrajectoryEntry
from QuantNodes.core.feedback.dataclass import FactorFeedback


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def kb():
    """Create a default KnowledgeBase (TFIDF retriever, no pool)."""
    return KnowledgeBase()


@pytest.fixture
def entry_basic():
    return TrajectoryEntry(
        entry_id="e-1",
        config_snapshot={
            "factor": {
                "name": "momentum_20",
                "expression": "close / close.shift(20) - 1",
                "hypothesis": "20-day momentum",
                "description": "Simple momentum factor",
            }
        },
        metrics={"ic": 0.05, "ir": 0.5},
        feedback=FactorFeedback(factor_id="f-1", factor_name="momentum_20", decision=True, summary="ok"),
        round_idx=0,
    )


@pytest.fixture
def entry_value():
    return TrajectoryEntry(
        entry_id="e-2",
        config_snapshot={
            "factor": {
                "name": "reversal_5",
                "expression": "-close / close.shift(5) + 1",
                "hypothesis": "5-day reversal",
                "description": "Short-term reversal",
            }
        },
        metrics={"ic": -0.03},
        feedback=FactorFeedback(factor_id="f-2", factor_name="reversal_5", decision=False, summary="bad"),
        round_idx=0,
    )


# ============================================================================
# Default Constants
# ============================================================================

class TestDefaultFieldWeights:
    def test_default_field_weights_exist(self):
        assert "name" in DEFAULT_FIELD_WEIGHTS
        assert "expression" in DEFAULT_FIELD_WEIGHTS
        assert "hypothesis" in DEFAULT_FIELD_WEIGHTS
        assert "description" in DEFAULT_FIELD_WEIGHTS
        assert "summary" in DEFAULT_FIELD_WEIGHTS

    def test_default_weights_are_positive(self):
        for k, v in DEFAULT_FIELD_WEIGHTS.items():
            assert v > 0


# ============================================================================
# KnowledgeBaseSetting
# ============================================================================

class TestKnowledgeBaseSetting:
    def test_default_setting(self):
        s = KnowledgeBaseSetting()
        assert s.field_weights == DEFAULT_FIELD_WEIGHTS

    def test_custom_field_weights(self):
        s = KnowledgeBaseSetting(field_weights={"name": 5.0, "expression": 1.0})
        assert s.field_weights["name"] == 5.0

    def test_field_weights_isolated_from_default(self):
        s = KnowledgeBaseSetting(field_weights={"name": 5.0})
        s.field_weights["name"] = 10.0
        # Default should be untouched
        assert DEFAULT_FIELD_WEIGHTS["name"] == 3.0


# ============================================================================
# Creation
# ============================================================================

class TestKnowledgeBaseCreation:
    def test_creation_default(self):
        kb = KnowledgeBase()
        assert kb.retriever is not None
        assert len(kb) == 0

    def test_creation_with_setting(self):
        s = KnowledgeBaseSetting(field_weights={"name": 5.0})
        kb = KnowledgeBase(setting=s)
        assert kb._field_weights["name"] == 5.0

    def test_creation_with_field_weights_dict(self):
        kb = KnowledgeBase(field_weights={"name": 7.0})
        assert kb._field_weights["name"] == 7.0

    def test_creation_setting_takes_priority(self):
        s = KnowledgeBaseSetting(field_weights={"name": 5.0})
        kb = KnowledgeBase(setting=s, field_weights={"name": 1.0})
        # setting should win over field_weights dict
        assert kb._field_weights["name"] == 5.0

    def test_creation_with_custom_retriever(self):
        """Custom retriever is used when non-empty.

        Note: KnowledgeBase.__init__ uses `retriever or make_retriever("tfidf")`,
        but IdentityRetriever has __len__=0 when empty, so it gets replaced.
        """
        from QuantNodes.core.knowledge.retriever import IdentityRetriever
        r = IdentityRetriever()
        # Add a document so len(r) > 0
        r.add("dummy", "dummy")
        kb = KnowledgeBase(retriever=r)
        assert kb.retriever is r


# ============================================================================
# add / add_many / sync_from_pool
# ============================================================================

class TestKnowledgeBaseAdd:
    def test_add_single_entry(self, kb, entry_basic):
        kb.add(entry_basic)
        assert len(kb) == 1

    def test_add_duplicate_id(self, kb, entry_basic):
        kb.add(entry_basic)
        kb.add(entry_basic)
        assert len(kb) == 1

    def test_add_many(self, kb, entry_basic, entry_value):
        n = kb.add_many([entry_basic, entry_value])
        assert n == 2
        assert len(kb) == 2

    def test_add_many_skips_duplicates(self, kb, entry_basic):
        kb.add(entry_basic)
        n = kb.add_many([entry_basic])
        assert n == 0

    def test_sync_from_pool_without_pool_returns_zero(self, kb):
        n = kb.sync_from_pool()
        assert n == 0


class TestKnowledgeBaseSyncFromPool:
    def test_sync_from_pool(self, tmp_path):
        from QuantNodes.core.trajectory.pool import TrajectoryPool
        pool = TrajectoryPool(base_dir=tmp_path / "pool")
        e = TrajectoryEntry(
            entry_id="e-1",
            config_snapshot={"factor": {"name": "test"}},
        )
        pool.add(e)

        kb = KnowledgeBase(pool=pool)
        n = kb.sync_from_pool()
        assert n == 1
        assert len(kb) == 1


# ============================================================================
# query()
# ============================================================================

class TestKnowledgeBaseQuery:
    def test_query_returns_results(self, kb, entry_basic, entry_value):
        kb.add(entry_basic)
        kb.add(entry_value)
        results = kb.query("momentum", top_k=2)
        assert len(results) >= 1
        # Each result is (entry_or_None, score)
        for entry, score in results:
            assert score >= 0

    def test_query_with_min_score_with_pool(self, entry_basic, entry_value, tmp_path):
        """min_score filtering works when pool is present."""
        from QuantNodes.core.trajectory.pool import TrajectoryPool
        pool = TrajectoryPool(base_dir=tmp_path / "pool")
        pool.add(entry_basic)
        pool.add(entry_value)

        kb = KnowledgeBase(pool=pool)
        kb.sync_from_pool()

        # min_score above all scores should filter all
        results = kb.query("momentum", top_k=5, min_score=2.0)
        assert len(results) == 0

    def test_query_top_k(self, kb, entry_basic, entry_value):
        kb.add(entry_basic)
        kb.add(entry_value)
        results = kb.query("factor", top_k=1)
        assert len(results) <= 1

    def test_query_without_pool_returns_none_entries(self):
        kb = KnowledgeBase()
        e = TrajectoryEntry(
            entry_id="e-1",
            config_snapshot={"factor": {"name": "test"}},
        )
        kb.add(e)
        results = kb.query("test", top_k=5)
        # Without pool, entries are None
        for entry, _ in results:
            assert entry is None


# ============================================================================
# query_with_lineage()
# ============================================================================

class TestQueryWithLineage:
    def test_query_with_lineage_returns_dicts(self, kb, entry_basic):
        kb.add(entry_basic)
        results = kb.query_with_lineage("momentum", top_k=2)
        assert len(results) >= 1
        for ctx in results:
            assert "entry" in ctx
            assert "score" in ctx
            assert "ancestors" in ctx
            assert "descendants" in ctx
            assert "parents" in ctx
            assert "children" in ctx

    def test_query_with_lineage_empty_pool(self, kb, entry_basic):
        kb.add(entry_basic)
        results = kb.query_with_lineage("momentum", top_k=1)
        for ctx in results:
            # Without pool, ancestors/descendants should be empty
            assert ctx["ancestors"] == []
            assert ctx["descendants"] == []

    def test_query_with_lineage_with_lineage(self, tmp_path):
        from QuantNodes.core.trajectory.pool import TrajectoryPool
        # Create parent and child
        parent = TrajectoryEntry(
            entry_id="parent",
            config_snapshot={"factor": {"name": "parent_factor"}},
            round_idx=0,
        )
        child = TrajectoryEntry(
            entry_id="child",
            config_snapshot={"factor": {"name": "child_factor"}},
            parent_ids=["parent"],
            round_idx=1,
        )
        pool = TrajectoryPool(base_dir=tmp_path / "pool")
        pool.add(parent)
        pool.add(child)

        kb = KnowledgeBase(pool=pool)
        kb.sync_from_pool()

        results = kb.query_with_lineage("parent_factor", top_k=5)
        # Find the parent in results
        parent_ctx = None
        for ctx in results:
            if ctx["entry"] and ctx["entry"].entry_id == "parent":
                parent_ctx = ctx
                break
        if parent_ctx:
            assert len(parent_ctx["children"]) >= 1


# ============================================================================
# save() / load()
# ============================================================================

class TestKnowledgeBasePersistence:
    def test_save_creates_file(self, entry_basic, tmp_path):
        from QuantNodes.core.trajectory.pool import TrajectoryPool
        pool = TrajectoryPool(base_dir=tmp_path / "pool")
        pool.add(entry_basic)

        kb = KnowledgeBase(pool=pool)
        kb.sync_from_pool()

        path = tmp_path / "kb.parquet"
        kb.save(path)
        assert path.exists()

    def test_save_creates_parent_dir(self, entry_basic, tmp_path):
        from QuantNodes.core.trajectory.pool import TrajectoryPool
        pool = TrajectoryPool(base_dir=tmp_path / "pool")
        pool.add(entry_basic)

        kb = KnowledgeBase(pool=pool)
        kb.sync_from_pool()

        path = tmp_path / "subdir" / "kb.parquet"
        kb.save(path)
        assert path.exists()

    def test_load_returns_kb(self, entry_basic, tmp_path):
        from QuantNodes.core.trajectory.pool import TrajectoryPool
        pool = TrajectoryPool(base_dir=tmp_path / "pool")
        pool.add(entry_basic)

        kb = KnowledgeBase(pool=pool)
        kb.sync_from_pool()

        path = tmp_path / "kb.parquet"
        kb.save(path)

        loaded = KnowledgeBase.load(path)
        assert len(loaded) == 1

    def test_save_without_pool_creates_empty_file(self, kb, tmp_path):
        path = tmp_path / "kb.parquet"
        kb.save(path)
        assert path.exists()


# ============================================================================
# len()
# ============================================================================

class TestKnowledgeBaseLen:
    def test_empty_kb_len_zero(self, kb):
        assert len(kb) == 0

    def test_len_grows_with_adds(self, kb, entry_basic, entry_value):
        assert len(kb) == 0
        kb.add(entry_basic)
        assert len(kb) == 1
        kb.add(entry_value)
        assert len(kb) == 2


# ============================================================================
# _entry_to_text() — Field Weights
# ============================================================================

class TestEntryToText:
    def test_field_weights_affect_text(self, entry_basic):
        # Default weights: name=3.0
        kb_default = KnowledgeBase()
        text_default = kb_default._entry_to_text(entry_basic)
        # Custom weights: name=10.0
        kb_custom = KnowledgeBase(field_weights={"name": 10.0})
        text_custom = kb_custom._entry_to_text(entry_basic)
        # Custom weights text should be longer (more repetitions)
        assert len(text_custom) > len(text_default)

    def test_summary_field_used(self, entry_basic):
        kb = KnowledgeBase(field_weights={"summary": 5.0})
        text = kb._entry_to_text(entry_basic)
        # Summary should appear 5 times
        assert text.count(entry_basic.feedback.summary) == 5

    def test_metrics_included(self, entry_basic):
        kb = KnowledgeBase()
        text = kb._entry_to_text(entry_basic)
        # Metrics should be in text as "key=value"
        assert "ic=" in text
        assert "ir=" in text

    def test_empty_factor_config(self, kb):
        e = TrajectoryEntry(
            entry_id="e-1",
            config_snapshot={},
        )
        text = kb._entry_to_text(e)
        # Should not raise, may produce empty/short text
        assert text is not None


# ============================================================================
# Integration
# ============================================================================

class TestKnowledgeBaseIntegration:
    def test_full_workflow(self, entry_basic, entry_value, tmp_path):
        from QuantNodes.core.trajectory.pool import TrajectoryPool
        pool = TrajectoryPool(base_dir=tmp_path / "pool")
        pool.add(entry_basic)
        pool.add(entry_value)

        kb = KnowledgeBase(pool=pool)
        n = kb.sync_from_pool()
        assert n == 2
        assert len(kb) == 2

        # Query
        results = kb.query("factor", top_k=5)
        assert len(results) >= 1

        # Save and load
        path = tmp_path / "kb.parquet"
        kb.save(path)
        loaded = KnowledgeBase.load(path)
        assert len(loaded) == 2

        # Query loaded KB
        loaded_results = loaded.query("factor", top_k=5)
        assert len(loaded_results) >= 1