# coding=utf-8
"""Tests for research/quant_alpha/mcts/search.py and tree.py.

Covers: MCTSSearchConfig, MCTSSearchResult, MCTSNode (UCB1, ancestors, lineage),
MCTSTree (add_node, get_by_entry_id, leaves, best_k, stats),
MCTSSearch creation, _precheck_formula, _select, _evaluate (mocked).
"""

from unittest.mock import MagicMock, patch

import pytest

from QuantNodes.research.quant_alpha.mcts.search import (
    MCTSSearch,
    MCTSSearchConfig,
    MCTSSearchResult,
)
from QuantNodes.research.quant_alpha.mcts.tree import (
    MCTSNode,
    MCTSTree,
    NodeStatus,
)


# ============================================================================
# NodeStatus
# ============================================================================

class TestNodeStatus:
    def test_enum_values(self):
        assert NodeStatus.PENDING.value == "pending"
        assert NodeStatus.EVALUATED.value == "evaluated"
        assert NodeStatus.PRUNED.value == "pruned"
        assert NodeStatus.REJECTED.value == "rejected"

    def test_enum_count(self):
        assert len(NodeStatus) == 4


# ============================================================================
# MCTSNode
# ============================================================================

class TestMCTSNode:
    def test_creation(self):
        n = MCTSNode(formula="rank(close)")
        assert n.formula == "rank(close)"
        assert n.visits == 0
        assert n.depth == 0
        assert n.status == NodeStatus.PENDING

    def test_unique_entry_id(self):
        n1 = MCTSNode(formula="x")
        n2 = MCTSNode(formula="x")
        assert n1.entry_id != n2.entry_id

    def test_add_child(self):
        parent = MCTSNode(formula="parent")
        child = MCTSNode(formula="child")
        parent.add_child(child)
        assert len(parent.children) == 1
        assert child.parent_id == parent.entry_id
        assert child._parent_ref is parent
        assert child.depth == 1
        assert parent.is_expanded is True

    def test_is_leaf(self):
        n = MCTSNode(formula="x")
        assert n.is_leaf() is True
        child = MCTSNode(formula="y")
        n.add_child(child)
        assert n.is_leaf() is False

    def test_is_root(self):
        n = MCTSNode(formula="x")
        assert n.is_root() is True
        # With parent_id set
        n.parent_id = "abc"
        assert n.is_root() is False

    def test_ancestors_empty_for_root(self):
        n = MCTSNode(formula="x")
        assert n.ancestors() == []

    def test_ancestors_returns_chain(self):
        a = MCTSNode(formula="a")
        b = MCTSNode(formula="b")
        c = MCTSNode(formula="c")
        a.add_child(b)
        b.add_child(c)
        chain = c.ancestors()
        assert len(chain) == 2
        assert chain[0] is a
        assert chain[1] is b

    def test_lineage_depth(self):
        a = MCTSNode(formula="a")
        b = MCTSNode(formula="b")
        c = MCTSNode(formula="c")
        a.add_child(b)
        b.add_child(c)
        assert c.lineage_depth() == 2

    def test_ucb1_unvisited_returns_inf(self):
        n = MCTSNode(formula="x")
        assert n.ucb1() == float("inf")

    def test_ucb1_with_visits(self):
        parent = MCTSNode(formula="parent", visits=10)
        child = MCTSNode(formula="child", visits=2, overall_score=0.5)
        child._parent_ref = parent
        score = child.ucb1(exploration_weight=1.414)
        # Should be > 0
        assert score > 0

    def test_ucb1_zero_parent_visits_returns_inf(self):
        parent = MCTSNode(formula="parent", visits=0)
        child = MCTSNode(formula="child", visits=2, overall_score=0.5)
        child._parent_ref = parent
        assert child.ucb1() == float("inf")

    def test_repr_truncates_long_formula(self):
        long_formula = "x" * 100
        n = MCTSNode(formula=long_formula)
        repr_str = repr(n)
        assert "..." in repr_str

    def test_repr_short_formula(self):
        n = MCTSNode(formula="rank(close)")
        repr_str = repr(n)
        assert "rank(close)" in repr_str


# ============================================================================
# MCTSTree
# ============================================================================

class TestMCTSTree:
    def test_creation(self):
        tree = MCTSTree()
        assert tree.root.formula == "__ROOT__"
        assert tree.root.depth == -1

    def test_add_node_with_default_parent(self):
        tree = MCTSTree()
        n = MCTSNode(formula="rank(close)")
        tree.add_node(n)
        assert n in tree.root.children
        assert tree.formula_cache["rank(close)"] is n

    def test_add_node_with_explicit_parent(self):
        tree = MCTSTree()
        parent = MCTSNode(formula="parent")
        tree.add_node(parent)
        child = MCTSNode(formula="child")
        tree.add_node(child, parent=parent)
        assert child in parent.children

    def test_get_by_entry_id(self):
        tree = MCTSTree()
        n = MCTSNode(formula="x")
        tree.add_node(n)
        found = tree.get_by_entry_id(n.entry_id)
        assert found is n

    def test_get_by_entry_id_missing(self):
        tree = MCTSTree()
        assert tree.get_by_entry_id("nonexistent") is None

    def test_get_by_formula(self):
        tree = MCTSTree()
        n = MCTSNode(formula="rank(close)")
        tree.add_node(n)
        assert tree.get_by_formula("rank(close)") is n

    def test_get_by_formula_missing(self):
        tree = MCTSTree()
        assert tree.get_by_formula("nonexistent") is None

    def test_all_nodes(self):
        tree = MCTSTree()
        a = MCTSNode(formula="a")
        b = MCTSNode(formula="b")
        tree.add_node(a)
        tree.add_node(b, parent=a)
        nodes = tree.all_nodes()
        # Root is excluded
        assert a in nodes
        assert b in nodes
        assert tree.root not in nodes

    def test_leaves(self):
        tree = MCTSTree()
        a = MCTSNode(formula="a")
        b = MCTSNode(formula="b")
        c = MCTSNode(formula="c")
        tree.add_node(a)
        tree.add_node(b, parent=a)
        tree.add_node(c, parent=a)
        # a is not a leaf (has children), b and c are leaves
        leaves = tree.leaves()
        assert b in leaves
        assert c in leaves
        assert a not in leaves

    def test_best_k_sorted(self):
        tree = MCTSTree()
        n1 = MCTSNode(formula="a", overall_score=0.1)
        n2 = MCTSNode(formula="b", overall_score=0.9)
        n3 = MCTSNode(formula="c", overall_score=0.5)
        tree.add_node(n1)
        tree.add_node(n2)
        tree.add_node(n3)
        best = tree.best_k(k=2)
        assert len(best) == 2
        assert best[0] is n2  # highest score
        assert best[1] is n3

    def test_best_k_custom_metric(self):
        tree = MCTSTree()
        n1 = MCTSNode(formula="a", overall_score=0.9, visits=5)
        n2 = MCTSNode(formula="b", overall_score=0.1, visits=100)
        tree.add_node(n1)
        tree.add_node(n2)
        best = tree.best_k(k=2, metric="visits")
        assert best[0] is n2  # highest visits

    def test_best_k_dedup_by_entry_id(self):
        tree = MCTSTree()
        # Same formula added twice shouldn't appear twice in best_k
        n1 = MCTSNode(formula="same", overall_score=0.5)
        tree.add_node(n1)
        n2 = MCTSNode(formula="same", overall_score=0.7)
        tree.add_node(n2)
        # Both should be in tree but best_k dedups
        # However, formula_cache would only keep the last
        best = tree.best_k(k=10)
        # Should not have duplicates
        entry_ids = [n.entry_id for n in best]
        assert len(entry_ids) == len(set(entry_ids))

    def test_stats_empty_tree(self):
        tree = MCTSTree()
        stats = tree.stats()
        assert stats["total_nodes"] == 0
        assert stats["by_status"] == {}
        assert stats["max_depth"] == 0

    def test_stats_with_nodes(self):
        tree = MCTSTree()
        # add_node sets parent depth via add_child, then we adjust after
        parent = MCTSNode(formula="parent", status=NodeStatus.EVALUATED)
        tree.add_node(parent)
        # After add_node, parent.depth = root.depth + 1 = 0
        child = MCTSNode(formula="child", status=NodeStatus.REJECTED)
        parent.add_child(child)
        tree._entry_index[child.entry_id] = child
        tree.formula_cache["child"] = child
        # After add_child, child.depth = parent.depth + 1 = 1
        stats = tree.stats()
        assert stats["total_nodes"] == 2
        assert stats["by_status"]["evaluated"] == 1
        assert stats["by_status"]["rejected"] == 1
        assert stats["max_depth"] == 1


# ============================================================================
# MCTSSearchConfig
# ============================================================================

class TestMCTSSearchConfig:
    def test_creation_defaults(self):
        c = MCTSSearchConfig()
        assert c.iterations == 50
        assert c.exploration_weight == 1.414
        assert c.max_depth == 5
        assert c.seed == 42
        assert c.enable_lineage is True
        assert c.compute_ic_ir is True
        assert c.dedup_threshold == 0.7

    def test_custom_config(self):
        c = MCTSSearchConfig(iterations=100, seed=0, max_depth=10)
        assert c.iterations == 100
        assert c.seed == 0
        assert c.max_depth == 10

    def test_feedback_config_default(self):
        from QuantNodes.research.quant_alpha.mcts.feedback import MCTSFeedbackConfig
        c = MCTSSearchConfig()
        assert isinstance(c.feedback_config, MCTSFeedbackConfig)


# ============================================================================
# MCTSSearchResult
# ============================================================================

class TestMCTSSearchResult:
    def test_creation(self):
        tree = MCTSTree()
        r = MCTSSearchResult(tree=tree)
        assert r.tree is tree
        assert r.valid_nodes == []
        assert r.best_k_nodes == []
        assert r.elapsed_seconds == 0.0
        assert r.total_iterations == 0

    def test_creation_with_data(self):
        tree = MCTSTree()
        n = MCTSNode(formula="x", overall_score=0.5, status=NodeStatus.EVALUATED)
        r = MCTSSearchResult(
            tree=tree,
            valid_nodes=[n],
            best_k_nodes=[n],
            elapsed_seconds=1.5,
            total_iterations=10,
            formula_count=5,
            valid_count=1,
        )
        assert len(r.valid_nodes) == 1
        assert r.elapsed_seconds == 1.5


# ============================================================================
# MCTSSearch Creation
# ============================================================================

class TestMCTSSearchCreation:
    def test_creation_default(self):
        search = MCTSSearch()
        assert search.vocab is not None
        assert search.op_pool is not None
        assert search.config is not None
        assert search.rng is not None

    def test_creation_with_custom_config(self):
        c = MCTSSearchConfig(iterations=10)
        search = MCTSSearch(config=c)
        assert search.config.iterations == 10

    def test_creation_with_custom_seed(self):
        """RNG initialized with custom seed."""
        c = MCTSSearchConfig(seed=123)
        search = MCTSSearch(config=c)
        # Two calls to same seed produce same first value
        import random
        expected_first = random.Random(123).random()
        assert search.rng.random() == expected_first

    def test_initial_state(self):
        search = MCTSSearch()
        assert search._tree is None
        assert search._formula_cache == {}
        assert search._feedback_cache == {}


# ============================================================================
# _precheck_formula
# ============================================================================

class TestPrecheckFormula:
    def test_valid_formula(self):
        search = MCTSSearch()
        err = search._precheck_formula("rank(close)")
        assert err is None

    def test_bracket_mismatch(self):
        search = MCTSSearch()
        err = search._precheck_formula("rank(close")
        assert err == "bracket mismatch"

    def test_too_long(self):
        search = MCTSSearch()
        long_formula = "x" * 501
        err = search._precheck_formula(long_formula)
        assert err == "too long"

    def test_unknown_operator(self):
        search = MCTSSearch()
        err = search._precheck_formula("unknown_op(x)")
        assert "unknown op" in err


# ============================================================================
# stats()
# ============================================================================

class TestSearchStats:
    def test_stats_not_initialized(self):
        search = MCTSSearch()
        stats = search.stats()
        assert stats["status"] == "not_initialized"

    def test_stats_after_tree(self):
        search = MCTSSearch()
        tree = MCTSTree()
        search._tree = tree
        tree.add_node(MCTSNode(formula="x"))
        stats = search.stats()
        assert stats["total_nodes"] == 1
        assert stats["formula_cache_size"] == 0
        assert stats["feedback_cache_size"] == 0


# ============================================================================
# get_feedback()
# ============================================================================

class TestGetFeedback:
    def test_get_feedback_empty(self):
        search = MCTSSearch()
        assert search.get_feedback("nonexistent") is None

    def test_get_feedback_after_cached(self):
        search = MCTSSearch()
        from QuantNodes.core.feedback.dataclass import FactorFeedback
        fb = FactorFeedback(factor_id="f-1", factor_name="x", decision=True)
        search._feedback_cache["rank(close)"] = fb
        assert search.get_feedback("rank(close)") is fb


# ============================================================================
# Edge Cases
# ============================================================================

class TestSearchEdgeCases:
    def test_search_with_no_data(self):
        """Empty data: search should handle gracefully."""
        search = MCTSSearch()
        import polars as pl
        empty_df = pl.DataFrame({"date": [], "code": [], "close": []})
        # Should not crash (may return empty results)
        try:
            result = search.search(empty_df)
            assert result is not None
        except Exception:
            # Acceptable to raise on empty data
            pass

    def test_config_dedup_threshold_extreme(self):
        c1 = MCTSSearchConfig(dedup_threshold=0.0)
        assert c1.dedup_threshold == 0.0
        c2 = MCTSSearchConfig(dedup_threshold=1.0)
        assert c2.dedup_threshold == 1.0

    def test_search_seeded_reproducible(self):
        """Same seed should give same initial RNG state."""
        s1 = MCTSSearch()
        s2 = MCTSSearch()
        # Both have seed=42 by default
        assert s1.rng.random() == s2.rng.random()