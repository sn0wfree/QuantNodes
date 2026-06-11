"""谱系 RAG (Week 8) 测试 — 10 tests。

覆盖:
    - expand_lineage (4)
    - expand_lineage_batch (1)
    - build_rag_prompt 谱系 (3)
    - query_with_lineage 深度 (2)
"""
from __future__ import annotations

import tempfile

import pytest

from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.knowledge import (
    IdentityRetriever,
    KnowledgeBase,
    build_rag_prompt,
    expand_lineage,
    expand_lineage_batch,
)
from QuantNodes.core.trajectory import TrajectoryEntry, TrajectoryPool


# ============================================================================
# Fixtures
# ============================================================================

def _make_entry(
    eid: str,
    name: str,
    hypothesis: str = "momentum",
    description: str = "d",
    parent_ids: list | None = None,
    round_idx: int = 0,
    operation: str = "original",
    sharpe: float = 1.0,
) -> TrajectoryEntry:
    return TrajectoryEntry(
        entry_id=eid,
        round_idx=round_idx,
        operation=operation,
        parent_ids=parent_ids or [],
        config_snapshot={
            "factor": {
                "name": name,
                "expression": f"close - open ({name})",
                "hypothesis": hypothesis,
                "description": description,
            },
        },
        feedback=FactorFeedback(factor_name=name, decision=True, summary="ok"),
        metrics={"sharpe": sharpe},
    )


@pytest.fixture
def chain_pool() -> TrajectoryPool:
    """e1 → e2 → e3 → e4 (链状, 4 代)。"""
    pool = TrajectoryPool(tempfile.mkdtemp())
    pool.add(_make_entry("e1", "m20d", round_idx=0))
    pool.add(_make_entry("e2", "m60d", parent_ids=["e1"], round_idx=1, operation="mutation"))
    pool.add(_make_entry("e3", "m120d", parent_ids=["e2"], round_idx=2, operation="mutation"))
    pool.add(_make_entry("e4", "m240d", parent_ids=["e3"], round_idx=3, operation="crossover"))
    return pool


@pytest.fixture
def branch_pool() -> TrajectoryPool:
    """e1 → {e2, e3}, e2 → e4 (分支)。"""
    pool = TrajectoryPool(tempfile.mkdtemp())
    pool.add(_make_entry("e1", "root", round_idx=0))
    pool.add(_make_entry("e2", "branch_a", parent_ids=["e1"], round_idx=1, operation="mutation"))
    pool.add(_make_entry("e3", "branch_b", parent_ids=["e1"], round_idx=1, operation="mutation"))
    pool.add(_make_entry("e4", "leaf", parent_ids=["e2"], round_idx=2, operation="crossover"))
    return pool


# ============================================================================
# 1. expand_lineage (4)
# ============================================================================

def test_expand_lineage_ancestors_chain(chain_pool):
    """链状 4 代, depth=3 上溯 3 层。"""
    expanded = expand_lineage(chain_pool, "e4", max_ancestor_depth=3, max_descendant_depth=0)
    ancestors = [(d, e.entry_id) for d, e in expanded["ancestors"]]
    assert ancestors == [(1, "e3"), (2, "e2"), (3, "e1")]


def test_expand_lineage_descendants_branch(branch_pool):
    """分支 DAG, depth=2 下探 2 层。"""
    expanded = expand_lineage(branch_pool, "e1", max_ancestor_depth=0, max_descendant_depth=2)
    descendants = [(d, e.entry_id) for d, e in expanded["descendants"]]
    # depth 1: e2, e3; depth 2: e4 (from e2)
    assert (1, "e2") in descendants
    assert (1, "e3") in descendants
    assert (2, "e4") in descendants


def test_expand_lineage_both_directions(branch_pool):
    """同时上溯 + 下探 (e2 视角: 上 1=e1, 下 1=e4)。"""
    expanded = expand_lineage(branch_pool, "e2", max_ancestor_depth=2, max_descendant_depth=2)
    ancestors = [(d, e.entry_id) for d, e in expanded["ancestors"]]
    descendants = [(d, e.entry_id) for d, e in expanded["descendants"]]
    assert ancestors == [(1, "e1")]
    assert descendants == [(1, "e4")]


def test_expand_lineage_max_limit_respected(chain_pool):
    """max_ancestors 限制生效。"""
    expanded = expand_lineage(
        chain_pool, "e4",
        max_ancestor_depth=10,
        max_ancestors=2,  # 只取 2 个
    )
    assert len(expanded["ancestors"]) == 2
    # 浅的在前
    assert expanded["ancestors"][0][0] == 1
    assert expanded["ancestors"][1][0] == 2


# ============================================================================
# 2. expand_lineage_batch (1)
# ============================================================================

def test_expand_lineage_batch_dedup(branch_pool):
    """批量展开, 去重。"""
    results = expand_lineage_batch(
        branch_pool, ["e2", "e3", "e1"],
        max_ancestor_depth=1, max_descendant_depth=0,
    )
    assert len(results) == 3
    # e2 和 e3 都展开自 e1
    e1_parents = {r["root"].entry_id for r in results}
    assert e1_parents == {"e2", "e3", "e1"}


# ============================================================================
# 3. build_rag_prompt 谱系 (3)
# ============================================================================

def test_rag_prompt_includes_ancestor(branch_pool):
    """prompt 包含 ancestor 段。"""
    kb = KnowledgeBase(IdentityRetriever(), pool=branch_pool)
    for e in branch_pool.all():
        kb.add(e)
    # query "branch_a mutation" 应能选出 e2 (有 ancestor e1)
    prompt = build_rag_prompt(
        "branch_a mutation crossover", "leaf", kb=kb, top_k=1,
        max_ancestor_depth=2, max_descendant_depth=0,
    )
    assert "↑ ancestor" in prompt
    assert "root" in prompt  # e1.name = "root"


def test_rag_prompt_includes_descendant(branch_pool):
    """prompt 包含 descendant 段。"""
    kb = KnowledgeBase(IdentityRetriever(), pool=branch_pool)
    for e in branch_pool.all():
        kb.add(e)
    prompt = build_rag_prompt(
        "momentum", "20-day", kb=kb, top_k=1,
        max_ancestor_depth=0, max_descendant_depth=2,
    )
    # e1 有 2 个 descendants
    assert "↓ descendant" in prompt


def test_rag_prompt_disable_lineage(branch_pool):
    """include_lineage=False 时, prompt 不含谱系段。"""
    kb = KnowledgeBase(IdentityRetriever(), pool=branch_pool)
    for e in branch_pool.all():
        kb.add(e)
    prompt = build_rag_prompt(
        "momentum", "20-day", kb=kb, top_k=1,
        include_lineage=False,
        max_ancestor_depth=10, max_descendant_depth=10,
    )
    assert "↑ ancestor" not in prompt
    assert "↓ descendant" not in prompt
    assert "谱系上下文" not in prompt


# ============================================================================
# 4. query_with_lineage 深度 (2)
# ============================================================================

def test_query_with_lineage_ancestors_depth(branch_pool):
    """query_with_lineage max_ancestor_depth=2 返回 2 代。"""
    kb = KnowledgeBase(IdentityRetriever(), pool=branch_pool)
    for e in branch_pool.all():
        kb.add(e)
    # query "leaf" 应该匹配 e4
    results = kb.query_with_lineage("leaf", top_k=1, max_ancestor_depth=2, max_descendant_depth=0)
    assert len(results) == 1
    ctx = results[0]
    # e4 的 ancestors: e2 (depth 1), e1 (depth 2)
    assert {a.entry_id for a in ctx["ancestors"]} == {"e2", "e1"}


def test_query_with_lineage_legacy_fields(branch_pool):
    """兼容旧字段 parents/children (depth=1)。"""
    kb = KnowledgeBase(IdentityRetriever(), pool=branch_pool)
    for e in branch_pool.all():
        kb.add(e)
    results = kb.query_with_lineage("branch_a", top_k=1, max_ancestor_depth=1, max_descendant_depth=1)
    ctx = results[0]
    # parents = depth-1 ancestors, children = depth-1 descendants
    assert {p.entry_id for p in ctx["parents"]} == {"e1"}
    assert {c.entry_id for c in ctx["children"]} == {"e4"}


# ============================================================================
# 5. EvolutionLoop 集成 (1)
# ============================================================================

def test_evolution_loop_propagates_depth_to_hypothesizer():
    """EvolutionLoop 把 max_ancestor_depth 传给 Hypothesizer。"""
    from QuantNodes.core.evolution import EvolutionLoop, EvolutionSetting
    pool = TrajectoryPool(tempfile.mkdtemp())
    settings = EvolutionSetting(enabled=True, max_rounds=0)
    loop = EvolutionLoop(
        settings, pool=pool,
        evaluate_fn=lambda c: (True, {"sharpe": 0.5}, FactorFeedback(
            factor_id=c.factor_id, factor_name=c.name, decision=True, summary="ok",
        )),
        knowledge_base=KnowledgeBase(IdentityRetriever(), pool=pool),
        rag_top_k=2,
        max_ancestor_depth=3,
        max_descendant_depth=1,
    )
    assert loop.hypothesizer.max_ancestor_depth == 3
    assert loop.hypothesizer.max_descendant_depth == 1
    assert loop.max_ancestor_depth == 3
    assert loop.max_descendant_depth == 1
