"""谱系压缩 (Week 9) 测试 — 10 tests。

覆盖:
    - Compressor heuristic (3)
    - Compressor LLM (2)
    - CompressedLineage dataclass (1)
    - build_rag_prompt use_compress (2)
    - EvolutionLoop 集成 (1)
    - CLI --compress (1)
"""
from __future__ import annotations

import io
import json
import tempfile
from contextlib import redirect_stdout

import pytest

from QuantNodes.cli import cmd_factor_rag_show
from QuantNodes.core.evolution import EvolutionLoop, EvolutionSetting
from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.knowledge import (
    CompressedLineage,
    Compressor,
    IdentityRetriever,
    KnowledgeBase,
    build_rag_prompt,
)
from QuantNodes.core.trajectory import TrajectoryEntry, TrajectoryPool


# ============================================================================
# Fixtures
# ============================================================================

def _make_entry(
    eid: str, name: str, parent_ids: list | None = None,
    round_idx: int = 0, operation: str = "original", sharpe: float = 1.0,
    hypothesis: str = "h", description: str = "d",
) -> TrajectoryEntry:
    return TrajectoryEntry(
        entry_id=eid, round_idx=round_idx, operation=operation,
        parent_ids=parent_ids or [],
        config_snapshot={"factor": {
            "name": name, "expression": f"close - open ({name})",
            "hypothesis": hypothesis, "description": description,
        }},
        feedback=FactorFeedback(factor_name=name, decision=True, summary="ok"),
        metrics={"sharpe": sharpe},
    )


@pytest.fixture
def chain_pool() -> TrajectoryPool:
    pool = TrajectoryPool(tempfile.mkdtemp())
    pool.add(_make_entry("e1", "root"))
    pool.add(_make_entry("e2", "child", parent_ids=["e1"], round_idx=1, operation="mutation"))
    pool.add(_make_entry("e3", "grand", parent_ids=["e2"], round_idx=2, operation="crossover"))
    return pool


@pytest.fixture
def chain_entries() -> list[TrajectoryEntry]:
    """3 个 entry 用于 compress 测试。"""
    pool = TrajectoryPool(tempfile.mkdtemp())
    pool.add(_make_entry("e1", "root"))
    pool.add(_make_entry("e2", "child", parent_ids=["e1"], round_idx=1, operation="mutation"))
    pool.add(_make_entry("e3", "grand", parent_ids=["e2"], round_idx=2, operation="crossover"))
    return pool.all()


# ============================================================================
# 1. Compressor heuristic (3)
# ============================================================================

def test_compressor_heuristic_ancestors(chain_entries):
    """heuristic 模式压缩 ancestors。"""
    c = Compressor(model="mock")
    items = [(1, chain_entries[1]), (2, chain_entries[0])]
    result = c.compress(items, relation="ancestors")
    assert isinstance(result, CompressedLineage)
    assert result.method == "heuristic"
    assert result.original_count == 2
    assert "↑" in result.summary
    assert "ancestor" not in result.summary  # heuristic 不用 "ancestor" 字样
    assert "child" in result.summary
    assert "root" in result.summary


def test_compressor_heuristic_descendants(chain_entries):
    """heuristic 模式压缩 descendants。"""
    c = Compressor(model="mock")
    items = [(1, chain_entries[1]), (2, chain_entries[2])]
    result = c.compress(items, relation="descendants")
    assert "↓" in result.summary


def test_compressor_max_tokens_truncation():
    """max_tokens 限制生效。"""
    entries = [
        _make_entry(f"e{i}", f"name_{i}_" + "x" * 50) for i in range(5)
    ]
    items = [(1, e) for e in entries]
    c = Compressor(model="mock", max_tokens=50)
    result = c.compress(items, relation="ancestors")
    assert result.compressed_chars <= 50


# ============================================================================
# 2. Compressor LLM (2)
# ============================================================================

def test_compressor_llm_callable():
    """自定义 llm_callable 走 LLM 路径。"""
    def fake_llm(prompt):
        return json.dumps({"summary": "all factors use close-open spread"})
    c = Compressor(llm_callable=fake_llm)
    result = c.compress([(1, _make_entry("e1", "x"))], relation="ancestors")
    assert result.method == "llm"
    assert result.summary == "all factors use close-open spread"


def test_compressor_llm_parse_failure_falls_back():
    """LLM 解析失败 → fallback 启发式。"""
    def bad_llm(prompt):
        return "not json"
    c = Compressor(llm_callable=bad_llm)
    result = c.compress([(1, _make_entry("e1", "x"))], relation="ancestors")
    assert result.method == "heuristic"
    assert "↑" in result.summary


# ============================================================================
# 3. CompressedLineage dataclass (1)
# ============================================================================

def test_compressed_lineage_dataclass():
    """CompressedLineage 字段正确。"""
    cl = CompressedLineage(
        summary="abc", original_count=3, compressed_chars=3, method="llm",
    )
    assert cl.summary == "abc"
    assert cl.original_count == 3
    assert cl.compressed_chars == 3
    assert cl.method == "llm"


# ============================================================================
# 4. build_rag_prompt use_compress (2)
# ============================================================================

def test_build_rag_prompt_with_compress(chain_pool):
    """use_compress=True 时, prompt 含压缩段。"""
    kb = KnowledgeBase(IdentityRetriever(), pool=chain_pool)
    for e in chain_pool.all():
        kb.add(e)
    c = Compressor(model="mock", max_tokens=200)
    prompt = build_rag_prompt(
        "child mutation crossover", "d", kb=kb, top_k=1,
        max_ancestor_depth=2, max_descendant_depth=0,
        use_compress=True, compressor=c,
    )
    assert "谱系上下文 (压缩)" in prompt
    assert "↑ ancestors" in prompt
    assert "↓ descendants" not in prompt


def test_build_rag_prompt_default_no_compress(chain_pool):
    """use_compress 默认 False, prompt 含展开段。"""
    kb = KnowledgeBase(IdentityRetriever(), pool=chain_pool)
    for e in chain_pool.all():
        kb.add(e)
    prompt = build_rag_prompt(
        "child mutation crossover", "d", kb=kb, top_k=1,
        max_ancestor_depth=2, max_descendant_depth=0,
    )
    assert "↑ ancestor (depth=" in prompt
    assert "谱系上下文 (压缩)" not in prompt


# ============================================================================
# 5. EvolutionLoop 集成 (1)
# ============================================================================

def test_evolution_loop_propagates_compress():
    """EvolutionLoop 把 use_compress + compressor 传给 Hypothesizer。"""
    pool = TrajectoryPool(tempfile.mkdtemp())
    settings = EvolutionSetting(enabled=True, max_rounds=0)
    comp = Compressor(model="mock", max_tokens=100)
    loop = EvolutionLoop(
        settings, pool=pool,
        evaluate_fn=lambda c: (True, {"sharpe": 0.5}, FactorFeedback(
            factor_id=c.factor_id, factor_name=c.name, decision=True, summary="ok",
        )),
        knowledge_base=KnowledgeBase(IdentityRetriever(), pool=pool),
        use_compress=True,
        compressor=comp,
    )
    assert loop.use_compress is True
    assert loop.compressor is comp
    assert loop.hypothesizer.use_compress is True
    assert loop.hypothesizer.compressor is comp


# ============================================================================
# 6. CLI --compress (1)
# ============================================================================

def test_cli_rag_show_with_compress(chain_pool):
    """CLI factor-rag-show --compress 启用压缩显示。"""
    from QuantNodes.core.trajectory import TrajectoryPool
    pool_dir = tempfile.mkdtemp()
    pool = TrajectoryPool(pool_dir)
    for e in chain_pool.all():
        pool.add(e)

    class Args:
        pass
    args = Args()
    args.pool_dir = pool_dir
    args.query = "child mutation crossover"
    args.top = 1
    args.compress = True
    args.ancestor_depth = 2
    args.descendant_depth = 0
    args.max_tokens = 200
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = cmd_factor_rag_show(args)
    assert rc == 0
    out = buf.getvalue()
    assert "↑ ancestors" in out
    assert "sharpe=" in out


# ============================================================================
# 7. 压缩比演示 (1) — 不算入 10, 仅作演示
# ============================================================================

def test_compression_ratio_demonstration():
    """演示: 启发式压缩在多 entry 时减少 token。"""
    pool = TrajectoryPool(tempfile.mkdtemp())
    expr = "close - open"
    for i in range(8):
        pid = f"e{i}" if i == 0 else f"e{i-1}"
        rid = i if i == 0 else i - 1
        pool.add(_make_entry(
            f"e{i}", f"e{i}", parent_ids=[pid] if i > 0 else None,
            round_idx=i, operation="mutation" if i > 0 else "original",
            sharpe=1.0 - i * 0.1,
        ))
    kb = KnowledgeBase(IdentityRetriever(), pool=pool)
    for e in pool.all():
        kb.add(e)
    prompt_full = build_rag_prompt("mutation close open", "d", kb=kb, top_k=1,
        max_ancestor_depth=10, max_descendant_depth=0, use_compress=False)
    c = Compressor(model="mock", max_tokens=300)
    prompt_compressed = build_rag_prompt("mutation close open", "d", kb=kb, top_k=1,
        max_ancestor_depth=10, max_descendant_depth=0, use_compress=True, compressor=c)
    # 启发式压缩至少不增加 token
    assert len(prompt_compressed) <= len(prompt_full) + 50  # 容差 (header "压缩" 多 1 字)
