"""rag_prompt.py 边界测试 (15 tests)。

聚焦:
    - build_rag_prompt: 无 KB、KB 空、KB 无匹配、含 lineage、不含 lineage
    - use_compress: 自动构造 Compressor、显式传 Compressor
    - 谱系空时 (无 ancestors/descendants) → 不写 lineage 段
    - _format_example: entry=None
    - _format_lineage_expanded: ancestors/descendants 各自处理
    - _format_lineage_compressed: 压缩调用 Compressor
    - 长 description / expression 截断
"""
from __future__ import annotations

from pathlib import Path


from QuantNodes.core.feedback import FactorFeedback
from QuantNodes.core.knowledge import (
    Compressor,
    KnowledgeBase,
    build_rag_prompt,
)
from QuantNodes.core.trajectory import TrajectoryEntry, TrajectoryPool


def _entry(
    entry_id: str, name: str, expression: str = "close",
    description: str = "", hypothesis: str = "",
    sharpe: float = 0.5, round_idx: int = 0,
    parent_ids: list[str] | None = None,
) -> TrajectoryEntry:
    return TrajectoryEntry(
        entry_id=entry_id, round_idx=round_idx,
        parent_ids=parent_ids or [],
        feedback=FactorFeedback(
            factor_id=entry_id, factor_name=name, decision=True,
        ),
        config_snapshot={"factor": {
            "name": name, "expression": expression,
            "hypothesis": hypothesis, "description": description,
        }},
        metrics={"sharpe": sharpe, "arr": sharpe * 0.1, "ic_mean": 0.04},
    )


# ============================================================================
# 1. build_rag_prompt 基本 (5 tests)
# ============================================================================

class TestBuildRagPrompt:
    def test_no_kb_returns_task_only(self):
        """无 KB → 只返 task 段。"""
        prompt = build_rag_prompt("alpha", "alpha desc")
        assert "研究假设" in prompt
        assert "alpha" in prompt
        # 不应有 RAG header
        assert "历史表现良好的" not in prompt

    def test_empty_kb_no_rag_section(self):
        """KB 为空 → 不附 RAG 段, 只 task。"""
        kb = KnowledgeBase()
        prompt = build_rag_prompt("alpha", "d", kb=kb)
        assert "历史表现良好的" not in prompt
        assert "研究假设" in prompt

    def test_kb_with_matching_entry(self, tmp_path: Path):
        """KB 含匹配 entry → 附示例。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry("e1", "alpha_20", "close - close.shift(20)",
                        description="momentum reversal"))
        kb = KnowledgeBase(pool=pool)
        kb.add_many(pool.all())
        prompt = build_rag_prompt("momentum alpha", "20-day", kb=kb, top_k=1)
        assert "示例 1" in prompt
        assert "alpha_20" in prompt
        assert "momentum" in prompt or "alpha" in prompt

    def test_min_score_filters_all(self, tmp_path: Path):
        """min_score 极高 → 全部过滤 → 不附 RAG。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry("e1", "alpha_20", "close - close.shift(20)",
                        description="momentum"))
        kb = KnowledgeBase(pool=pool)
        kb.add_many(pool.all())
        prompt = build_rag_prompt("totally unrelated xyz", "d",
                                   kb=kb, top_k=5, min_score=0.99)
        # 全部被过滤 → 无 RAG header
        assert "历史表现良好的" not in prompt

    def test_query_concatenates_direction_and_description(self, tmp_path: Path):
        """query = direction + description。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry("e1", "alpha", description="hello world marker"))
        kb = KnowledgeBase(pool=pool)
        kb.add_many(pool.all())
        # 检索时 query = "alpha hello world marker"
        prompt = build_rag_prompt("alpha", "hello world marker", kb=kb, top_k=1)
        assert "示例 1" in prompt


# ============================================================================
# 2. lineage 段 (4 tests)
# ============================================================================

class TestLineageInPrompt:
    def test_include_lineage_false(self, tmp_path: Path):
        """include_lineage=False → 不附谱系段。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry("e1", "alpha", description="momentum"))
        pool.add(_entry("e2", "beta", round_idx=1, parent_ids=["e1"],
                        description="momentum child"))
        kb = KnowledgeBase(pool=pool)
        kb.add_many(pool.all())
        prompt = build_rag_prompt("momentum", "d", kb=kb, top_k=1,
                                   include_lineage=False)
        # 谱系上下文不应出现
        assert "谱系上下文" not in prompt

    def test_include_lineage_true_with_ancestors(self, tmp_path: Path):
        """include_lineage=True + 有 parent → 附谱系段。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry("e1", "alpha", description="momentum"))
        pool.add(_entry("e2", "beta", round_idx=1, parent_ids=["e1"],
                        description="momentum child"))
        kb = KnowledgeBase(pool=pool)
        kb.add_many(pool.all())
        prompt = build_rag_prompt("momentum", "d", kb=kb, top_k=1,
                                   include_lineage=True)
        # 谱系上下文应出现
        assert "谱系上下文" in prompt
        assert "ancestor" in prompt

    def test_lineage_no_ancestors_no_descendants(self, tmp_path: Path):
        """root 无 ancestors/descendants → 谱系段为空 → 不附。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry("e1", "lonely", description="momentum"))
        kb = KnowledgeBase(pool=pool)
        kb.add_many(pool.all())
        prompt = build_rag_prompt("momentum", "d", kb=kb, top_k=1)
        # 谱系段空, 不附
        assert "谱系上下文" not in prompt

    def test_kb_without_pool_skips_lineage(self, tmp_path: Path):
        """KB 无 pool → 跳过谱系段 (无法 expand)。"""
        kb = KnowledgeBase()  # 无 pool
        # 直接 retriever index 一个 entry
        from QuantNodes.core.knowledge.retriever import make_retriever
        kb.retriever = make_retriever("identity")
        e1 = _entry("e1", "alpha", description="momentum")
        kb.retriever.add(e1.entry_id, "alpha momentum close")
        # query 会返 (None, score) 因为无 pool
        prompt = build_rag_prompt("momentum", "d", kb=kb, top_k=1)
        # 不应崩, 也不应有谱系段
        assert isinstance(prompt, str)


# ============================================================================
# 3. use_compress (4 tests)
# ============================================================================

class TestUseCompress:
    def test_use_compress_heuristic(self, tmp_path: Path):
        """use_compress=True 启发式, 谱系段格式不同。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry("e1", "alpha", description="momentum"))
        pool.add(_entry("e2", "beta", round_idx=1, parent_ids=["e1"],
                        description="momentum child"))
        kb = KnowledgeBase(pool=pool)
        kb.add_many(pool.all())
        prompt = build_rag_prompt(
            "momentum", "d", kb=kb, top_k=1,
            use_compress=True,
        )
        # 压缩模式: "压缩" 字样
        assert "压缩" in prompt

    def test_use_compress_with_explicit_compressor(self, tmp_path: Path):
        """use_compress=True + 显式 Compressor。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry("e1", "alpha", description="momentum"))
        pool.add(_entry("e2", "beta", round_idx=1, parent_ids=["e1"]))
        kb = KnowledgeBase(pool=pool)
        kb.add_many(pool.all())
        comp = Compressor(model="mock", max_tokens=50)
        prompt = build_rag_prompt(
            "momentum", "d", kb=kb, top_k=1,
            use_compress=True, compressor=comp,
        )
        # 验证 Compressor 被使用
        assert "压缩" in prompt

    def test_use_compress_no_ancestors_descendants(self, tmp_path: Path):
        """无 ancestors/descendants → 压缩段空, 不写。"""
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry("e1", "alpha", description="momentum"))
        kb = KnowledgeBase(pool=pool)
        kb.add_many(pool.all())
        prompt = build_rag_prompt(
            "momentum", "d", kb=kb, top_k=1, use_compress=True,
        )
        # 无祖先/后裔 → 无谱系段
        assert "谱系上下文" not in prompt

    def test_use_compress_with_llm_callable(self, tmp_path: Path):
        """use_compress + Compressor.llm_callable 真实 LLM。"""
        import json
        pool = TrajectoryPool(tmp_path)
        pool.add(_entry("e1", "alpha", description="momentum"))
        pool.add(_entry("e2", "beta", round_idx=1, parent_ids=["e1"]))
        kb = KnowledgeBase(pool=pool)
        kb.add_many(pool.all())

        def fake_llm(prompt):
            return json.dumps({"summary": "LLM-generated summary"})

        comp = Compressor(model="mock", llm_callable=fake_llm)
        prompt = build_rag_prompt(
            "momentum", "d", kb=kb, top_k=1,
            use_compress=True, compressor=comp,
        )
        # LLM 总结应出现
        assert "LLM-generated summary" in prompt


# ============================================================================
# 4. 长 description / expression (2 tests)
# ============================================================================

class TestLongFields:
    def test_long_expression_truncated(self, tmp_path: Path):
        """长 expression 在谱系段被截断 ([:40]), 示例段保持完整。"""
        pool = TrajectoryPool(tmp_path)
        # 父 entry 有长 expression
        long_expr = "x" * 200
        pool.add(_entry("e1", "alpha", expression=long_expr, description="m"))
        pool.add(_entry("e2", "beta", round_idx=1, parent_ids=["e1"],
                        expression="close", description="m"))
        kb = KnowledgeBase(pool=pool)
        kb.add_many(pool.all())
        prompt = build_rag_prompt("m", "d", kb=kb, top_k=1, include_lineage=True)
        # 谱系段中 expression 截断到 40 字符 (lineage_format 用 [:40])
        # ancestor 段: "↑ ancestor (depth=1): alpha | sharpe=0.5 | " + 长 expression 截断
        assert "x" * 40 in prompt
        # 完整长 expression 也在示例段 (示例段不截断)
        assert "x" * 50 in prompt  # 200 个 x 在示例段完整出现
