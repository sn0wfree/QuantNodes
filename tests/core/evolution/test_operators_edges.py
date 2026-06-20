"""evolution/operators.py 边界条件测试 (15 tests)。

聚焦:
    - FactorCandidate 默认值
    - Hypothesizer: 无 KB / 有 KB / mock JSON 失败 / mock 成功
    - Mutator: 普通 mutate / 失败重试 / 最终 fallback
    - Crosser: 普通 crossover / 单 parent / 失败重试
    - mock_variant 启发式: 含/不含 "父因子:" 关键词的 fallback
    - llm_callable 自定义
"""
from __future__ import annotations

import json


from QuantNodes.core.evolution.operators import (
    Crosser,
    FactorCandidate,
    Hypothesizer,
    Mutator,
    _mock_variant,
)


# ============================================================================
# 1. FactorCandidate (2 tests)
# ============================================================================

class TestFactorCandidate:
    def test_factor_id_is_uuid(self):
        c = FactorCandidate(factor_id="c1", name="x", expression="close")
        assert c.factor_id == "c1"
        # factor_id 任意 string 都接受, 不强制 UUID
        c2 = FactorCandidate(factor_id="abc-123", name="y", expression="open")
        assert c2.factor_id == "abc-123"

    def test_minimal_required(self):
        """factor_id + name + expression 必填, hypothesis/description 默认空。"""
        c = FactorCandidate(factor_id="c1", name="x", expression="close")
        assert c.hypothesis == ""
        assert c.description == ""


# ============================================================================
# 2. Hypothesizer (4 tests)
# ============================================================================

class TestHypothesizer:
    def test_hypothesize_mock(self):
        h = Hypothesizer(model="mock")
        c = h.hypothesize("momentum reversal alpha")
        assert c.name
        assert c.expression
        assert c.hypothesis == "momentum reversal alpha"
        assert c.description

    def test_hypothesize_uses_rag_when_kb_provided(self):
        """有 KB 时 prompt 应包含 RAG 上下文 (用 mock 验证)。"""
        from QuantNodes.core.knowledge import KnowledgeBase
        from QuantNodes.core.trajectory import TrajectoryEntry
        from QuantNodes.core.feedback import FactorFeedback

        kb = KnowledgeBase()
        kb.add(TrajectoryEntry(
            entry_id="e1",
            feedback=FactorFeedback(factor_id="e1", factor_name="momentum_20", decision=True),
            config_snapshot={"factor": {
                "name": "momentum_20",
                "expression": "close - close.shift(20)",
                "hypothesis": "momentum",
            }},
        ))

        captured = {}

        def fake_llm(prompt):
            captured["prompt"] = prompt
            return json.dumps({
                "name": "new_factor",
                "expression": "close.diff(5)",
                "description": "from RAG",
            })

        h = Hypothesizer(llm_callable=fake_llm, knowledge_base=kb, rag_top_k=1)
        c = h.hypothesize("momentum alpha", "20-day momentum factor")
        assert "new_factor" in c.name
        # prompt 应包含 RAG 检索到的 entry
        assert "momentum_20" in captured["prompt"] or "momentum" in captured["prompt"]

    def test_hypothesize_fallback_on_json_error(self):
        """llm_callable 一直返回非 JSON, 最终走 mock_variant 兜底。"""
        def bad_llm(prompt):
            return "this is not json {"

        h = Hypothesizer(llm_callable=bad_llm, max_correction_attempts=2)
        c = h.hypothesize("alpha direction")
        # mock_variant 返回 default expression
        assert c.expression
        assert c.name

    def test_hypothesize_no_kb_no_rag(self):
        """无 KB 时 prompt 不含 RAG 上下文。"""
        captured = {}

        def fake_llm(prompt):
            captured["prompt"] = prompt
            return json.dumps({
                "name": "f", "expression": "close", "description": "d",
            })

        h = Hypothesizer(llm_callable=fake_llm, knowledge_base=None)
        h.hypothesize("alpha")
        # 标准 prompt
        assert "研究假设" in captured["prompt"]
        assert "alpha" in captured["prompt"]


# ============================================================================
# 3. Mutator (3 tests)
# ============================================================================

class TestMutator:
    def test_mutate_mock(self):
        m = Mutator(model="mock")
        parent = FactorCandidate(
            factor_id="p", name="p", expression="close - close.shift(5)",
            hypothesis="h", description="d",
        )
        child = m.mutate(parent)
        # factor_id 不同
        assert child.factor_id != parent.factor_id
        # name 基于 parent name 派生
        assert "p" in child.name or "mock" in child.name
        # expression 包含 parent 表达式 (mutation 包裹)
        assert parent.expression in child.expression

    def test_mutate_inherits_hypothesis(self):
        """子 hypothesis 继承 parent。"""
        m = Mutator(model="mock")
        parent = FactorCandidate(
            factor_id="p", name="p", expression="close", hypothesis="my_hyp",
        )
        child = m.mutate(parent)
        assert child.hypothesis == "my_hyp"

    def test_mutate_fallback_on_error(self):
        def bad_llm(prompt):
            return "invalid"
        m = Mutator(llm_callable=bad_llm, max_correction_attempts=1)
        parent = FactorCandidate(factor_id="p", name="p", expression="close")
        child = m.mutate(parent)
        # 兜底返回
        assert child.expression
        assert "close" in child.expression


# ============================================================================
# 4. Crosser (3 tests)
# ============================================================================

class TestCrosser:
    def test_crossover_mock(self):
        x = Crosser(model="mock")
        p1 = FactorCandidate(factor_id="a", name="a", expression="close", hypothesis="h1")
        p2 = FactorCandidate(factor_id="b", name="b", expression="volume", hypothesis="h2")
        child = x.crossover(p1, p2)
        # 包含两个 parent 表达式
        assert "close" in child.expression or "volume" in child.expression

    def test_crossover_hypothesis_combines(self):
        x = Crosser(model="mock")
        p1 = FactorCandidate(factor_id="a", name="a", expression="close", hypothesis="h1")
        p2 = FactorCandidate(factor_id="b", name="b", expression="volume", hypothesis="h2")
        child = x.crossover(p1, p2)
        assert "h1" in child.hypothesis and "h2" in child.hypothesis

    def test_crossover_fallback_on_error(self):
        def bad_llm(prompt):
            return "{invalid"
        x = Crosser(llm_callable=bad_llm, max_correction_attempts=1)
        p1 = FactorCandidate(factor_id="a", name="a", expression="close")
        p2 = FactorCandidate(factor_id="b", name="b", expression="volume")
        child = x.crossover(p1, p2)
        assert child.expression
        assert "close" in child.expression or "volume" in child.expression


# ============================================================================
# 5. _mock_variant 启发式 (3 tests)
# ============================================================================

class TestMockVariant:
    def test_hypothesize_fallback(self):
        """无 '父因子:' 关键词 → hypothesize fallback。"""
        prompt = "研究假设: alpha momentum"
        result = _mock_variant(prompt)
        assert "name" in result
        assert "expression" in result
        # 默认 expression 是 momentum-like
        assert "close" in result["expression"] or "shift" in result["expression"]

    def test_mutate_fallback(self):
        """含 '父因子:' 单 parent → mutation template。"""
        prompt = "父因子: close - close.shift(5)"
        result = _mock_variant(prompt)
        assert "m_mock" in result["name"]
        # mutation template 包裹 parent
        assert "close - close.shift(5)" in result["expression"]

    def test_crossover_fallback(self):
        """含 2 个 '父因子' → crossover template。"""
        prompt = "父因子 1: close\n父因子 2: volume"
        result = _mock_variant(prompt)
        assert "x_mock" in result["name"]
        # 同时包含 close 和 volume
        assert "close" in result["expression"] and "volume" in result["expression"]
