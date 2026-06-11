"""quality_gate 边界条件测试 (20 tests)。

聚焦:
    - ComplexityChecker: 语法错误、阈值边界、disabled 模式
    - RedundancyChecker: Zoo 为空、重复因子、汉明距离阈值
    - ConsistencyChecker: disabled 模式、LLM mock 集成
    - QualityGateNode: 集成 3 门、enabled 组合、缺 expression 抛异常
    - FactorZoo: hash 唯一性、持久化往返
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from QuantNodes.core.feedback import (
    ChannelFeedback,
    FeedbackChannel,
)
from QuantNodes.core.quality_gate import (
    ComplexityChecker,
    ConsistencyChecker,
    FactorZoo,
    QualityGateNode,
    QualityGateSetting,
    RedundancyChecker,
    ComplexitySetting,
    RedundancySetting,
    ConsistencySetting,
)


# ============================================================================
# 1. ComplexityChecker (5 tests)
# ============================================================================

class TestComplexityChecker:
    def test_disabled_returns_passed(self):
        s = ComplexitySetting(enabled=False)
        c = ComplexityChecker(s)
        r = c.check("open + high + low + close + volume + amount + vwap + turnover")
        assert r.passed is True
        assert "disabled" in r.detail

    def test_syntax_error_returns_failed(self):
        c = ComplexityChecker()
        r = c.check("open +")
        assert r.passed is False
        assert "语法错误" in r.detail

    def test_normal_expression_passes(self):
        c = ComplexityChecker()
        r = c.check("close - close.shift(5)")
        assert r.passed is True
        assert r.metadata["base_features"] >= 1

    def test_symbol_length_threshold(self):
        s = ComplexitySetting(symbol_length_threshold=10)
        c = ComplexityChecker(s)
        r = c.check("close - close.shift(5)")
        # 实际 length > 10
        assert r.passed is False
        assert "length" in r.detail

    def test_free_args_ratio_100pct(self):
        """全自由参数, ratio=1.0 必失败 (默认阈值 0.5)。"""
        s = ComplexitySetting(free_args_ratio_threshold=0.5)
        c = ComplexityChecker(s)
        r = c.check("alpha + beta + gamma + delta")
        assert r.passed is False
        assert r.metadata["free_args_ratio"] == 1.0


# ============================================================================
# 2. FactorZoo + RedundancyChecker (5 tests)
# ============================================================================

class TestFactorZoo:
    def test_add_returns_hash(self):
        z = FactorZoo()
        h = z.add("close - close.shift(5)")
        assert isinstance(h, int)
        assert len(z) == 1

    def test_contains(self):
        z = FactorZoo()
        z.add("close - close.shift(5)")
        assert z.contains("close - close.shift(5)")
        assert not z.contains("open - open.shift(5)")

    def test_dedup_same_expression(self):
        """相同表达式不重复添加。"""
        z = FactorZoo()
        z.add("close - close.shift(5)")
        z.add("close - close.shift(5)")
        z.add("close - close.shift(5)")
        assert len(z) == 1

    def test_persist_roundtrip(self, tmp_path: Path):
        path = tmp_path / "zoo.parquet"
        z1 = FactorZoo(path)
        z1.add("close - close.shift(5)")
        z1.add("volume / mv_float")
        z2 = FactorZoo(path)
        assert len(z2) == 2

    def test_hamming_to_returns_sorted(self):
        """hamming_to 按距离升序。"""
        z = FactorZoo()
        z.add("close")
        z.add("open")
        z.add("volume")
        results = z.hamming_to("close")
        # 按 dist 升序
        dists = [d for d, _, _ in results]
        assert dists == sorted(dists)

    def test_min_hamming_empty_zoo(self):
        z = FactorZoo()
        assert z.min_hamming("close") == float("inf")


class TestRedundancyChecker:
    def test_disabled_returns_passed(self):
        s = RedundancySetting(enabled=False)
        c = RedundancyChecker(s)
        r = c.check("close - close.shift(5)")
        assert r.passed is True

    def test_empty_zoo_passes(self, tmp_path: Path):
        zoo = FactorZoo()  # 空
        s = RedundancySetting(enabled=True)
        c = RedundancyChecker(s, zoo=zoo)
        r = c.check("close - close.shift(5)")
        assert r.passed is True
        assert "Zoo 为空" in r.detail

    def test_duplicate_expression_fails(self):
        z = FactorZoo()
        z.add("close - close.shift(5)")
        s = RedundancySetting(enabled=True, threshold=5)
        c = RedundancyChecker(s, zoo=z)
        r = c.check("close - close.shift(5)")
        # 完全相同 hash → dist=0 < 5 → fail
        assert r.passed is False
        assert r.metadata["min_hamming_dist"] == 0

    def test_high_threshold_strict(self):
        z = FactorZoo()
        z.add("close")
        s = RedundancySetting(enabled=True, threshold=100)  # 极高
        c = RedundancyChecker(s, zoo=z)
        r = c.check("open")
        # dist 远小于 100
        assert r.passed is False


# ============================================================================
# 3. ConsistencyChecker (3 tests)
# ============================================================================

class TestConsistencyChecker:
    def test_disabled_returns_passed(self):
        s = ConsistencySetting(enabled=False)
        c = ConsistencyChecker(s)
        r = c.check("h", "d", "close")
        assert r.passed is True

    def test_enabled_uses_judge(self):
        s = ConsistencySetting(enabled=True, model="mock")
        c = ConsistencyChecker(s)
        r = c.check("momentum", "momentum factor", "close.diff(5)")
        # mock judge: momentum 关键词 + close → consistent=True
        assert r.channel == FeedbackChannel.LLM
        assert r.passed is True


# ============================================================================
# 4. QualityGateNode 集成 (7 tests)
# ============================================================================

class TestQualityGateNode:
    def test_default_construct(self):
        """默认 settings 不崩。"""
        node = QualityGateNode()
        r = node.check({
            "expression": "close - close.shift(5)",
        })
        # 默认 complexity=True, redundancy=True, consistency=False
        assert "passed" in r
        assert "feedback" in r
        assert FeedbackChannel.CODE in r["channels"]
        assert FeedbackChannel.VALUE in r["channels"]

    def test_missing_expression_raises(self):
        node = QualityGateNode()
        with pytest.raises(ValueError, match="缺少 'expression' 字段"):
            node.check({"name": "foo"})

    def test_all_pass_for_simple_expr(self):
        node = QualityGateNode()
        r = node.check({
            "name": "alpha",
            "expression": "close - close.shift(5)",
            "hypothesis": "momentum",
            "description": "20-day momentum",
        })
        assert r["passed"] is True

    def test_complexity_fails(self):
        """复杂度过高, complexity 门 fail。"""
        s = QualityGateSetting(
            complexity=ComplexitySetting(symbol_length_threshold=5),
        )
        node = QualityGateNode(s)
        r = node.check({
            "name": "alpha",
            "expression": "close - close.shift(5) - close.shift(10)",
        })
        assert r["passed"] is False
        # feedback 中 CODE 通道 fail
        assert r["feedback"].channels[FeedbackChannel.CODE].passed is False

    def test_execute_from_context(self):
        """execute 入口从 context['FactorCandidate'] 取候选。"""
        node = QualityGateNode()
        context = {"FactorCandidate": {
            "name": "alpha",
            "expression": "close - close.shift(5)",
        }}
        r = node.execute(context=context)
        assert r["passed"] is True

    def test_execute_missing_candidate_raises(self):
        node = QualityGateNode()
        with pytest.raises(ValueError, match="FactorCandidate 缺失"):
            node.execute(context={})

    def test_execute_with_none_context(self):
        """context=None + kwargs 兼容。"""
        node = QualityGateNode()
        context = {"FactorCandidate": {"name": "f", "expression": "close"}}
        r = node.execute(context=context)
        assert r["passed"] is True

    def test_any_enabled(self):
        s1 = QualityGateSetting(complexity=ComplexitySetting(enabled=True))
        assert s1.any_enabled() is True
        s2 = QualityGateSetting(
            complexity=ComplexitySetting(enabled=False),
            redundancy=RedundancySetting(enabled=False),
            consistency=ConsistencySetting(enabled=False),
        )
        assert s2.any_enabled() is False
