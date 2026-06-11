"""QualityGate 模块测试 (25 tests)。

覆盖:
    - ComplexityChecker (6)
    - FactorZoo (4)
    - RedundancyChecker (4)
    - ConsistencyChecker (4)
    - QualityGateNode (5)
    - Settings (2)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from QuantNodes.core.feedback import (
    ChannelFeedback,
    FactorFeedback,
    FeedbackChannel,
    LLMJudge,
)
from QuantNodes.core.quality_gate import (
    ComplexityChecker,
    ComplexitySetting,
    ConsistencyChecker,
    ConsistencySetting,
    FactorZoo,
    QualityGateNode,
    QualityGateSetting,
    RedundancyChecker,
    RedundancySetting,
    ast_hash,
)


# ============================================================================
# 1. ComplexityChecker (6)
# ============================================================================

def test_complexity_simple_passes():
    """简单表达式通过。"""
    fb = ComplexityChecker().check("close - open")
    assert fb.passed is True
    assert "OK" in fb.detail
    assert fb.channel == FeedbackChannel.CODE


def test_complexity_long_fails():
    """长度超限失败。"""
    checker = ComplexityChecker(ComplexitySetting(symbol_length_threshold=50))
    long_expr = "close + " + " + ".join(f"x_{i}" for i in range(20))
    fb = checker.check(long_expr)
    assert fb.passed is False
    assert "length=" in fb.detail


def test_complexity_many_features_fails():
    """特征过多失败 (>5 base features)。"""
    expr = "open + high + low + close + volume + vwap + turnover + mv_float"
    fb = ComplexityChecker().check(expr)
    assert fb.passed is False
    assert "features=" in fb.detail


def test_complexity_high_free_args_fails():
    """自由参数过多失败。"""
    expr = "a + b + c + d"  # 全是 free args
    fb = ComplexityChecker().check(expr)
    assert fb.passed is False
    assert "free_args=" in fb.detail


def test_complexity_syntax_error_fails():
    """语法错误失败。"""
    fb = ComplexityChecker().check("close - )")
    assert fb.passed is False
    assert "语法错误" in fb.detail


def test_complexity_disabled_skips():
    """disabled=True 时直接返回 passed。"""
    fb = ComplexityChecker(ComplexitySetting(enabled=False)).check("a + b + c")
    assert fb.passed is True
    assert "disabled" in fb.detail


# ============================================================================
# 2. FactorZoo (4)
# ============================================================================

def test_zoo_empty():
    """空 Zoo 行为。"""
    zoo = FactorZoo()
    assert len(zoo) == 0
    assert zoo.min_hamming("close") == float("inf")
    assert not zoo.contains("close")


def test_zoo_add_and_contains():
    """添加 + contains。"""
    zoo = FactorZoo()
    h = zoo.add("close - open")
    assert len(zoo) == 1
    assert zoo.contains("close - open")
    assert isinstance(h, int)


def test_zoo_save_load(tmp_path):
    """持久化 + 重载。"""
    path = tmp_path / "zoo.parquet"
    zoo = FactorZoo(path)
    zoo.add("close - open")
    zoo.add("(close - close.shift(20)) / close.shift(20)")

    zoo2 = FactorZoo(path)
    assert len(zoo2) == 2
    assert zoo2.contains("(close - close.shift(20)) / close.shift(20)")


def test_zoo_hash_invariant():
    """hash 稳定性: 相同 AST 结构, 不同变量名 → 不同 hash。"""
    h1 = ast_hash("close - open")
    h2 = ast_hash("high - low")
    h3 = ast_hash("CLOSE - OPEN")  # 大小写不同, 结构相同, hash 不同
    assert h1 != h2
    # 大小写不影响 AST (Name.id 保留大小写)
    assert h1 != h3


# ============================================================================
# 3. RedundancyChecker (4)
# ============================================================================

def test_redundancy_empty_zoo_passes():
    """空 Zoo 通过。"""
    checker = RedundancyChecker()
    fb = checker.check("close - open")
    assert fb.passed is True
    assert "Zoo 为空" in fb.detail


def test_redundancy_identical_fails():
    """相同表达式失败 (hamming=0 < threshold=5)。"""
    zoo = FactorZoo()
    zoo.add("close - open")
    checker = RedundancyChecker(RedundancySetting(threshold=5), zoo=zoo)
    fb = checker.check("close - open")
    assert fb.passed is False
    assert "min_hamming_dist=0" in fb.detail


def test_redundancy_similar_passes():
    """相似表达式通过 (hamming > 5)。"""
    zoo = FactorZoo()
    zoo.add("close - open")
    checker = RedundancyChecker(RedundancySetting(threshold=5), zoo=zoo)
    # 完全不同结构
    fb = checker.check("(close - close.shift(20)) / close.shift(20)")
    assert fb.passed is True
    assert "min_hamming_dist=" in fb.detail


def test_redundancy_distance_threshold():
    """阈值边界测试。"""
    zoo = FactorZoo()
    zoo.add("close - open")
    # threshold=0: 任何因子都通过
    checker_low = RedundancyChecker(RedundancySetting(threshold=0), zoo=zoo)
    assert checker_low.check("close - open").passed is True
    # threshold=1: 相同 hash (hamming=0) 失败
    checker_high = RedundancyChecker(RedundancySetting(threshold=1), zoo=zoo)
    assert checker_high.check("close - open").passed is False


# ============================================================================
# 4. ConsistencyChecker (4)
# ============================================================================

def test_consistency_disabled_skips():
    """disabled=True 时直接通过。"""
    fb = ConsistencyChecker(ConsistencySetting(enabled=False)).check("h", "d", "e")
    assert fb.passed is True
    assert "disabled" in fb.detail


def test_consistency_passes():
    """关键词匹配 + 表达式含 returns → 一致。"""
    judge = LLMJudge(model="mock")
    checker = ConsistencyChecker(ConsistencySetting(enabled=True, model="mock"), judge=judge)
    fb = checker.check("momentum effect", "20-day momentum", "close / close.shift(20) - 1")
    assert fb.passed is True


def test_consistency_fails():
    """hypothesis + description 都为空 → 不一致。"""
    judge = LLMJudge(model="mock")
    checker = ConsistencyChecker(ConsistencySetting(enabled=True, model="mock"), judge=judge)
    fb = checker.check("", "", "close - open")
    assert fb.passed is False


def test_consistency_custom_callable():
    """支持自定义 llm_callable。"""
    def fake_llm(prompt):
        return json.dumps({"consistent": True, "reason": "fake", "score": 0.9})
    judge = LLMJudge(llm_callable=fake_llm)
    checker = ConsistencyChecker(ConsistencySetting(enabled=True), judge=judge)
    fb = checker.check("h", "d", "e")
    assert fb.passed is True
    assert "fake" in fb.detail


# ============================================================================
# 5. QualityGateNode (5)
# ============================================================================

def test_node_all_disabled_passes():
    """3 门全关 → 默认通过。"""
    settings = QualityGateSetting(
        complexity=ComplexitySetting(enabled=False),
        redundancy=RedundancySetting(enabled=False),
        consistency=ConsistencySetting(enabled=False),
    )
    node = QualityGateNode(settings)
    result = node.check({
        "expression": "anything goes",
        "hypothesis": "h",
        "description": "d",
    })
    assert result["passed"] is True
    assert result["channels"] == {}


def test_node_complexity_only():
    """仅 complexity 启用。"""
    settings = QualityGateSetting(
        complexity=ComplexitySetting(enabled=True, symbol_length_threshold=20),
        redundancy=RedundancySetting(enabled=False),
        consistency=ConsistencySetting(enabled=False),
    )
    node = QualityGateNode(settings)
    result = node.check({"expression": "x" * 100})
    assert result["passed"] is False
    assert FeedbackChannel.CODE in result["channels"]


def test_node_redundancy_only():
    """仅 redundancy 启用。"""
    settings = QualityGateSetting(
        complexity=ComplexitySetting(enabled=False),
        redundancy=RedundancySetting(enabled=True, threshold=5),
        consistency=ConsistencySetting(enabled=False),
    )
    node = QualityGateNode(settings)
    result = node.check({"expression": "close - open"})
    assert FeedbackChannel.VALUE in result["channels"]


def test_node_consistency_only():
    """仅 consistency 启用。"""
    settings = QualityGateSetting(
        complexity=ComplexitySetting(enabled=False),
        redundancy=RedundancySetting(enabled=False),
        consistency=ConsistencySetting(enabled=True, model="mock"),
    )
    node = QualityGateNode(settings)
    result = node.check({
        "expression": "close",
        "hypothesis": "hypothesis text",
        "description": "description text",
    })
    assert FeedbackChannel.LLM in result["channels"]


def test_node_all_enabled_passes():
    """3 门全开 + 简单因子 → 全部通过。"""
    settings = QualityGateSetting(
        complexity=ComplexitySetting(enabled=True),
        redundancy=RedundancySetting(enabled=True, threshold=5),
        consistency=ConsistencySetting(enabled=True, model="mock"),
    )
    node = QualityGateNode(settings)
    result = node.check({
        "factor_id": "t1",
        "name": "momentum",
        "expression": "(close - close.shift(20)) / close.shift(20)",
        "hypothesis": "momentum effect",
        "description": "20-day momentum",
    })
    assert result["passed"] is True
    assert len(result["channels"]) == 3
    assert isinstance(result["feedback"], FactorFeedback)


def test_node_execute_from_context():
    """execute() 从 context['FactorCandidate'] 读取。"""
    node = QualityGateNode()
    result = node.execute(context={
        "FactorCandidate": {
            "expression": "close - open",
        }
    })
    assert "passed" in result


def test_node_execute_missing_candidate_raises():
    """缺 FactorCandidate 抛 ValueError。"""
    node = QualityGateNode()
    with pytest.raises(ValueError, match="FactorCandidate 缺失"):
        node.execute(context={})


def test_node_missing_expression_raises():
    """candidate 缺 expression 抛 ValueError。"""
    node = QualityGateNode()
    with pytest.raises(ValueError, match="expression"):
        node.check({"hypothesis": "h"})


# ============================================================================
# 6. Settings (2)
# ============================================================================

def test_quality_gate_setting_any_enabled():
    """any_enabled() 正确判断。"""
    s1 = QualityGateSetting()  # default: complexity+redundancy on
    assert s1.any_enabled() is True

    s2 = QualityGateSetting(
        complexity=ComplexitySetting(enabled=False),
        redundancy=RedundancySetting(enabled=False),
        consistency=ConsistencySetting(enabled=False),
    )
    assert s2.any_enabled() is False


def test_settings_yaml_roundtrip():
    """Settings 可序列化往返。"""
    settings = QualityGateSetting(
        complexity=ComplexitySetting(
            enabled=True, symbol_length_threshold=150, base_features_threshold=4,
        ),
        redundancy=RedundancySetting(enabled=True, threshold=10, zoo_path="/tmp/zoo"),
        consistency=ConsistencySetting(enabled=True, model="deepseek-v3"),
    )
    d = settings.model_dump() if hasattr(settings, "model_dump") else settings.dict()
    s2 = QualityGateSetting(**d)
    assert s2.complexity.symbol_length_threshold == 150
    assert s2.redundancy.threshold == 10
    assert s2.consistency.model == "deepseek-v3"
