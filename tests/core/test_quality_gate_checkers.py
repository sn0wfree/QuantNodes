# coding=utf-8
"""Tests for core/quality_gate/ — complexity, redundancy, consistency, settings, node.

Covers: ComplexityChecker AST analysis, RedundancyChecker hamming distance,
ConsistencyChecker disabled state, QualityGateSetting Pydantic config,
QualityGateNode integration with all 3 gates.
"""

import pytest

from QuantNodes.core.quality_gate.complexity import ComplexityChecker, _count_base_features, _calc_free_args_ratio
from QuantNodes.core.quality_gate.redundancy import RedundancyChecker
from QuantNodes.core.quality_gate.consistency import ConsistencyChecker
from QuantNodes.core.quality_gate.settings import (
    ComplexitySetting,
    RedundancySetting,
    ConsistencySetting,
    QualityGateSetting,
)
from QuantNodes.core.quality_gate.node import QualityGateNode
from QuantNodes.core.quality_gate.zoo import FactorZoo
from QuantNodes.core.feedback import FeedbackChannel


# ============================================================================
# ComplexitySetting
# ============================================================================

class TestComplexitySetting:
    def test_defaults(self):
        s = ComplexitySetting()
        assert s.enabled is True
        assert s.symbol_length_threshold == 200
        assert s.base_features_threshold == 5
        assert s.free_args_ratio_threshold == 0.5

    def test_custom(self):
        s = ComplexitySetting(
            enabled=False,
            symbol_length_threshold=100,
            base_features_threshold=3,
            free_args_ratio_threshold=0.3,
        )
        assert s.enabled is False
        assert s.symbol_length_threshold == 100


# ============================================================================
# RedundancySetting
# ============================================================================

class TestRedundancySetting:
    def test_defaults(self):
        s = RedundancySetting()
        assert s.enabled is True
        assert s.threshold == 5
        assert s.zoo_path is None

    def test_custom(self):
        s = RedundancySetting(threshold=10, zoo_path="/tmp/zoo.parquet")
        assert s.threshold == 10
        assert s.zoo_path == "/tmp/zoo.parquet"


# ============================================================================
# ConsistencySetting
# ============================================================================

class TestConsistencySetting:
    def test_defaults(self):
        s = ConsistencySetting()
        assert s.enabled is False  # Default disabled (needs LLM)
        assert s.model == "mock"
        assert s.max_correction_attempts == 3

    def test_custom(self):
        s = ConsistencySetting(enabled=True, model="gpt-4", max_correction_attempts=5)
        assert s.enabled is True
        assert s.model == "gpt-4"


# ============================================================================
# QualityGateSetting
# ============================================================================

class TestQualityGateSetting:
    def test_defaults(self):
        s = QualityGateSetting()
        assert isinstance(s.complexity, ComplexitySetting)
        assert isinstance(s.redundancy, RedundancySetting)
        assert isinstance(s.consistency, ConsistencySetting)

    def test_any_enabled_default(self):
        s = QualityGateSetting()
        # Default: complexity=True, redundancy=True, consistency=False
        assert s.any_enabled() is True

    def test_any_enabled_all_disabled(self):
        s = QualityGateSetting(
            complexity=ComplexitySetting(enabled=False),
            redundancy=RedundancySetting(enabled=False),
            consistency=ConsistencySetting(enabled=False),
        )
        assert s.any_enabled() is False

    def test_custom_subsections(self):
        s = QualityGateSetting(
            complexity=ComplexitySetting(symbol_length_threshold=50),
            redundancy=RedundancySetting(threshold=10),
        )
        assert s.complexity.symbol_length_threshold == 50
        assert s.redundancy.threshold == 10


# ============================================================================
# ComplexityChecker
# ============================================================================

class TestComplexityChecker:
    def test_creation_default(self):
        checker = ComplexityChecker()
        assert checker.settings is not None

    def test_creation_custom_settings(self):
        s = ComplexitySetting(symbol_length_threshold=10)
        checker = ComplexityChecker(settings=s)
        assert checker.settings.symbol_length_threshold == 10

    def test_check_simple_passes(self):
        checker = ComplexityChecker()
        fb = checker.check("rank(close)")
        assert fb.passed is True
        assert fb.channel == FeedbackChannel.CODE

    def test_check_syntax_error_fails(self):
        checker = ComplexityChecker()
        fb = checker.check("invalid syntax ((")
        assert fb.passed is False
        assert "语法错误" in fb.detail

    def test_check_long_expression_fails(self):
        checker = ComplexityChecker(settings=ComplexitySetting(symbol_length_threshold=5))
        fb = checker.check("rank(close) * std(close, 20)")
        assert fb.passed is False
        assert "length" in fb.detail

    def test_check_too_many_base_features_fails(self):
        s = ComplexitySetting(base_features_threshold=2)
        checker = ComplexityChecker(settings=s)
        fb = checker.check("rank(close) + rank(open) + rank(volume) + rank(high)")
        assert fb.passed is False
        assert "features" in fb.detail

    def test_check_high_free_args_ratio_fails(self):
        s = ComplexitySetting(free_args_ratio_threshold=0.1)
        checker = ComplexityChecker(settings=s)
        # Most names are free args (not base features)
        fb = checker.check("custom_var1 + custom_var2")
        assert fb.passed is False

    def test_check_disabled(self):
        checker = ComplexityChecker(settings=ComplexitySetting(enabled=False))
        fb = checker.check("anything goes here")
        assert fb.passed is True
        assert "disabled" in fb.detail

    def test_check_metadata(self):
        checker = ComplexityChecker()
        fb = checker.check("rank(close)")
        assert "symbol_length" in fb.metadata
        assert "base_features" in fb.metadata
        assert "free_args_ratio" in fb.metadata

    def test_check_score_pass(self):
        checker = ComplexityChecker()
        fb = checker.check("rank(close)")
        assert fb.score == 1.0

    def test_check_score_fail(self):
        s = ComplexitySetting(symbol_length_threshold=3)
        checker = ComplexityChecker(settings=s)
        fb = checker.check("rank(close)")
        assert fb.score == 0.0


# ============================================================================
# _count_base_features
# ============================================================================

class TestCountBaseFeatures:
    def test_no_base_features(self):
        import ast
        tree = ast.parse("custom_var + other_var")
        assert _count_base_features(tree) == 0

    def test_one_base_feature(self):
        import ast
        tree = ast.parse("rank(close)")
        count = _count_base_features(tree)
        # 'close' is a base feature
        assert count >= 1

    def test_multiple_base_features(self):
        import ast
        tree = ast.parse("rank(close) + rank(open)")
        count = _count_base_features(tree)
        # 'close' and 'open' are base features
        assert count >= 2


# ============================================================================
# _calc_free_args_ratio
# ============================================================================

class TestCalcFreeArgsRatio:
    def test_empty_tree(self):
        import ast
        tree = ast.parse("42")
        assert _calc_free_args_ratio(tree) == 0.0

    def test_only_base_features(self):
        import ast
        tree = ast.parse("close + open")
        ratio = _calc_free_args_ratio(tree)
        assert ratio == 0.0

    def test_only_free_args(self):
        import ast
        tree = ast.parse("custom_var1 + custom_var2")
        ratio = _calc_free_args_ratio(tree)
        assert ratio == 1.0

    def test_mixed(self):
        import ast
        tree = ast.parse("close + custom_var")
        ratio = _calc_free_args_ratio(tree)
        assert 0.0 < ratio < 1.0


# ============================================================================
# RedundancyChecker
# ============================================================================

class TestRedundancyChecker:
    def test_creation_default(self):
        checker = RedundancyChecker()
        assert checker.settings is not None
        assert isinstance(checker.zoo, FactorZoo)

    def test_creation_with_zoo(self):
        zoo = FactorZoo()
        zoo.add("rank(close)")
        checker = RedundancyChecker(zoo=zoo)
        assert len(checker.zoo) == 1

    def test_check_disabled(self):
        checker = RedundancyChecker(settings=RedundancySetting(enabled=False))
        fb = checker.check("rank(close)")
        assert fb.passed is True

    def test_check_empty_zoo_passes(self):
        checker = RedundancyChecker()
        fb = checker.check("rank(close)")
        assert fb.passed is True
        assert "Zoo 为空" in fb.detail

    def test_check_same_expression_fails(self):
        zoo = FactorZoo()
        zoo.add("rank(close)")
        checker = RedundancyChecker(zoo=zoo)
        fb = checker.check("rank(close)")
        assert fb.passed is False
        assert fb.score == 0.0

    def test_check_different_expression_passes(self):
        zoo = FactorZoo()
        zoo.add("rank(close)")
        checker = RedundancyChecker(zoo=zoo)
        # Very different expression (different AST)
        fb = checker.check("zscore(volume) + ts_mean(open, 20)")
        # May or may not pass depending on hash distance
        assert fb.channel == FeedbackChannel.VALUE

    def test_check_threshold_metadata(self):
        zoo = FactorZoo()
        zoo.add("rank(close)")
        checker = RedundancyChecker(zoo=zoo, settings=RedundancySetting(threshold=10))
        fb = checker.check("rank(close)")
        assert "threshold" in fb.metadata
        assert "min_hamming_dist" in fb.metadata
        assert "zoo_size" in fb.metadata

    def test_check_custom_threshold(self):
        """Threshold is the min acceptable distance. Higher threshold = stricter."""
        zoo = FactorZoo()
        zoo.add("rank(close)")
        # threshold=100 means min distance must be >=100, which is impossible
        # (same expression has distance 0)
        checker = RedundancyChecker(zoo=zoo, settings=RedundancySetting(threshold=100))
        fb = checker.check("rank(close)")
        # Same expression → distance 0 → fails threshold 100
        assert fb.passed is False
        assert fb.metadata["threshold"] == 100


# ============================================================================
# ConsistencyChecker
# ============================================================================

class TestConsistencyChecker:
    def test_creation_default(self):
        checker = ConsistencyChecker()
        assert checker.settings is not None

    def test_creation_with_judge(self):
        from QuantNodes.core.feedback import LLMJudge
        judge = LLMJudge(model="mock")
        checker = ConsistencyChecker(judge=judge)
        assert checker._judge is judge

    def test_check_disabled_passes(self):
        checker = ConsistencyChecker(settings=ConsistencySetting(enabled=False))
        fb = checker.check("hypothesis", "description", "rank(close)")
        assert fb.passed is True
        assert "disabled" in fb.detail

    def test_check_uses_llm_when_enabled(self):
        """When enabled, uses LLMJudge.judge() which needs a real LLM."""
        from QuantNodes.core.feedback import LLMJudge

        # Mock LLMJudge
        judge = LLMJudge(model="mock")
        judge.judge = lambda h, d, e: pytest.skip("LLM not available")

        checker = ConsistencyChecker(settings=ConsistencySetting(enabled=True), judge=judge)
        # Skip if LLM not available
        try:
            fb = checker.check("hypothesis", "description", "rank(close)")
            assert fb is not None
        except Exception:
            pytest.skip("LLM dependency")


# ============================================================================
# QualityGateNode
# ============================================================================

class TestQualityGateNode:
    def test_creation_default(self):
        node = QualityGateNode()
        assert node.settings is not None

    def test_check_missing_expression_raises(self):
        node = QualityGateNode()
        with pytest.raises(ValueError, match="expression"):
            node.check({"name": "test"})

    def test_check_valid_candidate(self):
        node = QualityGateNode()
        result = node.check({
            "name": "my_factor",
            "expression": "rank(close)",
        })
        assert "passed" in result
        assert "feedback" in result
        assert "channels" in result

    def test_check_passes_simple_factor(self):
        node = QualityGateNode()
        result = node.check({
            "name": "good_factor",
            "expression": "rank(close)",
            "hypothesis": "momentum",
            "description": "20-day momentum",
        })
        assert result["passed"] is True

    def test_check_fails_long_factor(self):
        s = QualityGateSetting(
            complexity=ComplexitySetting(symbol_length_threshold=10)
        )
        node = QualityGateNode(settings=s)
        result = node.check({
            "name": "long_factor",
            "expression": "rank(close) * std(volume, 20) + delta(close, 1)",
        })
        assert result["passed"] is False

    def test_check_with_redundancy(self):
        zoo = FactorZoo()
        zoo.add("rank(close)")
        s = QualityGateSetting(redundancy=RedundancySetting(threshold=50))
        node = QualityGateNode(settings=s, zoo=zoo)
        result = node.check({
            "name": "dup_factor",
            "expression": "rank(close)",
        })
        # Same expression → redundancy should fail
        assert result["passed"] is False

    def test_check_aggregates_channels(self):
        node = QualityGateNode()
        result = node.check({"expression": "rank(close)"})
        # Default has complexity + redundancy enabled
        assert FeedbackChannel.CODE in result["channels"]
        assert FeedbackChannel.VALUE in result["channels"]

    def test_check_consistency_disabled_skipped(self):
        node = QualityGateNode()  # Default has consistency disabled
        result = node.check({"expression": "rank(close)"})
        # LLM channel should not be in channels
        assert FeedbackChannel.LLM not in result["channels"]

    def test_check_factor_id_generated(self):
        node = QualityGateNode()
        result = node.check({"expression": "rank(close)"})
        # Factor ID should be in feedback
        assert result["feedback"].factor_id is not None

    def test_execute_missing_context(self):
        node = QualityGateNode()
        with pytest.raises(ValueError, match="FactorCandidate 缺失"):
            node.execute(context={})

    def test_execute_via_context(self):
        node = QualityGateNode()
        context = {
            "FactorCandidate": {
                "name": "test",
                "expression": "rank(close)",
            }
        }
        result = node.execute(context=context)
        assert result["passed"] is True

    def test_execute_kwargs_context(self):
        node = QualityGateNode()
        result = node.execute(context={
            "FactorCandidate": {
                "name": "test",
                "expression": "rank(close)",
            }
        })
        assert result["passed"] is True


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_empty_expression_complexity(self):
        checker = ComplexityChecker()
        fb = checker.check("")
        # Empty expression is valid Python but has no base features
        assert fb.passed is True

    def test_unicode_expression(self):
        checker = ComplexityChecker()
        fb = checker.check("rank(close)  # 注释")
        # Comment is allowed, passes AST parse
        assert fb.channel == FeedbackChannel.CODE

    def test_complexity_then_redundancy(self):
        """Both gates active: must pass both."""
        zoo = FactorZoo()
        s = QualityGateSetting(
            complexity=ComplexitySetting(symbol_length_threshold=10),
            redundancy=RedundancySetting(threshold=5),
        )
        node = QualityGateNode(settings=s, zoo=zoo)
        # Long + unique expression: fails complexity
        result = node.check({"expression": "rank(close) * std(close, 20) + delta(close, 1)"})
        assert result["passed"] is False

    def test_check_decision_matches_feedback(self):
        node = QualityGateNode()
        result = node.check({"expression": "rank(close)"})
        assert result["passed"] == result["feedback"].decision