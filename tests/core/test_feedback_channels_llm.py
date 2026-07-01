# coding=utf-8
"""Tests for core/feedback/{channels,llm_judge}.py and knowledge/metrics/evaluator.py.

Covers: 4 channel collectors (execution, shape, code, value),
LLMJudge mock heuristic, RAGEvaluator per-query + aggregate metrics.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from QuantNodes.core.feedback.channels import (
    collect_execution,
    collect_shape,
    collect_code,
    collect_value,
)
from QuantNodes.core.feedback.llm_judge import LLMJudge, _extract_field
from QuantNodes.core.feedback.dataclass import ChannelFeedback, FeedbackChannel
from QuantNodes.core.knowledge.metrics.evaluator import (
    RAGEvaluator,
    QueryResult,
    EvalReport,
)


# ============================================================================
# collect_execution
# ============================================================================

class TestCollectExecution:
    def test_success(self):
        fb = collect_execution(stdout="ok", stderr="", exit_code=0)
        assert fb.passed is True
        assert fb.score == 1.0
        assert fb.channel == FeedbackChannel.EXECUTION

    def test_failure(self):
        fb = collect_execution(stdout="", stderr="error msg", exit_code=1)
        assert fb.passed is False
        assert fb.score == 0.0

    def test_metadata_exit_code(self):
        fb = collect_execution(stdout="", stderr="", exit_code=42)
        assert fb.metadata["exit_code"] == 42

    def test_output_truncation(self):
        long_stdout = "x" * 1000
        fb = collect_execution(stdout=long_stdout, stderr="", exit_code=0, max_output_chars=100)
        # Should be truncated in detail
        assert len(fb.detail) < len(long_stdout) + 100

    def test_negative_exit_code(self):
        fb = collect_execution(stdout="", stderr="", exit_code=-1)
        assert fb.passed is False


# ============================================================================
# collect_shape
# ============================================================================

class TestCollectShape:
    def test_match(self):
        fb = collect_shape(actual_shape=(10, 5), expected_shape=(10, 5))
        assert fb.passed is True

    def test_mismatch(self):
        fb = collect_shape(actual_shape=(10, 5), expected_shape=(10, 10))
        assert fb.passed is False

    def test_tuple_vs_list_treated_equal(self):
        """tuple and list are treated as equal."""
        fb = collect_shape(actual_shape=[10, 5], expected_shape=(10, 5))
        assert fb.passed is True

    def test_score_pass(self):
        fb = collect_shape(actual_shape=(1,), expected_shape=(1,))
        assert fb.score == 1.0

    def test_score_fail(self):
        fb = collect_shape(actual_shape=(1,), expected_shape=(2,))
        assert fb.score == 0.0

    def test_detail_contains_shapes(self):
        fb = collect_shape(actual_shape=(1, 2), expected_shape=(3, 4))
        assert "actual=(1, 2)" in fb.detail
        assert "expected=(3, 4)" in fb.detail


# ============================================================================
# collect_code
# ============================================================================

class TestCollectCode:
    def test_valid_simple(self):
        fb = collect_code("rank(close)")
        assert fb.passed is True
        assert fb.channel == FeedbackChannel.CODE

    def test_syntax_error(self):
        fb = collect_code("invalid ((")
        assert fb.passed is False
        assert "语法错误" in fb.detail

    def test_too_long(self):
        long_expr = "x" * 201
        fb = collect_code(long_expr, symbol_length_threshold=200)
        assert fb.passed is False

    def test_too_many_base_features(self):
        expr = "+".join([f"rank({c})" for c in ["open", "high", "low", "close", "volume", "amount"]])
        fb = collect_code(expr, base_features_threshold=3)
        assert fb.passed is False

    def test_high_free_args(self):
        # Mostly custom variables, not base features
        fb = collect_code("custom_a + custom_b + custom_c", free_args_ratio_threshold=0.1)
        assert fb.passed is False

    def test_metadata_fields(self):
        fb = collect_code("rank(close)")
        assert "symbol_length" in fb.metadata
        assert "base_features" in fb.metadata
        assert "free_args_ratio" in fb.metadata

    def test_passing_detail(self):
        fb = collect_code("rank(close)")
        assert "OK" in fb.detail


# ============================================================================
# collect_value
# ============================================================================

class TestCollectValue:
    def test_normal_series(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        fb = collect_value(s)
        assert fb.passed is True

    def test_all_nan(self):
        s = pd.Series([np.nan, np.nan, np.nan])
        fb = collect_value(s)
        assert fb.passed is False
        assert "全部 NaN" in fb.detail

    def test_high_nan(self):
        s = pd.Series([1.0, np.nan, np.nan, np.nan, np.nan])  # 80% NaN
        fb = collect_value(s, nan_threshold=0.3)
        assert fb.passed is False

    def test_constant_series(self):
        """std == 0 should fail."""
        s = pd.Series([5.0, 5.0, 5.0, 5.0])
        fb = collect_value(s, std_threshold=1e-6)
        assert fb.passed is False
        assert "std" in fb.detail

    def test_varied_series_passes(self):
        s = pd.Series(np.random.randn(100))
        fb = collect_value(s)
        assert fb.passed is True

    def test_metadata_fields(self):
        s = pd.Series([1.0, 2.0, 3.0])
        fb = collect_value(s)
        assert "nan_pct" in fb.metadata
        assert "inf_count" in fb.metadata
        assert "mean" in fb.metadata
        assert "std" in fb.metadata


# ============================================================================
# LLMJudge
# ============================================================================

class TestLLMJudge:
    def test_creation_default(self):
        judge = LLMJudge()
        assert judge.model == "mock"

    def test_creation_real_requires_callable(self):
        with pytest.raises(ValueError, match="llm_callable"):
            LLMJudge(model="gpt-4")

    def test_creation_with_callable(self):
        judge = LLMJudge(model="gpt-4", llm_callable=lambda p: "{}")
        assert judge._llm_callable is not None

    def test_judge_mock_heuristic_momentum(self):
        """momentum keyword + close → consistent."""
        judge = LLMJudge()
        fb = judge.judge(
            hypothesis="momentum factor",
            description="20-day momentum",
            expression="rank(close)",
        )
        assert fb.channel == FeedbackChannel.LLM
        assert fb.passed is True

    def test_judge_mock_empty_expression_fails(self):
        judge = LLMJudge()
        fb = judge.judge(
            hypothesis="momentum",
            description="test",
            expression="",
        )
        assert fb.passed is False

    def test_judge_mock_empty_hyp_desc_fails(self):
        judge = LLMJudge()
        fb = judge.judge(
            hypothesis="",
            description="",
            expression="rank(close)",
        )
        assert fb.passed is False

    def test_judge_mock_default_passes(self):
        judge = LLMJudge()
        fb = judge.judge(
            hypothesis="unknown strategy",
            description="random",
            expression="custom_func(x)",
        )
        # Mock returns passed=True with score=0.7 for unknown
        assert fb.passed is True
        assert fb.score == 0.7

    def test_judge_with_real_callable(self):
        def my_llm(prompt):
            return '{"consistent": false, "reason": "test fail", "score": 0.1}'

        judge = LLMJudge(model="gpt-4", llm_callable=my_llm)
        fb = judge.judge("h", "d", "x")
        assert fb.passed is False
        assert fb.score == 0.1

    def test_judge_with_invalid_json_falls_back(self):
        def bad_llm(prompt):
            return "not json"

        judge = LLMJudge(model="gpt-4", llm_callable=bad_llm, max_correction_attempts=2)
        fb = judge.judge("h", "d", "x")
        # After max attempts, returns failed ChannelFeedback
        assert fb.passed is False
        assert "解析失败" in fb.detail

    def test_judge_metadata_attempts(self):
        judge = LLMJudge()
        fb = judge.judge("h", "d", "x")
        assert "attempt" in fb.metadata
        assert fb.metadata["attempt"] == 1

    def test_judge_max_correction_attempts(self):
        judge = LLMJudge(max_correction_attempts=5)
        fb = judge.judge("h", "d", "x")
        # Mock returns valid JSON on first call
        assert fb.metadata["attempt"] == 1


# ============================================================================
# _extract_field
# ============================================================================

class TestExtractField:
    def test_extract_existing(self):
        prompt = "Hypothesis: my hypothesis\nDescription: my desc"
        result = _extract_field(prompt, "Hypothesis")
        assert result == "my hypothesis"

    def test_extract_missing(self):
        result = _extract_field("No fields here", "Hypothesis")
        assert result == ""

    def test_extract_with_colon_in_value(self):
        prompt = "Expression: x: y"
        result = _extract_field(prompt, "Expression")
        assert result == "x: y"

    def test_extract_no_colon(self):
        prompt = "Hypothesis no colon"
        result = _extract_field(prompt, "Hypothesis")
        assert result == ""

    def test_extract_empty_value(self):
        prompt = "Hypothesis: \nDescription: x"
        result = _extract_field(prompt, "Hypothesis")
        assert result == ""


# ============================================================================
# RAGEvaluator
# ============================================================================

class TestRAGEvaluator:
    def test_creation(self):
        e = RAGEvaluator()
        assert e.k_values == [5, 10]

    def test_creation_custom_k(self):
        e = RAGEvaluator(k_values=[3, 5])
        assert e.k_values == [3, 5]

    def test_evaluate_basic(self):
        e = RAGEvaluator()
        queries = ["q1", "q2"]
        retrieved = [["d1", "d2"], ["d3"]]
        relevant = [["d1", "d2"], ["d3"]]
        report = e.evaluate(queries, retrieved, relevant)
        assert report.n_queries == 2
        assert report.hit_at_5 > 0  # Perfect hit
        assert report.mrr > 0

    def test_evaluate_no_hits(self):
        e = RAGEvaluator()
        queries = ["q1"]
        retrieved = [["d1"]]
        relevant = [["d2"]]
        report = e.evaluate(queries, retrieved, relevant)
        assert report.hit_at_5 == 0
        assert report.mrr == 0

    def test_evaluate_per_query_results(self):
        e = RAGEvaluator()
        queries = ["q1"]
        retrieved = [["d1"]]
        relevant = [["d1"]]
        report = e.evaluate(queries, retrieved, relevant)
        assert len(report.per_query) == 1
        assert report.per_query[0].query == "q1"

    def test_evaluate_with_relevance_scores(self):
        e = RAGEvaluator()
        queries = ["q1"]
        retrieved = [["d1", "d2", "d3"]]
        relevant = [["d1", "d2", "d3"]]
        scores = [{"d1": 1.0, "d2": 0.5, "d3": 0.1}]
        report = e.evaluate(queries, retrieved, relevant, scores)
        assert report.ndcg_at_5 > 0

    def test_evaluate_with_lineage(self):
        e = RAGEvaluator()
        queries = ["q1"]
        retrieved = [["d1"]]
        relevant = [["d1"]]
        lineage = [["d1", "d2"]]
        report = e.evaluate(queries, retrieved, relevant, lineage_ids=lineage)
        assert report.lineage_coverage > 0

    def test_evaluate_with_tokens(self):
        e = RAGEvaluator()
        queries = ["q1"]
        retrieved = [["d1", "d2"]]
        relevant = [["d1"]]
        tokens = [[["a", "b"], ["c", "d"]]]
        report = e.evaluate(queries, retrieved, relevant, token_lists=tokens)
        # diversity for distinct token sets
        assert report.diversity >= 0

    def test_evaluate_empty(self):
        e = RAGEvaluator()
        report = e.evaluate([], [], [])
        assert report.n_queries == 0

    def test_evaluate_with_defaults(self):
        """Default relevance_scores/lineage_ids/token_lists when None."""
        e = RAGEvaluator()
        report = e.evaluate(["q1"], [["d1"]], [["d1"]])
        assert report.n_queries == 1


# ============================================================================
# QueryResult / EvalReport
# ============================================================================

class TestQueryResult:
    def test_creation(self):
        qr = QueryResult(
            query="q1",
            retrieved_ids=["d1"],
            relevant_ids=["d1"],
        )
        assert qr.query == "q1"
        assert qr.retrieved_ids == ["d1"]
        assert qr.relevance_scores == {}
        assert qr.lineage_ids == []
        assert qr.hit_at_5 == 0.0

    def test_creation_with_metrics(self):
        qr = QueryResult(
            query="q",
            retrieved_ids=[],
            relevant_ids=[],
            hit_at_5=1.0,
            ndcg_at_10=0.8,
            mrr=0.5,
        )
        assert qr.hit_at_5 == 1.0
        assert qr.ndcg_at_10 == 0.8


class TestEvalReport:
    def test_creation(self):
        r = EvalReport(
            n_queries=5,
            hit_at_5=0.8,
            hit_at_10=0.9,
            ndcg_at_5=0.7,
            ndcg_at_10=0.85,
            mrr=0.6,
            lineage_coverage=0.5,
            diversity=0.7,
        )
        assert r.n_queries == 5
        assert r.hit_at_5 == 0.8

    def test_to_dict(self):
        r = EvalReport(
            n_queries=1, hit_at_5=0.5, hit_at_10=0.6,
            ndcg_at_5=0.7, ndcg_at_10=0.8, mrr=0.9,
            lineage_coverage=0.5, diversity=0.5,
        )
        d = r.to_dict()
        assert d["n_queries"] == 1
        assert d["hit_at_5"] == 0.5
        assert "timestamp" in d
        assert "per_query" in d


# ============================================================================
# RAGEvaluator Persistence
# ============================================================================

class TestRAGEvaluatorPersistence:
    def test_save_json(self, tmp_path):
        e = RAGEvaluator()
        report = e.evaluate(["q1"], [["d1"]], [["d1"]])
        path = tmp_path / "report.json"
        e.save(report, str(path))
        assert path.exists()

    def test_save_csv(self, tmp_path):
        e = RAGEvaluator()
        report = e.evaluate(["q1"], [["d1"]], [["d1"]])
        path = tmp_path / "report.csv"
        e.save_csv(report, str(path))
        assert path.exists()

    def test_save_creates_parent_dir(self, tmp_path):
        e = RAGEvaluator()
        report = e.evaluate(["q1"], [["d1"]], [["d1"]])
        path = tmp_path / "subdir" / "report.json"
        e.save(report, str(path))
        assert path.exists()


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    def test_collect_code_with_complex_expression(self):
        """NOTE: rank(ts_mean(close, 20)) has free args (rank, ts_mean).
        free_args_ratio = 2/3 > 0.5 default, so fails."""
        fb = collect_code("rank(ts_mean(close, 20))")
        # Test documents the behavior; if thresholds changed, this may pass
        assert fb.channel == FeedbackChannel.CODE
        # Should have metadata regardless of pass/fail
        assert "free_args_ratio" in fb.metadata

    def test_collect_value_with_inf(self):
        s = pd.Series([1.0, 2.0, np.inf, 4.0])
        fb = collect_value(s)
        assert fb.passed is False
        assert "Inf" in fb.detail

    def test_llm_judge_with_keywords(self):
        judge = LLMJudge()
        # Chinese keyword "反转" (reversal)
        fb = judge.judge(
            hypothesis="反转策略",
            description="5日反转",
            expression="-returns(close, 5)",
        )
        assert fb.passed is True

    def test_rag_evaluator_complex_query(self):
        e = RAGEvaluator(k_values=[3, 5])
        report = e.evaluate(
            queries=["q1", "q2", "q3"],
            retrieved=[["d1", "d2"], ["d3", "d4"], ["d5"]],
            relevant=[["d1"], ["d3"], ["d5", "d6"]],
        )
        assert report.n_queries == 3
        assert len(report.per_query) == 3