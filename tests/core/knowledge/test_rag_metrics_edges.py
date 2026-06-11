"""RAG 评估指标 + RAGEvaluator 边界测试 (20 tests)。

聚焦:
    - HitRate@K: 命中/未命中/k 边界/多 query 平均
    - NDCG@K: 理想/非理想/全 0 relevance
    - MRR: 首个相关位置/全未命中
    - LineageCoverage: 全覆盖/部分覆盖/空 lineage
    - IntraListDiversity: 单元素/多元素/全相同/全不同
    - Jaccard: 边界
    - RAGEvaluator: 5 指标汇总/per_query/空 queries
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from QuantNodes.core.knowledge.metrics import (
    dcg_at_k,
    hit_rate_at_k,
    intra_list_diversity,
    jaccard_similarity,
    lineage_coverage,
    mean_hit_rate_at_k,
    mean_lineage_coverage,
    mean_ndcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    reciprocal_rank,
)
from QuantNodes.core.knowledge.metrics.evaluator import (
    EvalReport,
    QueryResult,
    RAGEvaluator,
)


# ============================================================================
# 1. HitRate@K (4 tests)
# ============================================================================

class TestHitRate:
    def test_hit_top_1(self):
        assert hit_rate_at_k(["a", "b", "c"], {"a"}, k=5) == 1.0

    def test_miss_top_k(self):
        assert hit_rate_at_k(["x", "y", "z"], {"a", "b"}, k=5) == 0.0

    def test_hit_outside_k(self):
        """相关 doc 在 k 之后 → miss。"""
        assert hit_rate_at_k(["a", "b", "c", "d"], {"d"}, k=3) == 0.0

    def test_mean_hit_rate(self):
        """50% query 命中 → 0.5。"""
        ret = [["a", "x"], ["x", "y"]]
        rel = [{"a"}, {"a"}]
        assert mean_hit_rate_at_k(ret, rel, k=5) == 0.5

    def test_empty_queries_returns_zero(self):
        assert mean_hit_rate_at_k([], [], k=5) == 0.0


# ============================================================================
# 2. DCG@K / NDCG@K (5 tests)
# ============================================================================

class TestNDCG:
    def test_dcg_perfect_ranking(self):
        """完美排序: dcg = 1/log2(2) + 1/log2(3) = 1.0 + 0.63。"""
        rels = {"a": 1.0, "b": 1.0}
        dcg = dcg_at_k(["a", "b"], rels, k=5)
        assert math.isclose(dcg, 1.0 + 1.0 / math.log2(3), rel_tol=1e-6)

    def test_dcg_zero_relevance(self):
        """所有 retrieved 不在 relevance → dcg=0。"""
        rels = {"a": 1.0}
        dcg = dcg_at_k(["x", "y", "z"], rels, k=5)
        assert dcg == 0.0

    def test_ndcg_perfect(self):
        """完美排序 → NDCG=1.0。"""
        rels = {"a": 1.0, "b": 1.0, "c": 1.0}
        assert ndcg_at_k(["a", "b", "c"], rels, k=5) == 1.0

    def test_ndcg_worse_with_graded_relevance(self):
        """用不同 relevance → 反序 NDCG<1。"""
        rels = {"a": 3.0, "b": 2.0, "c": 1.0}  # a 最相关
        n = ndcg_at_k(["c", "b", "a"], rels, k=5)
        # 反序: a (3.0) 排最后 → DCG 小于 ideal
        assert n < 1.0
        assert n > 0.0

    def test_ndcg_zero_ideal(self):
        """ideal=0 (空 relevance) → NDCG=0。"""
        assert ndcg_at_k(["a", "b"], {}, k=5) == 0.0

    def test_mean_ndcg(self):
        ret = [["a", "b"], ["c", "d"]]
        rel = [{"a": 1.0, "b": 1.0}, {"c": 1.0, "d": 1.0}]
        # 两个 perfect → 1.0
        m = mean_ndcg_at_k(ret, rel, k=5)
        assert math.isclose(m, 1.0, rel_tol=1e-6)


# ============================================================================
# 3. MRR (3 tests)
# ============================================================================

class TestMRR:
    def test_first_position(self):
        assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0

    def test_third_position(self):
        assert reciprocal_rank(["x", "y", "a"], {"a"}) == 1.0 / 3

    def test_no_match(self):
        assert reciprocal_rank(["x", "y", "z"], {"a"}) == 0.0

    def test_mean_mrr(self):
        ret = [["a", "b"], ["x", "a"]]
        rel = [{"a"}, {"a"}]
        # 1.0 + 0.5 = 1.5 / 2 = 0.75
        assert mean_reciprocal_rank(ret, rel) == 0.75


# ============================================================================
# 4. LineageCoverage (3 tests)
# ============================================================================

class TestLineageCov:
    def test_full_coverage(self):
        assert lineage_coverage(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_partial_coverage(self):
        # 2 of 3 lineage ids 在 retrieved
        assert lineage_coverage(["a", "b"], ["a", "b", "c"]) == 2.0 / 3

    def test_empty_lineage_returns_zero(self):
        # lineage 为空时返回 0.0 (避免除零)
        assert lineage_coverage(["a", "b"], []) == 0.0

    def test_no_overlap(self):
        assert lineage_coverage(["a", "b"], ["c", "d"]) == 0.0

    def test_mean_lineage_coverage(self):
        ret = [["a", "b"], ["a"]]
        lin = [["a", "b", "c"], ["a", "b"]]
        # 2/3 + 1/2 = 0.917
        m = mean_lineage_coverage(ret, lin)
        assert abs(m - (2.0/3 + 0.5) / 2) < 1e-6


# ============================================================================
# 5. Intra-List Diversity (3 tests)
# ============================================================================

class TestDiversity:
    def test_empty_returns_zero(self):
        assert intra_list_diversity([]) == 0.0

    def test_single_token_list(self):
        """单元素 → 1.0 (最大多样)。"""
        items = [[["a", "b", "c"]]]
        assert intra_list_diversity(items) == 1.0

    def test_identical_tokens_zero_diversity(self):
        """全相同 → 0 (无多样)。"""
        items = [[["a", "b"], ["a", "b"], ["a", "b"]]]
        assert intra_list_diversity(items) == 0.0

    def test_disjoint_tokens_max_diversity(self):
        """完全无重叠 → 1.0。"""
        items = [[["a", "b"], ["c", "d"]]]
        assert intra_list_diversity(items) == 1.0

    def test_partial_overlap(self):
        items = [[["a", "b", "c"], ["a", "b", "d"]]]
        # Jaccard: |{a,b} ∩ {a,b,d}| / |{a,b,c} ∪ {a,b,d}| = 2/4 = 0.5
        # div = 1 - 0.5 = 0.5
        d = intra_list_diversity(items)
        assert 0.4 < d < 0.6


# ============================================================================
# 6. Jaccard Similarity (2 tests)
# ============================================================================

class TestJaccard:
    def test_identical(self):
        assert jaccard_similarity(["a", "b"], ["a", "b"]) == 1.0

    def test_disjoint(self):
        assert jaccard_similarity(["a"], ["b"]) == 0.0

    def test_both_empty(self):
        assert jaccard_similarity([], []) == 0.0  # 防除零

    def test_one_empty(self):
        # set_b is empty → union is just set_a → sim = 0 / |set_a| = 0
        assert jaccard_similarity(["a"], []) == 0.0


# ============================================================================
# 7. RAGEvaluator 集成 (5 tests)
# ============================================================================

class TestRAGEvaluator:
    def test_empty_queries(self):
        e = RAGEvaluator()
        report = e.evaluate([], [], [])
        assert report.n_queries == 0
        assert report.hit_at_5 == 0.0
        assert report.diversity == 0.0

    def test_single_query_perfect(self):
        e = RAGEvaluator()
        report = e.evaluate(
            queries=["q1"],
            retrieved=[["a", "b", "c"]],
            relevant=[["a", "b", "c"]],
            relevance_scores=[{"a": 1.0, "b": 1.0, "c": 1.0}],
            lineage_ids=[["a", "b", "c"]],
            token_lists=[[["alpha"], ["beta"], ["gamma"]]],
        )
        assert report.n_queries == 1
        assert report.hit_at_5 == 1.0
        assert report.ndcg_at_5 == 1.0
        assert report.mrr == 1.0
        assert report.lineage_coverage == 1.0
        # diversity 高 (alpha/beta/gamma 无重叠)
        assert report.diversity > 0.9

    def test_single_query_zero_relevance(self):
        e = RAGEvaluator()
        report = e.evaluate(
            queries=["q1"],
            retrieved=[["x", "y"]],
            relevant=[[]],
            relevance_scores=[{}],
        )
        assert report.hit_at_5 == 0.0
        assert report.mrr == 0.0
        assert report.lineage_coverage == 0.0

    def test_per_query_results(self):
        e = RAGEvaluator()
        report = e.evaluate(
            queries=["q1", "q2"],
            retrieved=[["a", "b"], ["c", "d"]],
            relevant=[["a"], ["c"]],
        )
        assert len(report.per_query) == 2
        assert report.per_query[0].query == "q1"
        assert report.per_query[0].hit_at_5 == 1.0
        assert report.per_query[1].query == "q2"

    def test_save_json_roundtrip(self, tmp_path: Path):
        e = RAGEvaluator()
        report = e.evaluate(
            queries=["q1"],
            retrieved=[["a", "b"]],
            relevant=[["a"]],
        )
        path = tmp_path / "report.json"
        e.save(report, path)
        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert data["n_queries"] == 1
        assert "per_query" in data

    def test_save_csv(self, tmp_path: Path):
        e = RAGEvaluator()
        report = e.evaluate(
            queries=["q1", "q2"],
            retrieved=[["a"], ["b"]],
            relevant=[["a"], ["b"]],
        )
        path = tmp_path / "report.csv"
        e.save_csv(report, path)
        assert path.exists()

    def test_custom_k_values(self):
        e = RAGEvaluator(k_values=[3, 7])
        report = e.evaluate(
            queries=["q1"],
            retrieved=[["a", "b", "c", "d", "e"]],
            relevant=[["c"]],
        )
        # k=3: c 在 top 3 → hit
        assert report.hit_at_5 > 0  # 实际 hit_at_5 = 1 (因为 k_values[0]=3)

    def test_eval_report_to_dict(self):
        r = EvalReport(
            n_queries=1, hit_at_5=0.5, hit_at_10=0.8,
            ndcg_at_5=0.6, ndcg_at_10=0.7, mrr=0.5,
            lineage_coverage=0.4, diversity=0.9,
        )
        d = r.to_dict()
        assert d["n_queries"] == 1
        assert d["hit_at_5"] == 0.5
        assert d["diversity"] == 0.9
        assert "timestamp" in d
