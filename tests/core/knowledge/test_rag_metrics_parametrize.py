"""RAG metrics 全参数覆盖测试 (~40 tests, @pytest.mark.parametrize)。

遍历 hit_rate/ndcg/mrr/lineage_coverage/diversity/jaccard 每个函数的所有参数。
"""
from __future__ import annotations

import math

import pytest

from QuantNodes.core.knowledge.metrics import (
    dcg_at_k,
    hit_rate_at_k,
    intra_list_diversity,
    jaccard_similarity,
    lineage_coverage,
    mean_reciprocal_rank,
    ndcg_at_k,
    reciprocal_rank,
)


# ============================================================================
# 1. hit_rate_at_k 3 参数 (12 tests)
# ============================================================================

class TestHitRateAtKParams:
    @pytest.mark.parametrize("retrieved,relevant,k,expected", [
        (["a"], {"a"}, 1, 1.0),
        (["a"], {"a"}, 5, 1.0),  # k 超出
        (["a", "b", "c"], {"a"}, 3, 1.0),  # 首位置
        (["a", "b", "c"], {"c"}, 3, 1.0),  # 末位置
        (["a", "b", "c"], {"d"}, 3, 0.0),  # 完全 miss
        (["a", "b", "c", "d"], {"d"}, 3, 0.0),  # k 截断
        (["a", "b", "c", "d"], {"d"}, 4, 1.0),  # k 包含
        ([], {"a"}, 5, 0.0),  # 空
        (["a"], set(), 5, 0.0),  # 空 relevant
        (["a", "b"], {"a", "b"}, 2, 1.0),  # 多相关
    ])
    def test_combinations(self, retrieved, relevant, k, expected):
        assert hit_rate_at_k(retrieved, relevant, k=k) == expected

    @pytest.mark.parametrize("k", [0, 1, -1, 100, 1000])
    def test_k_edge_values(self, k):
        """k=0/负/超量都应不崩。"""
        r = hit_rate_at_k(["a", "b"], {"a"}, k=k)
        assert r in (0.0, 1.0)


# ============================================================================
# 2. dcg_at_k / ndcg_at_k 3 参数 (10 tests)
# ============================================================================

class TestDCGNDCGParams:
    @pytest.mark.parametrize("retrieved,relevance,k,min_val,max_val", [
        (["a", "b", "c"], {"a": 3.0, "b": 2.0, "c": 1.0}, 3, 0, 6),
        (["a", "b", "c"], {"a": 1.0, "b": 1.0, "c": 1.0}, 3, 2.0, 2.5),
        (["x", "y", "z"], {"a": 1.0}, 3, 0, 0),
        (["a"], {"a": 5.0}, 1, 5.0, 5.0),
        (["a", "b"], {}, 2, 0, 0),
    ])
    def test_dcg_bounds(self, retrieved, relevance, k, min_val, max_val):
        d = dcg_at_k(retrieved, relevance, k=k)
        assert min_val <= d <= max_val + 1

    @pytest.mark.parametrize("retrieved,relevance,expected_range", [
        (["a", "b", "c"], {"a": 1.0, "b": 1.0, "c": 1.0}, (1.0, 1.0)),  # perfect
        ([], {"a": 1.0}, (0.0, 0.0)),  # 空
        (["a"], {}, (0.0, 0.0)),  # 空 relevance → 0
        (["a", "b"], {"a": 0.0}, (0.0, 0.0)),  # 0 relevance
    ])
    def test_ndcg_ranges(self, retrieved, relevance, expected_range):
        n = ndcg_at_k(retrieved, relevance, k=5)
        lo, hi = expected_range
        assert lo <= n <= hi

    @pytest.mark.parametrize("k", [1, 2, 3, 5, 10, 100])
    def test_ndcg_various_k(self, k):
        rels = {chr(ord("a") + i): 1.0 for i in range(20)}
        # 完美排序
        ret = list(rels.keys())[:min(k, 20)]
        n = ndcg_at_k(ret, rels, k=k)
        assert n == pytest.approx(1.0, abs=1e-6)


# ============================================================================
# 3. reciprocal_rank / mean_reciprocal_rank 3 参数 (8 tests)
# ============================================================================

class TestMRRParams:
    @pytest.mark.parametrize("retrieved,relevant,expected", [
        (["a", "b", "c"], {"a"}, 1.0),
        (["a", "b", "c"], {"b"}, 0.5),
        (["a", "b", "c"], {"c"}, 1.0 / 3),
        (["a", "b", "c"], {"a", "b"}, 1.0),  # 首位置
        (["x", "y", "z"], {"a"}, 0.0),
        ([], {"a"}, 0.0),
        (["a"], set(), 0.0),
        (["a", "b", "c"], {"x", "y"}, 0.0),
    ])
    def test_reciprocal_rank(self, retrieved, relevant, expected):
        assert reciprocal_rank(retrieved, relevant) == expected

    @pytest.mark.parametrize("queries,expected", [
        ([(["a"], {"a"}), (["b"], {"b"})], 1.0),  # 两 perfect
        ([(["a"], {"b"}), (["a"], {"b"})], 0.0),  # 全 miss
        ([], 0.0),  # 空
    ])
    def test_mean_mrr(self, queries, expected):
        ret = [q[0] for q in queries]
        rel = [q[1] for q in queries]
        assert mean_reciprocal_rank(ret, rel) == expected


# ============================================================================
# 4. lineage_coverage 3 参数 (8 tests)
# ============================================================================

class TestLineageCoverageParams:
    @pytest.mark.parametrize("retrieved,lineage,expected", [
        (["a", "b", "c"], ["a", "b", "c"], 1.0),
        (["a", "b"], ["a", "b", "c"], 2.0 / 3),
        (["a"], ["a", "b", "c"], 1.0 / 3),
        ([], ["a", "b", "c"], 0.0),
        (["a", "b"], [], 0.0),  # 空 lineage
        (["a", "b"], ["x", "y"], 0.0),
        (["a", "b", "c", "d"], ["a"], 1.0),
    ])
    def test_coverage(self, retrieved, lineage, expected):
        assert lineage_coverage(retrieved, lineage) == pytest.approx(expected, abs=1e-6)

    def test_empty_inputs_zero(self):
        assert lineage_coverage([], []) == 0.0


# ============================================================================
# 5. jaccard 4 参数组合 (8 tests)
# ============================================================================

class TestJaccardParams:
    @pytest.mark.parametrize("a,b,expected", [
        (["x", "y"], ["x", "y"], 1.0),
        (["x"], ["y"], 0.0),
        ([], [], 0.0),  # 双空 (防除零)
        (["x"], [], 0.0),  # 一空
        ([], ["x"], 0.0),
        (["x", "y", "z"], ["x", "y"], 2.0 / 3),
        (["x", "y"], ["x", "y", "z"], 2.0 / 3),
        (["x", "y", "z"], ["a", "b"], 0.0),
    ])
    def test_jaccard(self, a, b, expected):
        assert jaccard_similarity(a, b) == pytest.approx(expected, abs=1e-6)


# ============================================================================
# 6. intra_list_diversity 2 参数 (8 tests)
# ============================================================================

class TestDiversityParams:
    @pytest.mark.parametrize("items,expected", [
        ([[["a"], ["a"]]], 0.0),  # 相同
        ([[["a"], ["b"]]], 1.0),  # 完全不同
        ([[["a", "b"], ["a", "b"]]], 0.0),  # 全同
        ([[["a", "b"], ["c", "d"]]], 1.0),  # 无交集
        ([[["a"], ["b"], ["c"]]], 1.0),  # 3 个全不同
        ([[]], 1.0),  # 单元素 query
        ([], 0.0),  # 空 (0 queries)
    ])
    def test_intra_diversity(self, items, expected):
        assert intra_list_diversity(items) == pytest.approx(expected, abs=1e-6)

    def test_similarity_fn_custom(self):
        """自定义 similarity_fn。"""
        def custom_sim(a, b):
            return 0.5 if set(a) & set(b) else 0.0
        items = [[["a"], ["b"]]]
        # sim=0 → div=1.0
        d = intra_list_diversity(items, similarity_fn=custom_sim)
        assert d == 1.0

    def test_multi_query_average(self):
        """多 query 求平均。"""
        items = [
            [["a"], ["a"]],  # div=0
            [["x"], ["y"]],  # div=1
        ]
        # 平均 = 0.5
        assert intra_list_diversity(items) == 0.5
