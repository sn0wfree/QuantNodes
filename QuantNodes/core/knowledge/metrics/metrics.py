"""RAG 评估指标 — 5 个核心指标。

- HitRate@K:    检索 Top-K 是否命中相关 entry
- NDCG@K:       Normalized Discounted Cumulative Gain (位置权重)
- MRR:          Mean Reciprocal Rank (首个相关 entry 的倒数排名)
- LineageCoverage: 检索结果覆盖的谱系比例 (vs. ground truth 谱系)
- IntraListDiversity: 检索结果内部多样性 (1 - 平均 pairwise similarity)
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence


# ============================================================================
# 1. HitRate@K
# ============================================================================

def hit_rate_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int = 5,
) -> float:
    """检索 Top-K 中是否含至少 1 个 relevant entry。

    Args:
        retrieved_ids: 检索器返回的 entry_id 列表 (有序)
        relevant_ids: 真实相关 entry_id 集合
        k: 截断 K

    Returns:
        float: 1.0 (命中) / 0.0 (未命中)
    """
    rel_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    return 1.0 if any(rid in rel_set for rid in top_k) else 0.0


def mean_hit_rate_at_k(
    queries_retrieved: Sequence[Sequence[str]],
    queries_relevant: Sequence[Iterable[str]],
    k: int = 5,
) -> float:
    """多 query 平均 HitRate@K。"""
    if not queries_retrieved:
        return 0.0
    scores = [
        hit_rate_at_k(ret, rel, k=k)
        for ret, rel in zip(queries_retrieved, queries_relevant)
    ]
    return sum(scores) / len(scores)


# ============================================================================
# 2. NDCG@K
# ============================================================================

def dcg_at_k(
    retrieved_ids: Sequence[str],
    relevance_scores: Mapping[str, float],
    k: int = 5,
) -> float:
    """Discounted Cumulative Gain。

    relevance_scores: doc_id -> relevance (通常 0/1, 可更高)
    """
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        rel = relevance_scores.get(rid, 0.0)
        # 标准公式: rel_i / log2(i+2)  (i=0 -> log2(2)=1)
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevance_scores: Mapping[str, float],
    k: int = 5,
) -> float:
    """Normalized DCG (0-1)。"""
    actual = dcg_at_k(retrieved_ids, relevance_scores, k=k)
    # ideal: 按 relevance 降序排列的 DCG
    ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    ideal = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    if ideal == 0:
        return 0.0
    return actual / ideal


def mean_ndcg_at_k(
    queries_retrieved: Sequence[Sequence[str]],
    queries_relevance: Sequence[Mapping[str, float]],
    k: int = 5,
) -> float:
    """多 query 平均 NDCG@K。"""
    if not queries_retrieved:
        return 0.0
    scores = [
        ndcg_at_k(ret, rel, k=k)
        for ret, rel in zip(queries_retrieved, queries_relevance)
    ]
    return sum(scores) / len(scores)


# ============================================================================
# 3. MRR (Mean Reciprocal Rank)
# ============================================================================

def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Iterable[str],
) -> float:
    """首个 relevant entry 的倒数排名 (0 表示未命中)。"""
    rel_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids, 1):
        if rid in rel_set:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(
    queries_retrieved: Sequence[Sequence[str]],
    queries_relevant: Sequence[Iterable[str]],
) -> float:
    """多 query 平均 MRR。"""
    if not queries_retrieved:
        return 0.0
    return sum(
        reciprocal_rank(ret, rel)
        for ret, rel in zip(queries_retrieved, queries_relevant)
    ) / len(queries_retrieved)


# ============================================================================
# 4. Lineage Coverage (谱系覆盖率)
# ============================================================================

def lineage_coverage(
    retrieved_ids: Iterable[str],
    lineage_ids: Iterable[str],
) -> float:
    """检索结果中包含的 ground truth 谱系 entry 比例。

    Args:
        retrieved_ids: 检索器返回的 entry_id 列表
        lineage_ids: ground truth 谱系 (e.g. expand_lineage 的 ancestors + descendants)

    Returns:
        float: 0-1, 1.0 = 谱系完全覆盖
    """
    ret_set = set(retrieved_ids)
    lineage_set = set(lineage_ids)
    if not lineage_set:
        return 0.0
    covered = sum(1 for lid in lineage_set if lid in ret_set)
    return covered / len(lineage_set)


def mean_lineage_coverage(
    queries_retrieved: Sequence[Iterable[str]],
    queries_lineage: Sequence[Iterable[str]],
) -> float:
    """多 query 平均 lineage coverage。"""
    if not queries_retrieved:
        return 0.0
    return sum(
        lineage_coverage(ret, lin)
        for ret, lin in zip(queries_retrieved, queries_lineage)
    ) / len(queries_retrieved)


# ============================================================================
# 5. Intra-List Diversity
# ============================================================================

def intra_list_diversity(
    items: Sequence[Sequence[Sequence[str]]],
    similarity_fn=None,
) -> float:
    """检索结果内部多样性: 1 - 平均 pairwise jaccard 相似度。

    Args:
        items: 多 query 的 token 化结果。结构 = Sequence[Sequence[Sequence[str]]]
            - 外层: 每个 query 一份
            - 中层: 每 query 的 Top-K 个条目
            - 内层: 每个条目的 token 列表
            例: [[['momentum', 'close'], ['reversal', 'open']], ...]
        similarity_fn: 自定义相似度函数 (默认 jaccard_similarity)

    Returns:
        float: 0-1, 1.0 = 完全多样
    """
    if not items:
        return 0.0
    if similarity_fn is None:
        similarity_fn = jaccard_similarity

    diversities: list[float] = []
    for tokens_list in items:
        if len(tokens_list) < 2:
            diversities.append(1.0)  # 单元素算最大多样
            continue
        sims: list[float] = []
        for i in range(len(tokens_list)):
            for j in range(i + 1, len(tokens_list)):
                sims.append(similarity_fn(tokens_list[i], tokens_list[j]))
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        diversities.append(1.0 - avg_sim)
    return sum(diversities) / len(diversities)


def jaccard_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    """Jaccard 相似度 = |A ∩ B| / |A ∪ B|。"""
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)
