"""RAG 评估指标 (Week 10)。

公开 API:
    - 5 个核心指标函数:
        hit_rate_at_k / mean_hit_rate_at_k
        ndcg_at_k / dcg_at_k / mean_ndcg_at_k
        reciprocal_rank / mean_reciprocal_rank
        lineage_coverage / mean_lineage_coverage
        intra_list_diversity / jaccard_similarity
    - RAGEvaluator: 汇总多 query 评估
    - EvalReport / QueryResult: 结果数据类
"""
from .metrics import (
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
from .evaluator import EvalReport, QueryResult, RAGEvaluator

__all__ = [
    "dcg_at_k",
    "hit_rate_at_k",
    "intra_list_diversity",
    "jaccard_similarity",
    "lineage_coverage",
    "mean_hit_rate_at_k",
    "mean_lineage_coverage",
    "mean_ndcg_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "reciprocal_rank",
    "EvalReport",
    "QueryResult",
    "RAGEvaluator",
]
