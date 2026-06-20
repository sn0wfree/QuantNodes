"""RAGEvaluator — 汇总 5 个指标, 输出统一报告。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from QuantNodes.core.path_utils import ensure_parent

from .metrics import (
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


@dataclass
class QueryResult:
    """单个 query 的评估结果。"""
    query: str
    retrieved_ids: list[str]
    relevant_ids: list[str]
    relevance_scores: dict[str, float] = field(default_factory=dict)
    lineage_ids: list[str] = field(default_factory=list)
    hit_at_5: float = 0.0
    hit_at_10: float = 0.0
    ndcg_at_5: float = 0.0
    ndcg_at_10: float = 0.0
    mrr: float = 0.0
    lineage_cov: float = 0.0
    diversity: float = 0.0


@dataclass
class EvalReport:
    """评估汇总报告。"""
    n_queries: int
    hit_at_5: float
    hit_at_10: float
    ndcg_at_5: float
    ndcg_at_10: float
    mrr: float
    lineage_coverage: float
    diversity: float
    per_query: list[QueryResult] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "hit_at_5": self.hit_at_5,
            "hit_at_10": self.hit_at_10,
            "ndcg_at_5": self.ndcg_at_5,
            "ndcg_at_10": self.ndcg_at_10,
            "mrr": self.mrr,
            "lineage_coverage": self.lineage_coverage,
            "diversity": self.diversity,
            "timestamp": self.timestamp.isoformat(),
            "per_query": [asdict(q) for q in self.per_query],
        }


class RAGEvaluator:
    """RAG 评估器 — 接受多 query 结果, 汇总 5 个指标。

    Args:
        k_values: HitRate/NDCG 评估的 K 列表 (默认 [5, 10])
    """

    def __init__(self, k_values: list[int] | None = None):
        self.k_values = k_values or [5, 10]

    def evaluate(
        self,
        queries: list[str],
        retrieved: list[list[str]],
        relevant: list[list[str]],
        relevance_scores: list[dict[str, float]] | None = None,
        lineage_ids: list[list[str]] | None = None,
        token_lists: list[list[list[str]]] | None = None,
    ) -> EvalReport:
        """评估多 query 结果。

        Args:
            queries: 查询文本列表
            retrieved: 每 query 的检索结果 (entry_id 列表, 有序)
            relevant: 每 query 的相关 entry_id 集合
            relevance_scores: 每 query 的 doc_id -> relevance (NDCG 用, 可选)
            lineage_ids: 每 query 的 ground truth 谱系 entry_id (可选)
            token_lists: 每 query 的检索条目 token 列表 (diversity 用, 可选)

        Returns:
            EvalReport
        """
        n = len(queries)
        relevance_scores = relevance_scores or [{} for _ in range(n)]
        lineage_ids = lineage_ids or [[] for _ in range(n)]
        token_lists = token_lists or [[] for _ in range(n)]

        # Per-query
        per_query: list[QueryResult] = []
        for q, ret, rel, scores, lin, toks in zip(
            queries, retrieved, relevant, relevance_scores, lineage_ids, token_lists
        ):
            k5 = self.k_values[0] if len(self.k_values) > 0 else 5
            k10 = self.k_values[1] if len(self.k_values) > 1 else 10
            h5 = hit_rate_at_k(ret, rel, k=k5)
            h10 = hit_rate_at_k(ret, rel, k=k10)
            n5 = ndcg_at_k(ret, scores, k=k5)
            n10 = ndcg_at_k(ret, scores, k=k10)
            mrr = reciprocal_rank(ret, rel)
            lin_cov = lineage_coverage(ret, lin)
            # diversity: 1 - avg pairwise jaccard of token_lists
            if len(toks) >= 2:
                sims = [
                    jaccard_similarity(toks[i], toks[j])
                    for i in range(len(toks))
                    for j in range(i + 1, len(toks))
                ]
                div = 1.0 - (sum(sims) / len(sims))
            else:
                div = 1.0
            per_query.append(QueryResult(
                query=q, retrieved_ids=list(ret), relevant_ids=list(rel),
                relevance_scores=dict(scores), lineage_ids=list(lin),
                hit_at_5=h5, hit_at_10=h10, ndcg_at_5=n5, ndcg_at_10=n10,
                mrr=mrr, lineage_cov=lin_cov, diversity=div,
            ))

        # 汇总
        h5_mean = mean_hit_rate_at_k(retrieved, relevant, k=self.k_values[0])
        h10_mean = mean_hit_rate_at_k(
            retrieved, relevant, k=self.k_values[1] if len(self.k_values) > 1 else 10
        )
        n5_mean = mean_ndcg_at_k(retrieved, relevance_scores, k=self.k_values[0])
        n10_mean = mean_ndcg_at_k(
            retrieved, relevance_scores, k=self.k_values[1] if len(self.k_values) > 1 else 10
        )
        mrr_mean = mean_reciprocal_rank(retrieved, relevant)
        lin_mean = mean_lineage_coverage(retrieved, lineage_ids)
        div_mean = intra_list_diversity(token_lists)

        return EvalReport(
            n_queries=n,
            hit_at_5=h5_mean, hit_at_10=h10_mean,
            ndcg_at_5=n5_mean, ndcg_at_10=n10_mean,
            mrr=mrr_mean,
            lineage_coverage=lin_mean,
            diversity=div_mean,
            per_query=per_query,
        )

    def save(self, report: EvalReport, path: Path | str) -> None:
        """保存为 JSON。"""
        path = Path(path)
        ensure_parent(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    def save_csv(self, report: EvalReport, path: Path | str) -> None:
        """保存 per-query 为 CSV。"""
        path = Path(path)
        ensure_parent(path)
        rows = [asdict(q) for q in report.per_query]
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
