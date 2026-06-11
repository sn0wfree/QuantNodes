"""Knowledge RAG — 因子知识库 + RAG prompt + 谱系展开 (Week 8)
+ 谱系压缩 (Week 9) + 评估指标 (Week 10)。

公开 API:
    - BaseRetriever (Protocol)
    - TFIDFRetriever (sklearn 实现)
    - IdentityRetriever (纯 Python fallback / 测试)
    - make_retriever(kind)
    - KnowledgeBase
    - build_rag_prompt() (含谱系 RAG + 压缩)
    - expand_lineage() / expand_lineage_batch() (Week 8)
    - Compressor / compress_lineage() (Week 9)
    - 5 个评估指标 + RAGEvaluator (Week 10)
"""
from .retriever import (
    BaseRetriever,
    IdentityRetriever,
    TFIDFRetriever,
    make_retriever,
)
from .knowledge_base import KnowledgeBase
from .lineage_compress import Compressor, CompressedLineage, compress_lineage
from .lineage_expand import expand_lineage, expand_lineage_batch
from .rag_prompt import build_rag_prompt
from .metrics import (
    EvalReport,
    QueryResult,
    RAGEvaluator,
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

__all__ = [
    "BaseRetriever",
    "TFIDFRetriever",
    "IdentityRetriever",
    "make_retriever",
    "KnowledgeBase",
    "expand_lineage",
    "expand_lineage_batch",
    "Compressor",
    "CompressedLineage",
    "compress_lineage",
    "build_rag_prompt",
    # Week 10 评估指标
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
