"""Knowledge RAG — 因子知识库 + RAG prompt 注入 + 谱系展开 (Week 8)。

公开 API:
    - BaseRetriever (Protocol)
    - TFIDFRetriever (sklearn 实现)
    - IdentityRetriever (纯 Python fallback / 测试)
    - make_retriever(kind)
    - KnowledgeBase
    - build_rag_prompt() (含谱系 RAG)
    - expand_lineage() / expand_lineage_batch() (Week 8)
"""
from .retriever import (
    BaseRetriever,
    IdentityRetriever,
    TFIDFRetriever,
    make_retriever,
)
from .knowledge_base import KnowledgeBase
from .lineage_expand import expand_lineage, expand_lineage_batch
from .rag_prompt import build_rag_prompt

__all__ = [
    "BaseRetriever",
    "TFIDFRetriever",
    "IdentityRetriever",
    "make_retriever",
    "KnowledgeBase",
    "expand_lineage",
    "expand_lineage_batch",
    "build_rag_prompt",
]
