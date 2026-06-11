"""Knowledge RAG — 因子知识库 + RAG prompt 注入。

公开 API:
    - BaseRetriever (Protocol)
    - TFIDFRetriever (sklearn 实现)
    - IdentityRetriever (纯 Python fallback / 测试)
    - make_retriever(kind)
    - KnowledgeBase
    - build_rag_prompt()
"""
from .retriever import (
    BaseRetriever,
    IdentityRetriever,
    TFIDFRetriever,
    make_retriever,
)
from .knowledge_base import KnowledgeBase
from .rag_prompt import build_rag_prompt

__all__ = [
    "BaseRetriever",
    "TFIDFRetriever",
    "IdentityRetriever",
    "make_retriever",
    "KnowledgeBase",
    "build_rag_prompt",
]
