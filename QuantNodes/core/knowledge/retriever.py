"""Retriever — 因子知识库检索抽象。

抽象:
    BaseRetriever: interface
    TFIDFRetriever: 基于 sklearn TfidfVectorizer (默认实现)
    IdentityRetriever: 无 embedding, 仅按文本匹配 (fallback / 测试)
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol, runtime_checkable


@runtime_checkable
class BaseRetriever(Protocol):
    """Retriever 协议。

    方法:
        add(doc_id, text) -> None: 添加文档
        query(text, top_k) -> list[(doc_id, score)]: 检索 top-k, 降序
        save(path) / load(path): 可选持久化
    """

    def add(self, doc_id: str, text: str) -> None: ...
    def query(self, text: str, top_k: int = 5) -> list[tuple[str, float]]: ...


# ============================================================================
# TFIDFRetriever (sklearn)
# ============================================================================

class TFIDFRetriever:
    """基于 sklearn TfidfVectorizer 的稀疏检索器。

    Args:
        token_pattern: sklearn 分词正则 (默认支持 alnum + 下划线)

    v3.0.0 graceful degradation: if ``sklearn`` is not installed,
    ``TFIDFRetriever`` falls back to :class:`IdentityRetriever`
    (pure-Python TF-IDF) instead of raising ``ModuleNotFoundError``.
    A ``RuntimeWarning`` is emitted so users can install sklearn
    explicitly: ``pip install scikit-learn``.
    """

    def __init__(self, token_pattern: str | None = None):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                token_pattern=token_pattern or r"(?u)\b\w+\b",
                lowercase=True,
            )
            self._sklearn = True
        except ImportError:
            import warnings
            warnings.warn(
                "scikit-learn not installed; TFIDFRetriever falling back to "
                "IdentityRetriever (pure-Python). Install scikit-learn for "
                "the full sklearn-backed retrieval: pip install scikit-learn",
                RuntimeWarning,
                stacklevel=2,
            )
            # Delegate to IdentityRetriever (we'll forward add/query through
            # the same interface by storing an internal instance)
            self._fallback = IdentityRetriever()
            self._sklearn = False
            self._vectorizer = None
            self._doc_ids = None  # sentinel; not used in fallback mode
            self._matrix = None
            return

        self._doc_ids: list[str] = []
        self._texts: list[str] = []
        self._matrix = None  # 懒构建

    def add(self, doc_id: str, text: str) -> None:
        """添加一条文档, 标记 matrix 失效。"""
        if not self._sklearn:
            # Fallback to IdentityRetriever (pure-Python TF-IDF)
            return self._fallback.add(doc_id, text)
        if doc_id in self._doc_ids:
            # 更新: 移除旧的
            idx = self._doc_ids.index(doc_id)
            self._doc_ids.pop(idx)
            self._texts.pop(idx)
        self._doc_ids.append(doc_id)
        self._texts.append(text)
        self._matrix = None

    def _ensure_matrix(self):
        if self._matrix is None and self._texts:
            self._matrix = self._vectorizer.fit_transform(self._texts)

    def query(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        """返回 top_k 文档, 列表 [(doc_id, score)] 降序。"""
        if not self._sklearn:
            return self._fallback.query(text, top_k)
        if not self._doc_ids:
            return []
        self._ensure_matrix()
        q_vec = self._vectorizer.transform([text])
        # cosine = (q · d) / (||q|| * ||d||), TfidfVectorizer 已 L2 normalize
        scores = (self._matrix @ q_vec.T).toarray().ravel()
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True,
        )[:top_k]
        return [(self._doc_ids[i], float(s)) for i, s in ranked if s > 0]

    def __len__(self) -> int:
        if not self._sklearn:
            return len(self._fallback)
        return len(self._doc_ids)


# ============================================================================
# IdentityRetriever (纯 Python, 无 sklearn 依赖)
# ============================================================================

_WORD_RE = re.compile(r"(?u)\b\w+\b")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def _compute_idf(docs: list[list[str]]) -> dict[str, float]:
    """IDF = log(N / df) + 1, 平滑版。"""
    n = len(docs)
    df: Counter = Counter()
    for tokens in docs:
        for unique in set(tokens):
            df[unique] += 1
    return {term: math.log((n + 1) / (df_t + 1)) + 1 for term, df_t in df.items()}


class IdentityRetriever:
    """纯 Python TF-IDF (无 sklearn 依赖), 用于测试和小规模场景。

    实现: 词频 × IDF, cosine 相似度。
    """

    def __init__(self):
        self._doc_ids: list[str] = []
        self._tokenized: list[list[str]] = []
        self._idf: dict[str, float] = {}
        self._norms: list[float] = []
        self._dirty = True

    def add(self, doc_id: str, text: str) -> None:
        if doc_id in self._doc_ids:
            idx = self._doc_ids.index(doc_id)
            self._doc_ids.pop(idx)
            self._tokenized.pop(idx)
        self._doc_ids.append(doc_id)
        self._tokenized.append(_tokenize(text))
        self._dirty = True

    def _rebuild(self) -> None:
        self._idf = _compute_idf(self._tokenized)
        self._norms = []
        for tokens in self._tokenized:
            tf = Counter(tokens)
            vec = {t: (tf[t] / max(len(tokens), 1)) * self._idf.get(t, 0.0)
                   for t in tokens}
            self._norms.append(math.sqrt(sum(v * v for v in vec.values())) or 1e-9)
        self._dirty = False

    def query(self, text: str, top_k: int = 5) -> list[tuple[str, float]]:
        if not self._doc_ids:
            return []
        if self._dirty:
            self._rebuild()
        q_tokens = _tokenize(text)
        q_tf = Counter(q_tokens)
        q_vec = {t: (q_tf[t] / max(len(q_tokens), 1)) * self._idf.get(t, 0.0)
                 for t in q_tokens}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1e-9
        scores: list[tuple[int, float]] = []
        for i, tokens in enumerate(self._tokenized):
            tf = Counter(tokens)
            d_vec = {t: (tf[t] / max(len(tokens), 1)) * self._idf.get(t, 0.0)
                     for t in tokens}
            dot = sum(q_vec.get(t, 0) * d_vec.get(t, 0) for t in q_vec)
            sim = dot / (q_norm * self._norms[i])
            if sim > 0:
                scores.append((i, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [(self._doc_ids[i], float(s)) for i, s in scores[:top_k]]

    def __len__(self) -> int:
        return len(self._doc_ids)


# ============================================================================
# Factory
# ============================================================================

def make_retriever(kind: str = "tfidf") -> BaseRetriever:
    """构造 retriever。

    Args:
        kind: "tfidf" (sklearn, 优先) / "identity" (纯 Python)
    """
    if kind == "tfidf":
        return TFIDFRetriever()
    if kind == "identity":
        return IdentityRetriever()
    raise ValueError(f"未知 retriever kind: {kind!r}")
