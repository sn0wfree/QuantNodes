"""KnowledgeBase — 因子知识库 (TrajectoryEntry + Retriever)。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd
from pydantic import BaseModel, Field

from ..trajectory import TrajectoryEntry, TrajectoryPool
from .retriever import BaseRetriever, make_retriever


# H19: 默认字段权重 (外部可覆盖)
DEFAULT_FIELD_WEIGHTS: dict[str, float] = {
    "name": 3.0,
    "expression": 2.0,
    "hypothesis": 2.5,
    "description": 2.0,
    "summary": 1.0,
}


class KnowledgeBaseSetting(BaseModel):
    """H19: KnowledgeBase 字段权重配置 (Pydantic)。"""
    field_weights: dict[str, float] = Field(
        default_factory=lambda: dict(DEFAULT_FIELD_WEIGHTS),
        description="字段权重: 控制检索文本构造时各字段的重复次数",
    )


class KnowledgeBase:
    """因子知识库。

    Args:
        retriever: BaseRetriever 实现 (默认 TFIDF)
        pool: 可选 TrajectoryPool 同步源
        setting: H19 字段权重配置 (None=默认)
        field_weights: H19 字段权重 dict (None=默认, 兼容旧 API)
    """

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        pool: Optional[TrajectoryPool] = None,
        setting: Optional[KnowledgeBaseSetting] = None,
        field_weights: Optional[dict[str, float]] = None,
    ):
        self.retriever = retriever or make_retriever("tfidf")
        self.pool = pool
        self._entry_ids: set[str] = set()  # 已索引的 entry_id
        # H19: 字段权重 — setting 优先, field_weights 其次, 默认最后
        if setting is not None:
            self._field_weights = dict(setting.field_weights)
        elif field_weights is not None:
            self._field_weights = dict(field_weights)
        else:
            self._field_weights = dict(DEFAULT_FIELD_WEIGHTS)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, entry: TrajectoryEntry) -> None:
        """添加 entry 到知识库 (含 retriever indexing)。"""
        text = self._entry_to_text(entry)
        self.retriever.add(entry.entry_id, text)
        self._entry_ids.add(entry.entry_id)

    def add_many(self, entries) -> int:
        """批量添加, 返回新加入数。"""
        n = 0
        for e in entries:
            if e.entry_id not in self._entry_ids:
                self.add(e)
                n += 1
        return n

    def sync_from_pool(self) -> int:
        """从 pool 同步所有未索引 entry。"""
        if self.pool is None:
            return 0
        return self.add_many(self.pool.all())

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[tuple[TrajectoryEntry, float]]:
        """返回 top_k 检索结果, 包含 entry 对象 + 相似度。

        排序: score 降序
        """
        results = self.retriever.query(text, top_k=top_k)
        if not self.pool:
            return [(None, score) for _, score in results]
        out: list[tuple[TrajectoryEntry, float]] = []
        for doc_id, score in results:
            if score < min_score:
                continue
            try:
                entry = self.pool.get(doc_id)
            except KeyError:
                continue
            out.append((entry, score))
        return out

    def query_with_lineage(
        self,
        text: str,
        top_k: int = 5,
        max_ancestor_depth: int = 2,
        max_descendant_depth: int = 2,
    ) -> list[dict[str, Any]]:
        """检索 + 附加谱系 (parents + children, 可配深度) 上下文。

        Args:
            text: 查询文本
            top_k: 检索数量
            max_ancestor_depth: 祖先深度 (1=parent, 2=grandparent, ...)
            max_descendant_depth: 后裔深度
        """
        results = self.query(text, top_k=top_k)
        out = []
        for entry, score in results:
            ctx = {
                "entry": entry,
                "score": score,
                "ancestors": [],
                "descendants": [],
                "parents": [],
                "children": [],
            }
            if self.pool and entry is not None:
                from .lineage_expand import expand_lineage
                expanded = expand_lineage(
                    self.pool, entry.entry_id,
                    max_ancestor_depth=max_ancestor_depth,
                    max_descendant_depth=max_descendant_depth,
                )
                ctx["ancestors"] = [e for _, e in expanded["ancestors"]]
                ctx["descendants"] = [e for _, e in expanded["descendants"]]
                # 兼容旧字段 (depth=1)
                ctx["parents"] = [e for d, e in expanded["ancestors"] if d == 1]
                ctx["children"] = [e for d, e in expanded["descendants"] if d == 1]
            out.append(ctx)
        return out

    # ------------------------------------------------------------------
    # 持久化 (Parquet 索引 + JSON 单文件 - 与 TrajectoryPool 复用)
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """保存知识库索引 (entry_id 列表 + retriever 文本)。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        if self.pool:
            for e in self.pool.all():
                rows.append({
                    "entry_id": e.entry_id,
                    "text": self._entry_to_text(e),
                })
        df = pd.DataFrame(rows)
        df.to_parquet(path, index=False)

    @classmethod
    def load(
        cls,
        path: Path | str,
        pool: Optional[TrajectoryPool] = None,
        retriever: Optional[BaseRetriever] = None,
    ) -> "KnowledgeBase":
        """从 Parquet 索引加载 (需配合 pool)。"""
        path = Path(path)
        df = pd.read_parquet(path)
        kb = cls(retriever=retriever, pool=pool)
        for _, row in df.iterrows():
            kb.retriever.add(str(row["entry_id"]), str(row["text"]))
            kb._entry_ids.add(str(row["entry_id"]))
        return kb

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _entry_to_text(self, entry: TrajectoryEntry) -> str:
        """把 entry 转为可检索文本 (按字段权重重复关键 token)。"""
        parts: list[str] = []
        cfg = entry.config_snapshot or {}
        factor_cfg = cfg.get("factor", {}) if isinstance(cfg, dict) else {}
        for src, key in (
            (factor_cfg.get("name", ""), "name"),
            (factor_cfg.get("expression", ""), "expression"),
            (factor_cfg.get("hypothesis", ""), "hypothesis"),
            (factor_cfg.get("description", ""), "description"),
        ):
            weight = int(self._field_weights.get(key, 1))
            parts.extend([str(src)] * weight)
        if entry.feedback:
            parts.extend(
                [str(entry.feedback.summary or "")]
                * int(self._field_weights.get("summary", 1))
            )
        if entry.metrics:
            for k, v in entry.metrics.items():
                parts.append(f"{k}={v}")
        return " ".join(parts)

    def __len__(self) -> int:
        return len(self._entry_ids)
