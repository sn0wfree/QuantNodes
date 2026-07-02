# coding=utf-8
"""
factor_pool.py - 因子池抽象 (v3.0.2)

提供 FactorPool 类:
- 内存中的 dict[str, FactorEntry] 池
- 与 WikiFactorProxy 双向同步 (from_wiki / to_wiki)
- dedup / select / summary / save_json / load_json
- 线程安全 (add / extend / to_wiki 内部加锁)

设计动机:
- LogicDrivenPipeline / AlphaPipeline 仅返回 final_pool: List[FactorMetrics] (临时)
- 跨多次 mine 任务 / 跨 LLM 客户端都需要持久化池
- Wiki 已经是唯一持久化层, 池作为 in-mem 镜像存在

Usage::

    from QuantNodes.research.quant_alpha.factor_pool import FactorPool, FactorEntry
    from QuantNodes.research.wiki import WikiFactorProxy

    pool = FactorPool(wiki_path="wiki_auto")
    pool.from_wiki(WikiFactorProxy(wiki_path="wiki_auto"))
    pool.add(FactorEntry(formula_id="alpha101-alpha006", formula="...",
                        source_lib="alpha101", ir=1.0))
    pool.select(top_n=5, by="ir")
    n_written = pool.to_wiki(WikiFactorProxy(wiki_path="wiki_auto"))
    pool.save_json(Path("data/mine_runs/pool.json"))
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

__all__ = ["FactorEntry", "FactorPool"]


@dataclass
class FactorEntry:
    """池内单条因子记录

    Attributes:
        formula_id:    唯一键, 推荐 "{source_lib}-{source_id}" (例如 "alpha101-alpha006")
        formula:       因子公式字符串
        source_lib:    来源库 ("alpha101" / "alpha158" / "alpha191")
        source_id:     在来源库中的 ID (例如 "alpha006")
        ir:            信息比率 (Information Ratio)
        ic_mean:       IC 均值
        rank_ic:       Rank IC
        tags:          标签列表
        discovered_at: 发现时间 (ISO 8601 字符串)
        wiki_path:     对应的 Wiki page path (e.g. "Logic/alpha101-alpha006.md")
        structured:    序列化的 WikiLogicStructured (dict 形式, 可选)
        evidence:      序列化的 LogicPerformanceEvidence (dict 形式, 可选)
    """

    formula_id: str
    formula: str
    source_lib: str = ""
    source_id: str = ""
    ir: float = 0.0
    ic_mean: float = 0.0
    rank_ic: float = 0.0
    tags: List[str] = field(default_factory=list)
    discovered_at: str = ""
    wiki_path: Optional[str] = None
    structured: Optional[Dict[str, Any]] = None
    evidence: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.discovered_at:
            self.discovered_at = datetime.utcnow().isoformat(timespec="seconds")

    @classmethod
    def from_logic_result(
        cls,
        formula_id: str,
        formula: str,
        source_lib: str,
        source_id: str = "",
        ir: float = 0.0,
        ic_mean: float = 0.0,
        rank_ic: float = 0.0,
        tags: Optional[List[str]] = None,
        structured: Optional[Dict[str, Any]] = None,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> "FactorEntry":
        """便捷构造: 从 Logic Mining 结果拼 FactorEntry"""
        return cls(
            formula_id=formula_id,
            formula=formula,
            source_lib=source_lib,
            source_id=source_id or formula_id,
            ir=ir,
            ic_mean=ic_mean,
            rank_ic=rank_ic,
            tags=list(tags or []),
            structured=structured,
            evidence=evidence,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactorEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class FactorPool:
    """线程安全的 in-mem 因子池

    - 内部存储 Dict[formula_id, FactorEntry]
    - add / extend / dedup / select / summary 操作均线程安全
    - from_wiki / to_wiki 与 WikiFactorProxy 双向同步
    - save_json / load_json 支持离线持久化 (与 wiki 解耦)
    """

    def __init__(self, wiki_path: str = "wiki") -> None:
        self.wiki_path = wiki_path
        self._entries: Dict[str, FactorEntry] = {}
        self._lock = threading.Lock()
        self._failed_writes: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # 基础 CRUD
    # ------------------------------------------------------------------
    def add(self, entry: FactorEntry) -> bool:
        """添加一条; 已存在则覆盖; 返回 True=新增, False=覆盖"""
        with self._lock:
            existed = entry.formula_id in self._entries
            self._entries[entry.formula_id] = entry
            return not existed

    def extend(self, entries: Iterable[FactorEntry]) -> int:
        """批量添加; 返回新增 (不含覆盖) 数量"""
        added = 0
        with self._lock:
            for e in entries:
                if e.formula_id not in self._entries:
                    self._entries[e.formula_id] = e
                    added += 1
                else:
                    self._entries[e.formula_id] = e
        return added

    def remove(self, formula_id: str) -> bool:
        with self._lock:
            return self._entries.pop(formula_id, None) is not None

    def get(self, formula_id: str) -> Optional[FactorEntry]:
        with self._lock:
            return self._entries.get(formula_id)

    def contains(self, formula_id: str) -> bool:
        with self._lock:
            return formula_id in self._entries

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._failed_writes.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __iter__(self):
        with self._lock:
            return iter(list(self._entries.values()))

    def __contains__(self, formula_id: str) -> bool:
        return self.contains(formula_id)

    def keys(self) -> List[str]:
        with self._lock:
            return list(self._entries.keys())

    def values(self) -> List[FactorEntry]:
        with self._lock:
            return list(self._entries.values())

    # ------------------------------------------------------------------
    # dedup / select / summary
    # ------------------------------------------------------------------
    def dedup(self, by: str = "formula_id") -> int:
        """去重; 返回被移除的数量

        Args:
            by: "formula_id" (默认, 直接按主键) /
                "formula"   (按公式字符串, 保留 IR 较高者) /
                "source_id" (按 (source_lib, source_id) 元组)
        """
        if by == "formula_id":
            return 0  # 主键天然唯一

        with self._lock:
            if by == "formula":
                seen: Dict[str, FactorEntry] = {}
                for fid, e in self._entries.items():
                    if e.formula not in seen or e.ir > seen[e.formula].ir:
                        seen[e.formula] = e
                removed = len(self._entries) - len(seen)
                self._entries = {e.formula_id: e for e in seen.values()}
                return removed

            if by == "source_id":
                seen2: Dict[tuple, FactorEntry] = {}
                for fid, e in self._entries.items():
                    key = (e.source_lib, e.source_id)
                    if key not in seen2 or e.ir > seen2[key].ir:
                        seen2[key] = e
                removed = len(self._entries) - len(seen2)
                self._entries = {e.formula_id: e for e in seen2.values()}
                return removed

            raise ValueError(f"Unknown dedup key: {by!r}")

    def select(self, top_n: int = 10, by: str = "ir") -> List[FactorEntry]:
        """按 IR (默认) / ic_mean / rank_ic 排序, 取前 top_n"""
        if top_n <= 0:
            return []
        with self._lock:
            entries = list(self._entries.values())
        key_fn = {
            "ir": lambda e: e.ir,
            "ic_mean": lambda e: e.ic_mean,
            "rank_ic": lambda e: e.rank_ic,
        }.get(by, lambda e: e.ir)
        entries.sort(key=key_fn, reverse=True)
        return entries[:top_n]

    def filter(
        self,
        *,
        source_lib: Optional[str] = None,
        min_ir: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> List[FactorEntry]:
        """按条件过滤"""
        with self._lock:
            entries = list(self._entries.values())
        out = []
        for e in entries:
            if source_lib and e.source_lib != source_lib:
                continue
            if min_ir is not None and e.ir < min_ir:
                continue
            if tags and not any(t in e.tags for t in tags):
                continue
            out.append(e)
        return out

    def summary(self) -> Dict[str, Any]:
        """池整体统计"""
        with self._lock:
            entries = list(self._entries.values())
        if not entries:
            return {
                "n_total": 0,
                "by_source_lib": {},
                "ir_stats": {},
                "n_with_wiki": 0,
            }
        irs = [e.ir for e in entries]
        by_lib: Dict[str, int] = {}
        for e in entries:
            by_lib[e.source_lib] = by_lib.get(e.source_lib, 0) + 1
        irs_sorted = sorted(irs)
        n = len(irs_sorted)
        median = (
            irs_sorted[n // 2]
            if n % 2 == 1
            else (irs_sorted[n // 2 - 1] + irs_sorted[n // 2]) / 2.0
        )
        return {
            "n_total": n,
            "by_source_lib": by_lib,
            "ir_stats": {
                "min": min(irs),
                "max": max(irs),
                "mean": sum(irs) / n,
                "median": median,
            },
            "n_with_wiki": sum(1 for e in entries if e.wiki_path),
        }

    # ------------------------------------------------------------------
    # Wiki 双向同步
    # ------------------------------------------------------------------
    def from_wiki(self, proxy: Any) -> int:
        """从 Wiki 读取 Logic 页 → 池; 返回加载数量

        Args:
            proxy: WikiFactorProxy 实例
        """
        try:
            existing_logics = proxy.list_logics(limit=10_000)
        except Exception as exc:
            logger.warning("from_wiki: list_logics failed: %s", exc)
            return 0

        loaded = 0
        with self._lock:
            for logic in existing_logics:
                structured_dict = None
                if getattr(logic, "structured", None) is not None:
                    try:
                        structured_dict = logic.structured.to_dict()
                    except Exception:
                        structured_dict = None

                evidence_dict = None
                if getattr(logic, "performance_evidence", None) is not None:
                    try:
                        evidence_dict = logic.performance_evidence.to_dict()
                    except Exception:
                        evidence_dict = None

                src_detail = getattr(logic, "source_detail", None) or {}
                entry = FactorEntry(
                    formula_id=logic.name,
                    formula=getattr(logic, "extracted_formula", "") or "",
                    source_lib=src_detail.get("source_lib", "wiki") if isinstance(src_detail, dict) else "wiki",
                    source_id=logic.name,
                    ir=(evidence_dict or {}).get("best_ir", 0.0),
                    ic_mean=(evidence_dict or {}).get("best_ic", 0.0),
                    rank_ic=0.0,
                    tags=list(src_detail.get("tags", []) if isinstance(src_detail, dict) else []),
                    discovered_at=getattr(logic, "created_at", "") or "",
                    wiki_path=getattr(logic, "wiki_page_name", None),
                    structured=structured_dict,
                    evidence=evidence_dict,
                )
                self._entries[entry.formula_id] = entry
                loaded += 1
        return loaded

    def to_wiki(self, proxy: Any) -> int:
        """把池内条目写入 Wiki Logic 页; 返回成功数量

        单条失败不中断, 记录到 self._failed_writes
        """
        from datetime import datetime as _dt

        from QuantNodes.research.wiki import LogicSource, WikiLogic

        with self._lock:
            entries = list(self._entries.values())

        written = 0
        for entry in entries:
            try:
                logic = WikiLogic(
                    name=entry.formula_id,
                    content=(
                        f"# Logic: {entry.formula_id}\n\n"
                        f"**Source formula**: `{entry.formula}`\n\n"
                        f"**Source library**: `{entry.source_lib}`\n\n"
                        f"**IR**: {entry.ir:.4f}  **IC**: {entry.ic_mean:.4f}  "
                        f"**Rank IC**: {entry.rank_ic:.4f}\n\n"
                        f"**Discovered at**: {entry.discovered_at}\n\n"
                        f"**Tags**: {', '.join(entry.tags) or '(none)'}\n"
                    ),
                    source=LogicSource.RESEARCH_REPORT,
                    extracted_formula=entry.formula,
                    source_detail={
                        "source_lib": entry.source_lib,
                        "source_id": entry.source_id,
                        "tags": ",".join(entry.tags),
                    },
                    validation_status="pending",
                    structured=_load_structured(entry.structured),
                    performance_evidence=_load_evidence(entry.evidence),
                    refinement_round=(entry.evidence or {}).get("refinement_round", 0),
                    created_at=entry.discovered_at or _dt.utcnow().isoformat(timespec="seconds"),
                )
                page = proxy.store_logic(logic)
                with self._lock:
                    entry.wiki_path = page
                written += 1
            except Exception as exc:
                with self._lock:
                    self._failed_writes.append(
                        {
                            "formula_id": entry.formula_id,
                            "error": repr(exc),
                        }
                    )
                logger.warning("to_wiki: failed for %s: %s", entry.formula_id, exc)
        return written

    def failed_writes(self) -> List[Dict[str, str]]:
        """获取写入失败的条目列表 (快照)"""
        with self._lock:
            return list(self._failed_writes)

    # ------------------------------------------------------------------
    # JSON 离线持久化
    # ------------------------------------------------------------------
    def save_json(self, path: Path) -> None:
        """序列化池到 JSON 文件"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = {
                "version": 1,
                "wiki_path": self.wiki_path,
                "saved_at": datetime.utcnow().isoformat(timespec="seconds"),
                "n_entries": len(self._entries),
                "entries": [e.to_dict() for e in self._entries.values()],
                "failed_writes": list(self._failed_writes),
            }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: Path) -> "FactorPool":
        """从 JSON 文件恢复池"""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        pool = cls(wiki_path=data.get("wiki_path", "wiki"))
        for ed in data.get("entries", []):
            pool.add(FactorEntry.from_dict(ed))
        with pool._lock:
            pool._failed_writes.extend(data.get("failed_writes", []))
        return pool


def _load_structured(d: Optional[Dict[str, Any]]) -> Any:
    """从 dict 还原 WikiLogicStructured 对象 (用于 wiki 写入)"""
    if not d:
        return None
    try:
        from QuantNodes.research.quant_alpha.logic_mining.models import WikiLogicStructured

        return WikiLogicStructured.from_dict(d)
    except Exception:
        return None


def _load_evidence(d: Optional[Dict[str, Any]]) -> Any:
    """从 dict 还原 LogicPerformanceEvidence 对象"""
    if not d:
        return None
    try:
        from QuantNodes.research.quant_alpha.logic_mining.models import (
            LogicPerformanceEvidence,
        )

        return LogicPerformanceEvidence.from_dict(d)
    except Exception:
        return None