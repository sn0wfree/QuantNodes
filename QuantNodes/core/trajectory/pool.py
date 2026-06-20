"""TrajectoryPool — 演化轨迹池, 双层 Parquet + JSON 持久化。"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterator

import pandas as pd

from ..constants import PARQUET_COLUMNS as _PARQUET_COLUMNS
from .entry import TrajectoryEntry
from .lineage import children_of, descendants, lineage
from QuantNodes.core.path_utils import ensure_dir


_PARQUET_NAME = "trajectories.parquet"
_ENTRIES_SUBDIR = "entries"


class TrajectoryPool:
    """演化轨迹池 — 持久化每轮实验的完整记录。

    双层存储:
        - Layer 1: {parquet_name}.parquet (元数据, append)
        - Layer 2: entries/{entry_id}.json (完整记录, 独立子目录)

    Phase H2 (2026-06-20):
      - all()/_sorted_cache memoizes the timestamp-sorted list and
        invalidates only on add() or reset(). Avoids repeated sorts when
        many callers iterate the pool in tight loops (selector, loop,
        cli/commands/factor.py, knowledge_base.py).
      - _load() uses df.to_dict("records") instead of df.iterrows()
        (5-10x faster bulk load on startup with large pools).

    H1 (parquet O(n^2) append) deferred: ParquetWriter requires strict
    schema uniformity per column, but the existing Parquet file holds
    heterogeneous types (int/float/bool/str/list per column) and tests
    assert type preservation. Re-implementing with native types would
    require inferring the schema from existing data on every open.

    Args:
        base_dir: 池根目录 (自动创建)
        parquet_name: Parquet 文件名 (默认 "trajectories.parquet"),
                       允许不同实验共用 base_dir 各自存盘。
    """

    def __init__(
        self,
        base_dir: Path | str,
        parquet_name: str = _PARQUET_NAME,
    ):
        self.base_dir = Path(base_dir)
        ensure_dir(self.base_dir)
        self._entries_dir = self.base_dir / _ENTRIES_SUBDIR
        ensure_dir(self._entries_dir)
        self._parquet_name = parquet_name
        self._entries: dict[str, TrajectoryEntry] = {}
        self._parquet_path = self.base_dir / self._parquet_name
        self._lock = threading.Lock()
        self._sorted_cache: list[TrajectoryEntry] | None = None
        self._load()

    def add(self, entry: TrajectoryEntry) -> None:
        """添加一条 entry, 自动持久化。"""
        with self._lock:
            self._entries[entry.entry_id] = entry
            self._persist(entry)
            self._sorted_cache = None  # invalidate cache (H2)

    def get(self, entry_id: str) -> TrajectoryEntry:
        """按 ID 获取 entry, 不存在抛 KeyError。"""
        if entry_id not in self._entries:
            raise KeyError(f"entry_id 不存在: {entry_id}")
        return self._entries[entry_id]

    def all(self) -> list[TrajectoryEntry]:
        """返回所有 entry 列表 (按时间排序)。

        Cached: subsequent calls return the same list until add()/reset()
        invalidates the cache.
        """
        if self._sorted_cache is None:
            self._sorted_cache = sorted(self._entries.values(), key=lambda e: e.timestamp)
        return list(self._sorted_cache)

    def __iter__(self) -> Iterator[TrajectoryEntry]:
        return iter(self.all())

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def size(self) -> int:
        return len(self._entries)

    def reset(self) -> None:
        """清空内存 + 删除所有持久化文件 (entries/ 子目录 + Parquet)。"""
        with self._lock:
            self._entries.clear()
            self._sorted_cache = None
            if self._parquet_path.exists():
                self._parquet_path.unlink()
            if self._entries_dir.exists():
                for p in self._entries_dir.glob("*.json"):
                    p.unlink()

    # ------------------------------------------------------------------
    # 过滤
    # ------------------------------------------------------------------

    def by_round(self, round_idx: int) -> list[TrajectoryEntry]:
        return [e for e in self.all() if e.round_idx == round_idx]

    def by_operation(self, operation: str) -> list[TrajectoryEntry]:
        return [e for e in self.all() if e.operation == operation]

    def filter(self, decision: bool | None = None) -> list[TrajectoryEntry]:
        """按 feedback.decision 过滤 (None=不过滤)。"""
        if decision is None:
            return self.all()
        return [
            e for e in self.all()
            if e.feedback is not None and e.feedback.decision == decision
        ]

    def best(self, top_n: int = 5, metric: str = "sharpe") -> list[TrajectoryEntry]:
        """按 metric 降序, 返回 Top-N。"""
        return sorted(
            self.all(),
            key=lambda e: float(e.metrics.get(metric, 0) or 0),
            reverse=True,
        )[:top_n]

    def random(self, n: int, seed: int | None = None) -> list[TrajectoryEntry]:
        """从池中随机抽 n 条。"""
        import numpy as np
        rng = np.random.default_rng(seed)
        k = min(n, len(self._entries))
        if k == 0:
            return []
        indices = rng.choice(len(self._entries), size=k, replace=False)
        all_entries = self.all()
        return [all_entries[int(i)] for i in indices]

    # ------------------------------------------------------------------
    # 谱系 (代理到 lineage.py)
    # ------------------------------------------------------------------

    def children_of(self, parent_id: str) -> list[TrajectoryEntry]:
        return children_of(self._entries, parent_id)

    def lineage(self, entry_id: str) -> list[TrajectoryEntry]:
        return lineage(self._entries, entry_id)

    def descendants(self, entry_id: str, max_depth: int | None = None) -> list[TrajectoryEntry]:
        return descendants(self._entries, entry_id, max_depth=max_depth)

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _persist(self, entry: TrajectoryEntry) -> None:
        """持久化单条 entry (Parquet append + JSON 单文件)。

        保留原有 read+concat+rewrite 路径 (O(n^2) 但 schema 兼容)。
        H1 的 ParquetWriter 增量追加方案因 schema 类型一致性限制暂未启用,
        详见模块 docstring。H2 的 _sorted_cache 与 _load 优化已生效。
        """
        new_row = pd.DataFrame([entry.to_parquet_row()], columns=list(_PARQUET_COLUMNS))
        if self._parquet_path.exists():
            existing = pd.read_parquet(self._parquet_path)
            for col in _PARQUET_COLUMNS:
                if col not in existing.columns:
                    existing[col] = None
            existing = existing[list(_PARQUET_COLUMNS)].astype(object)
            new_row = new_row.astype(object)
            combined = pd.concat([existing, new_row], ignore_index=True)
        else:
            combined = new_row
        combined.to_parquet(self._parquet_path, index=False)

        json_path = self._entries_dir / f"{entry.entry_id}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(entry.to_json_dict(), f, ensure_ascii=False, indent=2)

    def _load(self) -> None:
        """启动时从磁盘加载元数据 + JSON 详情。

        H2: vectorized row->dict via to_dict('records') instead of iterrows.
        """
        if not self._parquet_path.exists():
            return
        try:
            df = pd.read_parquet(self._parquet_path)
        except Exception:
            return
        records = df.to_dict("records")
        for row in records:
            entry_id = str(row.get("entry_id", ""))
            if not entry_id:
                continue
            json_path = self._entries_dir / f"{entry_id}.json"
            if not json_path.exists():
                continue
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self._entries[entry_id] = TrajectoryEntry.from_json_dict(data)
            except Exception:
                continue
