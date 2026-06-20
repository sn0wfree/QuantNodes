# coding=utf-8
"""
缓存元数据管理

管理缓存文件的元数据 (创建时间、过期时间、行数、日期范围等)。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List


@dataclass
class CacheMeta:
    """缓存元数据"""
    table: str = ""
    cache_key: str = ""
    created_at: str = ""          # ISO 格式
    last_accessed: str = ""       # ISO 格式
    ttl_days: int = 7
    row_count: int = 0
    columns: List[str] = field(default_factory=list)
    date_range: List[str] = field(default_factory=list)  # [start, end]
    source: str = ""
    query_filter: str = ""


class CacheMetadata:
    """缓存元数据管理器"""

    META_FILENAME = "metadata.json"

    def _get_meta_path(self, table_dir: Path) -> Path:
        return table_dir / self.META_FILENAME

    def load(self, table_dir: Path) -> Optional[CacheMeta]:
        """加载元数据, 不存在或损坏返回 None"""
        path = self._get_meta_path(table_dir)
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            valid_keys = CacheMeta.__dataclass_fields__
            filtered = {k: v for k, v in data.items() if k in valid_keys}
            return CacheMeta(**filtered)
        except Exception:
            return None

    def save(self, table_dir: Path, meta: CacheMeta) -> None:
        """保存元数据"""
        table_dir.mkdir(parents=True, exist_ok=True)
        path = self._get_meta_path(table_dir)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(meta), f, indent=2, ensure_ascii=False)

    def is_expired(self, meta: CacheMeta) -> bool:
        """检查缓存是否过期"""
        if not meta.created_at:
            return True
        try:
            created = datetime.fromisoformat(meta.created_at)
            return datetime.now() > created + timedelta(days=meta.ttl_days)
        except Exception:
            return True

    def touch(self, meta: CacheMeta) -> None:
        """更新最后访问时间"""
        meta.last_accessed = datetime.now().isoformat()

    def create(
        self,
        table: str,
        cache_key: str,
        source: str,
        query_filter: str,
        ttl_days: int,
        row_count: int,
        columns: List[str],
        date_range: List[str],
    ) -> CacheMeta:
        """创建新的元数据"""
        now = datetime.now().isoformat()
        return CacheMeta(
            table=table,
            cache_key=cache_key,
            created_at=now,
            last_accessed=now,
            ttl_days=ttl_days,
            row_count=row_count,
            columns=columns,
            date_range=date_range,
            source=source,
            query_filter=query_filter,
        )
