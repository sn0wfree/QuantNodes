"""TrajectoryEntry — 单条演化轨迹。"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd

from ..constants import METRIC_KEYS as _METRIC_KEYS
from ..feedback import FactorFeedback


class Operation(str, Enum):
    """M1: 演化操作类型 — 替代硬编码字符串 'original'/'mutation'/'crossover'。

    str Enum 同时兼容 dataclass.field(default="original") 旧用法。
    """
    ORIGINAL = "original"
    MUTATION = "mutation"
    CROSSOVER = "crossover"


@dataclass
class TrajectoryEntry:
    """单条演化轨迹 — QuantaAlpha `Trace.hist` 等价物。"""

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    round_idx: int = 0
    operation: str = Operation.ORIGINAL  # M1: enum 默认值, 仍兼容 str
    config_snapshot: dict = field(default_factory=dict)
    context_subset: dict = field(default_factory=dict)
    feedback: FactorFeedback | None = None
    parent_ids: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    # ------------------------------------------------------------------
    # Parquet 序列化 (元数据)
    # ------------------------------------------------------------------

    def to_parquet_row(self) -> dict:
        """展平为单行 dict 供 Parquet 写入。"""
        row = {
            "entry_id": self.entry_id,
            "round_idx": int(self.round_idx),
            "operation": str(self.operation),
            "parent_ids": ",".join(self.parent_ids),
            "decision": bool(self.feedback.decision) if self.feedback else False,
            "duration_ms": float(self.feedback.duration_ms) if self.feedback else 0.0,
            "timestamp": self.timestamp.isoformat(),
            "factor_name": str(self.feedback.factor_name) if self.feedback else "",
            "summary": str(self.feedback.summary) if self.feedback else "",
        }
        for key in _METRIC_KEYS:
            row[key] = self.metrics.get(key)
        return row

    # ------------------------------------------------------------------
    # JSON 序列化 (完整记录)
    # ------------------------------------------------------------------

    def to_json_dict(self) -> dict:
        """转为 JSON-safe dict。"""
        return {
            "entry_id": self.entry_id,
            "round_idx": self.round_idx,
            "operation": self.operation,
            "config_snapshot": _jsonify(self.config_snapshot),
            "context_subset": _jsonify(self.context_subset),
            "feedback": self.feedback.to_dict() if self.feedback else None,
            "parent_ids": list(self.parent_ids),
            "metrics": _jsonify(self.metrics),
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_json_dict(cls, d: dict) -> "TrajectoryEntry":
        """从 JSON dict 还原。"""
        return cls(
            entry_id=d["entry_id"],
            round_idx=int(d.get("round_idx", 0)),
            operation=str(d.get("operation", "original")),
            config_snapshot=d.get("config_snapshot", {}),
            context_subset=d.get("context_subset", {}),
            feedback=(
                FactorFeedback.from_dict(d["feedback"])
                if d.get("feedback") is not None
                else None
            ),
            parent_ids=list(d.get("parent_ids", [])),
            metrics=d.get("metrics", {}),
            timestamp=datetime.fromisoformat(d["timestamp"]),
        )


def _jsonify(obj: Any) -> Any:
    """把 pd.Timestamp / np.ndarray / datetime 等转为 JSON-safe 类型。"""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
