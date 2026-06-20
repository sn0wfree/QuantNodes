"""FactorFeedback 数据类 — 5 通道结构化反馈。

QuantaAlpha `CoSTEERSingleFeedback` 的 QuantNodes 适配版本。
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..constants import EXTENDED_METRIC_KEYS as KNOWN_METRICS
from QuantNodes.core.path_utils import ensure_parent


class FeedbackChannel(str, Enum):
    """5 通道反馈信号。"""
    EXECUTION = "execution"
    SHAPE = "shape"
    CODE = "code"
    VALUE = "value"
    LLM = "llm"


@dataclass
class ChannelFeedback:
    """单通道反馈信号。"""
    channel: FeedbackChannel
    passed: bool
    detail: str
    score: float = 1.0
    metadata: dict = field(default_factory=dict)

    def values(self):
        """供 FeedbackCollector.add() 拆解: (passed, detail, score).

        metadata 通过 ChannelFeedback 自身访问, 不在此拆解。
        """
        return (self.passed, self.detail, self.score)


@dataclass
class FactorFeedback:
    """完整因子反馈 — QuantaAlpha CoSTEERSingleFeedback 等价物。"""
    factor_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    factor_name: str = ""
    channels: dict[FeedbackChannel, ChannelFeedback] = field(default_factory=dict)
    decision: bool = False
    summary: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """序列化为可 JSON 化的字典。"""
        return {
            "factor_id": self.factor_id,
            "factor_name": self.factor_name,
            "channels": {
                ch.value: {
                    "channel": fb.channel.value,
                    "passed": fb.passed,
                    "detail": fb.detail,
                    "score": fb.score,
                    "metadata": dict(fb.metadata),
                }
                for ch, fb in self.channels.items()
            },
            "decision": self.decision,
            "summary": self.summary,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FactorFeedback":
        """从字典反序列化。"""
        return cls(
            factor_id=d["factor_id"],
            factor_name=d["factor_name"],
            channels={
                FeedbackChannel(k): ChannelFeedback(
                    channel=FeedbackChannel(v["channel"]),
                    passed=v["passed"],
                    detail=v["detail"],
                    score=v["score"],
                    metadata=v.get("metadata", {}),
                )
                for k, v in d.get("channels", {}).items()
            },
            decision=d["decision"],
            summary=d.get("summary", ""),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            duration_ms=d.get("duration_ms", 0.0),
            metadata=d.get("metadata", {}),
        )

    def save_json(self, path: Path) -> None:
        """保存为 JSON 格式 (供调试)。"""
        path = Path(path)
        ensure_parent(path)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> "FactorFeedback":
        """从 JSON 加载。"""
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_parquet_row(self) -> dict:
        """展平为单行 dict 供 Parquet 写入。"""
        row = {
            "factor_id": self.factor_id,
            "factor_name": self.factor_name,
            "decision": self.decision,
            "summary": self.summary,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "metadata": json.dumps(self.metadata, ensure_ascii=False),
        }
        for ch in FeedbackChannel:
            prefix = ch.value
            fb = self.channels.get(ch)
            row[f"{prefix}_passed"] = bool(fb.passed) if fb else None
            row[f"{prefix}_score"] = float(fb.score) if fb else None
            row[f"{prefix}_detail"] = str(fb.detail) if fb else None
        return row

    def save_parquet(self, path: Path) -> None:
        """追加到 Parquet 文件 (或创建新文件)。"""
        path = Path(path)
        ensure_parent(path)
        new_row = pd.DataFrame([self.to_parquet_row()])
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, new_row], ignore_index=True)
        else:
            combined = new_row
        combined.to_parquet(path, index=False)

    @classmethod
    def load_parquet(cls, path: Path) -> list["FactorFeedback"]:
        """从 Parquet 加载为 FactorFeedback 列表。"""
        df = pd.read_parquet(path)
        results = []
        for _, row in df.iterrows():
            channels = {}
            for ch in FeedbackChannel:
                prefix = ch.value
                if pd.notna(row.get(f"{prefix}_passed")):
                    metadata = {}
                    if prefix == "execution" and "exit_code" in row:
                        metadata["exit_code"] = int(row["exit_code"])
                    channels[ch] = ChannelFeedback(
                        channel=ch,
                        passed=bool(row[f"{prefix}_passed"]),
                        detail=str(row[f"{prefix}_detail"]),
                        score=float(row[f"{prefix}_score"]),
                        metadata=metadata,
                    )
            try:
                metadata = json.loads(row.get("metadata", "{}"))
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            results.append(cls(
                factor_id=str(row["factor_id"]),
                factor_name=str(row["factor_name"]),
                channels=channels,
                decision=bool(row["decision"]),
                summary=str(row["summary"]),
                timestamp=datetime.fromisoformat(str(row["timestamp"])),
                duration_ms=float(row["duration_ms"]),
                metadata=metadata,
            ))
        return results


def ensure_feedback(result: Any, factor_id: str, factor_name: str) -> FactorFeedback:
    """把节点返回的 dict 包装为 FactorFeedback (兼容现有节点)。

    Args:
        result: 节点返回值 (FactorFeedback / dict / 其他)
        factor_id: 因子 UUID
        factor_name: 因子名称

    Returns:
        FactorFeedback

    Raises:
        TypeError: result 类型不支持
    """
    if isinstance(result, FactorFeedback):
        if not result.factor_id:
            result.factor_id = factor_id
        if not result.factor_name:
            result.factor_name = factor_name
        return result
    if isinstance(result, dict):
        metadata = {k: _safe_scalar(result[k]) for k in KNOWN_METRICS if k in result}
        return FactorFeedback(
            factor_id=factor_id,
            factor_name=factor_name,
            decision=True,
            summary=f"dict 返回, {len(result)} 个字段",
            metadata=metadata,
        )
    raise TypeError(f"节点返回类型不支持: {type(result).__name__}")


def _safe_scalar(v: Any) -> Any:
    """转 pd.Series/np.ndarray 标量为 Python scalar。"""
    if isinstance(v, pd.Series):
        if len(v) == 0:
            return None
        v = v.iloc[0]
    if isinstance(v, (np.ndarray,)):
        if v.size == 0:
            return None
        v = v.flat[0]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v
