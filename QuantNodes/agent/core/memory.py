# coding=utf-8
"""
Memory System with DreamStore Extension

Phase 4.2: Dream System
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List
from pathlib import Path
import json


@dataclass
class Dream:
    """Dream entry for insights"""
    id: str
    timestamp: str
    type: str
    content: str
    insights: List[str] = field(default_factory=list)
    source: str = ""
    confidence: float = 0.0
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DreamConfig:
    """Dream system configuration"""
    max_dreams_per_day: int = 10
    min_confidence: float = 0.7
    auto_inject: bool = True
    inject_position: str = "prepend"
    retention_days: int = 30


class DreamStore:
    """Dream Storage Layer"""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace) / "dream"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._dreams_file = self.workspace / "dreams.jsonl"
        self._memory_file = self.workspace.parent / "memory" / "memory.md"

    def save_dream(self, dream: Dream) -> None:
        """Save a dream entry"""
        with open(self._dreams_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(dream.__dict__, ensure_ascii=False) + "\n")

    def get_recent_dreams(self, limit: int = 10) -> List[Dream]:
        """Get recent dreams"""
        dreams = []
        if not self._dreams_file.exists():
            return dreams
        with open(self._dreams_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-limit:]:
            try:
                data = json.loads(line)
                dreams.append(Dream(**data))
            except Exception:
                continue
        return list(reversed(dreams))

    def get_dreams_by_type(
        self, dream_type: str, limit: int = 10
    ) -> List[Dream]:
        """Get dreams by type"""
        all_dreams = self.get_recent_dreams(limit * 10)
        return [d for d in all_dreams if d.type == dream_type][:limit]

    def get_injection_content(self, config: DreamConfig) -> str:
        """Get content for memory injection"""
        dreams = self.get_recent_dreams(config.max_dreams_per_day)
        high_confidence = [d for d in dreams if d.confidence >= config.min_confidence]
        if not high_confidence:
            return ""
        lines = ["\n## Dream Insights\n"]
        for dream in high_confidence:
            lines.append(f"**{dream.type}** ({dream.timestamp[:10]}): {dream.content}")
            if dream.insights:
                for insight in dream.insights:
                    lines.append(f"  - {insight}")
        return "\n".join(lines)

    def inject_to_memory(self, config: DreamConfig) -> None:
        """Inject dreams to memory.md"""
        injection = self.get_injection_content(config)
        if not injection:
            return
        if self._memory_file.exists():
            with open(self._memory_file, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = ""
        if config.inject_position == "prepend":
            content = injection + "\n\n" + content
        else:
            content = content + "\n\n" + injection
        with open(self._memory_file, "w", encoding="utf-8") as f:
            f.write(content)


class MemoryStore:
    """Extended Memory Storage (Compatible with Phase 1)"""

    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace) / "memory"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._memory_file = self.workspace / "memory.md"
        self._history_file = self.workspace / "history.jsonl"
        self._dream_store = DreamStore(workspace)

    def read_memory(self) -> str:
        """Read long-term memory"""
        if self._memory_file.exists():
            with open(self._memory_file, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def write_memory(self, content: str) -> None:
        """Write long-term memory"""
        with open(self._memory_file, "w", encoding="utf-8") as f:
            f.write(content)

    def append_history(self, entry: Dict[str, Any]) -> None:
        """Append session summary to history"""
        entry["timestamp"] = datetime.now().isoformat()
        with open(self._history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_memory_context(self) -> str:
        """Get memory context for prompt injection"""
        memory = self.read_memory()
        if memory:
            return f"记忆:\n{memory}"
        return ""

    def get_dream_store(self) -> DreamStore:
        """Get DreamStore"""
        return self._dream_store