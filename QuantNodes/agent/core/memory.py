# coding=utf-8
"""
记忆系统（简化版）

Phase 1: 基础文件存储，不做Dream/Consolidator
"""

from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class MemoryStore:
    """文件系统持久化记忆存储"""

    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace) / "memory"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._memory_file = self.workspace / "memory.md"
        self._history_file = self.workspace / "history.jsonl"

    def read_memory(self) -> str:
        """读取长期记忆"""
        if self._memory_file.exists():
            with open(self._memory_file, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def write_memory(self, content: str) -> None:
        """写入长期记忆"""
        with open(self._memory_file, "w", encoding="utf-8") as f:
            f.write(content)

    def append_history(self, entry: Dict[str, Any]) -> None:
        """追加会话摘要到历史"""
        import json
        entry["timestamp"] = datetime.now().isoformat()
        with open(self._history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def get_memory_context(self) -> str:
        """获取记忆上下文（用于注入Prompt）"""
        memory = self.read_memory()
        if memory:
            return f"记忆:\n{memory}"
        return ""
