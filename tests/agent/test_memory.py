# coding=utf-8
"""
测试记忆系统
"""

import tempfile
from pathlib import Path
from QuantNodes.agent.core.memory import MemoryStore


class TestMemoryStore:
    def test_read_empty_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            result = store.read_memory()
            assert result == ""

    def test_write_and_read_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            store.write_memory("This is a test memory.")
            result = store.read_memory()
            assert result == "This is a test memory."

    def test_append_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            store.append_history({"type": "test", "content": "hello"})
            history_file = Path(tmpdir) / "memory" / "history.jsonl"
            assert history_file.exists()
            with open(history_file, "r") as f:
                lines = f.readlines()
                assert len(lines) == 1

    def test_get_memory_context_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            context = store.get_memory_context()
            assert context == ""

    def test_get_memory_context_with_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = MemoryStore(Path(tmpdir))
            store.write_memory("Test memory content")
            context = store.get_memory_context()
            assert "Test memory content" in context
