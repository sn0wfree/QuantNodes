# coding=utf-8
"""
测试上下文构建器
"""

import tempfile
from pathlib import Path
from QuantNodes.agent.core.context import ContextBuilder


class TestContextBuilder:
    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ContextBuilder(Path(tmpdir))
            assert builder is not None

    def test_load_system_prompt_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ContextBuilder(Path(tmpdir))
            result = builder.load_system_prompt()
            assert result == ""

    def test_load_system_prompt_with_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with open(tmp_path / "identity.md", "w") as f:
                f.write("# Test Agent\n")
            with open(tmp_path / "system_prompt.md", "w") as f:
                f.write("System prompt content\n")

            builder = ContextBuilder(tmp_path)
            result = builder.load_system_prompt()
            assert "# Test Agent" in result
            assert "System prompt content" in result

    def test_build_messages_with_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ContextBuilder(Path(tmpdir))
            history = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
            messages = builder.build_messages(history, "new message")
            assert len(messages) >= 3

    def test_build_messages_no_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ContextBuilder(Path(tmpdir))
            messages = builder.build_messages([], "hello")
            assert len(messages) >= 1
            assert messages[-1]["role"] == "user"
            assert messages[-1]["content"] == "hello"

    def test_build_runtime_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            builder = ContextBuilder(Path(tmpdir))
            result = builder._build_runtime_context(
                channel="cli",
                chat_id="test_chat",
            )
            assert "cli" in result
            assert "test_chat" in result
            assert "时区" in result
