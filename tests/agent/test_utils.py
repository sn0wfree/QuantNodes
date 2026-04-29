# coding=utf-8
"""
测试工具函数
"""

import tempfile
from pathlib import Path
from QuantNodes.agent.utils.helpers import truncate_text, count_tokens
from QuantNodes.agent.utils.prompt_templates import load_template, render_template


class TestTruncateText:
    def test_no_truncate_needed(self):
        result = truncate_text("hello", 10)
        assert result == "hello"

    def test_truncate(self):
        result = truncate_text("hello world", 8)
        assert result == "hello..."
        assert len(result) == 8

    def test_custom_suffix(self):
        result = truncate_text("helloworld", 7, suffix="..")
        assert result == "hello.."


class TestCountTokens:
    def test_count_tokens(self):
        text = "hello world"
        result = count_tokens(text)
        assert result == len(text) // 4

    def test_empty_string(self):
        assert count_tokens("") == 0


class TestPromptTemplates:
    def test_load_nonexistent_template(self):
        result = load_template("/nonexistent/path.md")
        assert result == ""

    def test_load_template(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("Hello {{name}}!")
            temp_path = f.name

        try:
            result = load_template(temp_path)
            assert result == "Hello {{name}}!"
        finally:
            Path(temp_path).unlink()

    def test_render_template(self):
        template = "Hello {{name}}! You are {{age}} years old."
        context = {"name": "Alice", "age": 30}
        result = render_template(template, context)
        assert result == "Hello Alice! You are 30 years old."

    def test_render_empty_context(self):
        template = "Hello {{name}}!"
        result = render_template(template, {})
        assert result == "Hello {{name}}!"
