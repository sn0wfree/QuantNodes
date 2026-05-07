# coding=utf-8
"""QuantNodes.agent.utils.prompt_templates 单元测试"""
import tempfile
import os
from pathlib import Path

from QuantNodes.agent.utils.prompt_templates import (
    load_template, render_template, render_template_file
)


class TestLoadTemplate:
    def test_load_template_from_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello World")
            temp_path = f.name
        try:
            result = load_template(temp_path)
            assert result == "Hello World"
        finally:
            os.unlink(temp_path)

    def test_load_template_from_path_object(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test Content")
            temp_path = Path(f.name)
        try:
            result = load_template(temp_path)
            assert result == "Test Content"
        finally:
            os.unlink(str(temp_path))

    def test_load_template_nonexistent_returns_empty_string(self):
        result = load_template("/nonexistent/path/to/file.txt")
        assert result == ""


class TestRenderTemplate:
    def test_render_simple_template(self):
        template = "Hello {{name}}"
        result = render_template(template, {"name": "World"})
        assert result == "Hello World"

    def test_render_multiple_placeholders(self):
        template = "{{greeting}} {{name}}!"
        result = render_template(template, {"greeting": "Hello", "name": "Alice"})
        assert result == "Hello Alice!"

    def test_render_with_numeric_value(self):
        template = "Value: {{value}}"
        result = render_template(template, {"value": 42})
        assert result == "Value: 42"

    def test_render_with_missing_placeholders_unchanged(self):
        template = "Hello {{name}}, you have {{count}} messages"
        result = render_template(template, {"name": "Bob"})
        assert result == "Hello Bob, you have {{count}} messages"

    def test_render_with_empty_context(self):
        template = "Hello {{name}}"
        result = render_template(template, {})
        assert result == "Hello {{name}}"

    def test_render_with_empty_template(self):
        result = render_template("", {"name": "World"})
        assert result == ""

    def test_render_multiple_same_placeholder(self):
        template = "{{name}} said: {{name}}"
        result = render_template(template, {"name": "Alice"})
        assert result == "Alice said: Alice"


class TestRenderTemplateFile:
    def test_render_template_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Hello {{name}}")
            temp_path = f.name
        try:
            result = render_template_file(temp_path, {"name": "World"})
            assert result == "Hello World"
        finally:
            os.unlink(temp_path)

    def test_render_template_file_with_path_object(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Value: {{value}}")
            temp_path = Path(f.name)
        try:
            result = render_template_file(temp_path, {"value": 123})
            assert result == "Value: 123"
        finally:
            os.unlink(str(temp_path))

    def test_render_template_file_nonexistent_returns_empty(self):
        result = render_template_file("/nonexistent/file.txt", {"name": "World"})
        assert result == ""
