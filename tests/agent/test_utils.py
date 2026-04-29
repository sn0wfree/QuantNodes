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

    def test_short_string(self):
        assert count_tokens("abc") == 0

    def test_exact_multiple(self):
        assert count_tokens("abcd") == 1
        assert count_tokens("abcdabcd") == 2

    def test_long_string(self):
        text = "x" * 1000
        assert count_tokens(text) == 250

    def test_whitespace_only(self):
        assert count_tokens("   ") == 0
        assert count_tokens("    ") == 1


class TestEnsureAsync:
    def test_ensure_async_with_sync_function(self):
        import asyncio
        from QuantNodes.agent.utils.helpers import ensure_async

        def sync_func(a, b):
            return a + b

        async def _test():
            result = await ensure_async(sync_func, 2, 3)
            return result

        result = asyncio.run(_test())
        assert result == 5

    def test_ensure_async_with_async_function(self):
        import asyncio
        from QuantNodes.agent.utils.helpers import ensure_async

        async def async_func(a, b):
            await asyncio.sleep(0.001)
            return a * b

        async def _test():
            result = await ensure_async(async_func, 5, 6)
            return result

        result = asyncio.run(_test())
        assert result == 30

    def test_ensure_async_with_kwargs(self):
        import asyncio
        from QuantNodes.agent.utils.helpers import ensure_async

        def sync_func_with_kwargs(a, b, multiplier=1):
            return (a + b) * multiplier

        async def _test():
            result = await ensure_async(sync_func_with_kwargs, 2, 3, multiplier=10)
            return result

        result = asyncio.run(_test())
        assert result == 50


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


class TestTruncateTextBoundary:
    def test_truncate_zero_max_length(self):
        result = truncate_text("hello", 0)
        assert result == "..."

    def test_truncate_exact_length(self):
        text = "hello"
        result = truncate_text(text, len(text))
        assert result == "hello"

    def test_truncate_one_over_length(self):
        text = "hello"
        result = truncate_text(text, len(text) + 1)
        assert result == "hello"

    def test_truncate_shorter_than_suffix(self):
        result = truncate_text("hi", 2)
        assert result == "hi"

    def test_truncate_empty_string(self):
        result = truncate_text("", 10)
        assert result == ""

    def test_truncate_with_long_suffix(self):
        result = truncate_text("hello world", 10, suffix="......")
        assert "......" in result
        assert len(result) <= 10


class TestCountTokensBoundary:
    def test_count_tokens_none_input(self):
        result = count_tokens(None)
        assert result == 0

    def test_count_tokens_non_string_input(self):
        result = count_tokens(12345)
        assert result > 0

    def test_count_tokens_unicode_characters(self):
        text = "你好世界"
        result = count_tokens(text)
        assert result > 0

    def test_count_tokens_mixed_content(self):
        text = "Hello 世界 123 !@#"
        result = count_tokens(text)
        assert result > 0

    def test_count_tokens_very_long_string(self):
        text = "x" * 10000
        result = count_tokens(text)
        assert result == 2500


class TestEnsureAsyncEdgeCases:
    def test_ensure_async_with_none_function(self):
        import asyncio
        from QuantNodes.agent.utils.helpers import ensure_async

        async def _test():
            result = await ensure_async(lambda: None)
            return result

        result = asyncio.run(_test())
        assert result is None

    def test_ensure_async_with_exception(self):
        import asyncio
        from QuantNodes.agent.utils.helpers import ensure_async

        def failing_func():
            raise ValueError("Test error")

        async def _test():
            try:
                await ensure_async(failing_func)
                return "no error"
            except ValueError as e:
                return str(e)

        result = asyncio.run(_test())
        assert result == "Test error"

    def test_ensure_async_with_many_args(self):
        import asyncio
        from QuantNodes.agent.utils.helpers import ensure_async

        def sum_many(*args):
            return sum(args)

        async def _test():
            result = await ensure_async(sum_many, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
            return result

        result = asyncio.run(_test())
        assert result == 55

    def test_ensure_async_with_only_kwargs(self):
        import asyncio
        from QuantNodes.agent.utils.helpers import ensure_async

        def kwargs_only(**kwargs):
            return kwargs

        async def _test():
            result = await ensure_async(kwargs_only, a=1, b=2, c=3)
            return result

        result = asyncio.run(_test())
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_ensure_async_preserves_function_reference(self):
        import asyncio
        from QuantNodes.agent.utils.helpers import ensure_async

        def identity(x):
            return x

        async def _test():
            result = await ensure_async(identity, identity)
            return result

        result = asyncio.run(_test())
        assert result == identity
