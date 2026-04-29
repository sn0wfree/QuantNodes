# coding=utf-8
"""
测试自动压缩模块
"""

from QuantNodes.agent.core.autocompact import truncate_history, microcompact


class TestTruncateHistory:
    def test_no_truncate_needed(self):
        messages = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        result = truncate_history(messages, max_messages=20)
        assert len(result) == 10

    def test_truncate_without_system(self):
        messages = [{"role": "user", "content": f"msg{i}"} for i in range(30)]
        result = truncate_history(messages, max_messages=20)
        assert len(result) == 20

    def test_truncate_keep_system(self):
        messages = [{"role": "system", "content": "system prompt"}]
        messages += [{"role": "user", "content": f"msg{i}"} for i in range(30)]
        result = truncate_history(messages, max_messages=20)
        assert len(result) == 21
        assert result[0]["role"] == "system"

    def test_empty_messages(self):
        result = truncate_history([])
        assert result == []


class TestMicrocompact:
    def test_no_truncate_needed(self):
        messages = [
            {"role": "tool", "content": "short result"},
            {"role": "assistant", "content": "OK"},
        ]
        result = microcompact(messages, max_tool_result_chars=100)
        assert result[0]["content"] == "short result"

    def test_truncate_long_tool_result(self):
        long_content = "x" * 1000
        messages = [{"role": "tool", "content": long_content}]
        result = microcompact(messages, max_tool_result_chars=100)
        assert len(result[0]["content"]) == 100

    def test_leave_non_tool_messages_untouched(self):
        long_content = "x" * 1000
        messages = [
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": long_content},
        ]
        result = microcompact(messages, max_tool_result_chars=100)
        assert len(result[0]["content"]) == 1000
        assert len(result[1]["content"]) == 1000
