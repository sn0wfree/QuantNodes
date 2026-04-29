# coding=utf-8
"""
测试Provider基类
"""

from QuantNodes.agent.providers.base import LLMProvider, LLMResponse, ToolCallRequest


class TestToolCallRequest:
    def test_init(self):
        req = ToolCallRequest(id="tc1", name="echo", arguments={"message": "hello"})
        assert req.id == "tc1"
        assert req.name == "echo"
        assert req.arguments["message"] == "hello"


class TestLLMResponse:
    def test_init_basic(self):
        resp = LLMResponse(content="Hello world")
        assert resp.content == "Hello world"
        assert resp.tool_calls == []

    def test_init_with_tool_calls(self):
        tc = ToolCallRequest(id="tc1", name="echo", arguments={"message": "hi"})
        resp = LLMResponse(content=None, tool_calls=[tc], finish_reason="tool_calls")
        assert resp.has_tool_calls is True
        assert resp.should_execute_tools is True

    def test_has_tool_calls_false(self):
        resp = LLMResponse(content="Hello")
        assert resp.has_tool_calls is False
        assert resp.should_execute_tools is False

    def test_error_response(self):
        resp = LLMResponse(content="error", error="some error")
        assert resp.error == "some error"


class MockLLMProvider(LLMProvider):
    async def chat(self, messages, tools=None, model=None, max_tokens=1024, temperature=0.7, tool_choice=None):
        return LLMResponse(content="Mock response")


class TestLLMProvider:
    def test_init(self):
        provider = MockLLMProvider()
        assert provider is not None

    def test_enforce_role_alternation_empty(self):
        result = LLMProvider._enforce_role_alternation([])
        assert result == []

    def test_enforce_role_alternation_merge(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "world"},
        ]
        result = LLMProvider._enforce_role_alternation(messages)
        assert len(result) == 1
        assert "hello" in result[0]["content"]
        assert "world" in result[0]["content"]

    def test_enforce_role_alternation_keep_system(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "world"},
        ]
        result = LLMProvider._enforce_role_alternation(messages)
        assert len(result) == 2
        assert result[0]["role"] == "system"

    def test_enforce_role_alternation_no_merge_assistant_tools(self):
        messages = [
            {"role": "assistant", "content": "msg1", "tool_calls": []},
            {"role": "assistant", "content": "msg2"},
        ]
        result = LLMProvider._enforce_role_alternation(messages)
        assert len(result) == 1
        # 有tool_calls的assistant消息优先保留

    def test_enforce_role_alternation_mixed_content_types(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {"role": "user", "content": "world"},
        ]
        result = LLMProvider._enforce_role_alternation(messages)
        assert len(result) == 1
        assert result[0]["content"] == "world"

    def test_enforce_role_alternation_alternate_roles(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "world"},
        ]
        result = LLMProvider._enforce_role_alternation(messages)
        assert len(result) == 3

    def test_enforce_role_alternation_three_consecutive(self):
        messages = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        result = LLMProvider._enforce_role_alternation(messages)
        assert len(result) == 1
        assert "a" in result[0]["content"]
        assert "b" in result[0]["content"]
        assert "c" in result[0]["content"]


class TestLLMProviderStreaming:
    def test_chat_stream_calls_callback(self):
        import asyncio

        callback_calls = []

        async def callback(content):
            callback_calls.append(content)

        async def _test():
            provider = MockLLMProvider()
            response = await provider.chat_stream(
                messages=[{"role": "user", "content": "hello"}],
                on_content_delta=callback,
            )
            return response

        result = asyncio.run(_test())
        assert len(callback_calls) == 1
        assert callback_calls[0] == "Mock response"
        assert result.content == "Mock response"

    def test_chat_stream_no_callback(self):
        import asyncio

        async def _test():
            provider = MockLLMProvider()
            response = await provider.chat_stream(
                messages=[{"role": "user", "content": "hello"}],
            )
            return response

        result = asyncio.run(_test())
        assert result.content == "Mock response"

    def test_chat_stream_empty_content_no_callback(self):
        import asyncio

        class EmptyContentProvider(LLMProvider):
            async def chat(self, messages, tools=None, model=None, **kwargs):
                return LLMResponse(content=None)

        async def _test():
            provider = EmptyContentProvider()
            response = await provider.chat_stream(
                messages=[{"role": "user", "content": "hello"}],
                on_content_delta=lambda x: None,
            )
            return response

        result = asyncio.run(_test())
        assert result.content is None
