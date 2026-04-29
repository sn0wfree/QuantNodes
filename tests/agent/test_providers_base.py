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
