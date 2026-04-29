# coding=utf-8
"""
测试 QuantNodes LLM Provider 适配器
"""

import asyncio
from typing import List, Dict, Any
from unittest.mock import Mock, MagicMock

import pytest

from QuantNodes.agent.providers.quantnodes import QuantNodesLLMProvider
from QuantNodes.agent.providers.base import LLMResponse, ToolCallRequest
from QuantNodes.ai.llm.base import Message, MessageRole


class MockLLMClient:
    def __init__(self, response_content: str = "Test response", usage: Dict | None = None):
        self.response_content = response_content
        self.usage = usage or {}
        self.call_count = 0
        self.last_messages = None

    def chat(self, messages: List[Message], **kwargs) -> Any:
        self.call_count += 1
        self.last_messages = messages
        response = Mock()
        response.content = self.response_content
        response.usage = self.usage
        return response


class TestQuantNodesLLMProviderInit:
    def test_init_with_client(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)
        assert provider.client == mock_client
        assert provider.default_model is None

    def test_init_with_default_model(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client, default_model="gpt-4")
        assert provider.default_model == "gpt-4"


class TestMessageConversion:
    def test_convert_single_user_message(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)

        messages = [{"role": "user", "content": "Hello"}]
        result = provider._convert_messages(messages)

        assert len(result) == 1
        assert result[0].role == MessageRole.USER
        assert result[0].content == "Hello"

    def test_convert_single_assistant_message(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)

        messages = [{"role": "assistant", "content": "Hi there"}]
        result = provider._convert_messages(messages)

        assert len(result) == 1
        assert result[0].role == MessageRole.ASSISTANT
        assert result[0].content == "Hi there"

    def test_convert_system_message(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)

        messages = [{"role": "system", "content": "Be helpful"}]
        result = provider._convert_messages(messages)

        assert len(result) == 1
        assert result[0].role == MessageRole.SYSTEM
        assert result[0].content == "Be helpful"

    def test_convert_multiple_messages(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)

        messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        result = provider._convert_messages(messages)

        assert len(result) == 3
        assert result[0].role == MessageRole.SYSTEM
        assert result[1].role == MessageRole.USER
        assert result[2].role == MessageRole.ASSISTANT

    def test_convert_message_without_content(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)

        messages = [{"role": "user"}]
        result = provider._convert_messages(messages)

        assert len(result) == 1
        assert result[0].content == ""

    def test_convert_message_without_role(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)

        messages = [{"content": "Hello"}]
        result = provider._convert_messages(messages)

        assert len(result) == 1
        assert result[0].role == MessageRole.USER


class TestToolCallParsing:
    def test_parse_single_tool_call(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)

        content = """Let me check that.

```tool_call
{
    "id": "call_1",
    "name": "echo",
    "arguments": {"message": "test"}
}
```"""

        result = provider._parse_tool_calls(content)

        assert len(result) == 1
        assert result[0].id == "call_1"
        assert result[0].name == "echo"
        assert result[0].arguments == {"message": "test"}

    def test_parse_multiple_tool_calls(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)

        content = """```tool_call
{"id": "call_1", "name": "echo", "arguments": {"message": "first"}}
```

```tool_call
{"id": "call_2", "name": "echo", "arguments": {"message": "second"}}
```"""

        result = provider._parse_tool_calls(content)

        assert len(result) == 2
        assert result[0].name == "echo"
        assert result[1].name == "echo"

    def test_parse_no_tool_calls(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)

        content = "Just a normal response without tool calls."
        result = provider._parse_tool_calls(content)

        assert len(result) == 0

    def test_parse_invalid_json_tool_call(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)

        content = """```tool_call
this is not valid json
```"""

        result = provider._parse_tool_calls(content)

        assert len(result) == 0

    def test_parse_tool_call_with_missing_fields(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)

        content = """```tool_call
{"name": "echo"}
```"""

        result = provider._parse_tool_calls(content)

        assert len(result) == 1
        assert result[0].id == "tc_0"
        assert result[0].name == "echo"
        assert result[0].arguments == {}

    def test_parse_tool_call_with_empty_content(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)

        result = provider._parse_tool_calls("")

        assert len(result) == 0


class TestQuantNodesLLMProviderChat:
    def test_chat_basic(self):
        mock_client = MockLLMClient(response_content="Hello from LLM")
        provider = QuantNodesLLMProvider(mock_client)

        async def _test():
            response = await provider.chat(
                messages=[{"role": "user", "content": "Hi"}]
            )
            return response

        result = asyncio.run(_test())

        assert isinstance(result, LLMResponse)
        assert result.content == "Hello from LLM"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"
        assert mock_client.call_count == 1

    def test_chat_with_usage(self):
        mock_client = MockLLMClient(
            response_content="Hi",
            usage={"prompt_tokens": 10, "completion_tokens": 5}
        )
        provider = QuantNodesLLMProvider(mock_client)

        async def _test():
            return await provider.chat(messages=[{"role": "user", "content": "Hi"}])

        result = asyncio.run(_test())

        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5

    def test_chat_with_tools(self):
        response_with_tool = """Let me check that.

```tool_call
{"id": "call_1", "name": "echo", "arguments": {"message": "test"}}
```"""
        mock_client = MockLLMClient(response_content=response_with_tool)
        provider = QuantNodesLLMProvider(mock_client)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echoes a message",
                    "parameters": {"type": "object", "properties": {}},
                }
            }
        ]

        async def _test():
            return await provider.chat(
                messages=[
                    {"role": "system", "content": "Be helpful"},
                    {"role": "user", "content": "Hi"}
                ],
                tools=tools,
            )

        result = asyncio.run(_test())

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "echo"
        assert result.finish_reason == "tool_calls"
        assert "Let me check that." in result.content

    def test_chat_with_model_parameter(self):
        mock_client = MockLLMClient(response_content="Hi")
        provider = QuantNodesLLMProvider(mock_client, default_model="default-model")

        async def _test():
            return await provider.chat(
                messages=[{"role": "user", "content": "Hi"}],
                model="custom-model",
            )

        asyncio.run(_test())

        assert mock_client.last_messages is not None

    def test_chat_with_temperature_and_max_tokens(self):
        mock_client = MockLLMClient(response_content="Hi")
        provider = QuantNodesLLMProvider(mock_client)

        async def _test():
            return await provider.chat(
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.5,
                max_tokens=512,
            )

        asyncio.run(_test())

        assert mock_client.call_count == 1

    def test_chat_role_alternation_applied(self):
        mock_client = MockLLMClient(response_content="Hi")
        provider = QuantNodesLLMProvider(mock_client)

        messages = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
        ]

        async def _test():
            return await provider.chat(messages=messages)

        asyncio.run(_test())

        assert len(mock_client.last_messages) == 1

    def test_chat_tool_system_message_without_system_prompt(self):
        mock_client = MockLLMClient(response_content="Hi")
        provider = QuantNodesLLMProvider(mock_client)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Test tool",
                    "parameters": {},
                }
            }
        ]

        async def _test():
            return await provider.chat(
                messages=[{"role": "user", "content": "Hi"}],
                tools=tools,
            )

        asyncio.run(_test())

        assert mock_client.call_count == 1


class TestLLMResponseProperties:
    def test_has_tool_calls_true(self):
        response = LLMResponse(
            content="Hi",
            tool_calls=[ToolCallRequest(id="1", name="echo", arguments={})],
        )
        assert response.has_tool_calls is True

    def test_has_tool_calls_false(self):
        response = LLMResponse(content="Hi")
        assert response.has_tool_calls is False

    def test_should_execute_tools_true_with_tool_calls(self):
        response = LLMResponse(
            content="Hi",
            tool_calls=[ToolCallRequest(id="1", name="echo", arguments={})],
            finish_reason="tool_calls",
        )
        assert response.should_execute_tools is True

    def test_should_execute_tools_true_with_stop(self):
        response = LLMResponse(
            content="Hi",
            tool_calls=[ToolCallRequest(id="1", name="echo", arguments={})],
            finish_reason="stop",
        )
        assert response.should_execute_tools is True

    def test_should_execute_tools_false_no_tool_calls(self):
        response = LLMResponse(content="Hi", finish_reason="stop")
        assert response.should_execute_tools is False

    def test_should_execute_tools_false_wrong_finish_reason(self):
        response = LLMResponse(
            content="Hi",
            tool_calls=[ToolCallRequest(id="1", name="echo", arguments={})],
            finish_reason="content_filter",
        )
        assert response.should_execute_tools is False

    def test_llm_response_defaults(self):
        response = LLMResponse(content="Hello")
        assert response.tool_calls == []
        assert response.finish_reason == "stop"
        assert response.usage == {}
        assert response.error is None

    def test_llm_response_with_error(self):
        response = LLMResponse(
            content=None,
            error="API rate limit exceeded",
        )
        assert response.content is None
        assert response.error == "API rate limit exceeded"

    def test_llm_response_usage_aggregation(self):
        response = LLMResponse(
            content="Hi",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
        assert response.usage["prompt_tokens"] == 100
        assert response.usage["completion_tokens"] == 50
        assert response.usage["total_tokens"] == 150


class TestToolCallRequest:
    def test_tool_call_request_all_fields(self):
        request = ToolCallRequest(
            id="call_123",
            name="my_tool",
            arguments={"param1": "value1", "param2": 42},
        )
        assert request.id == "call_123"
        assert request.name == "my_tool"
        assert request.arguments == {"param1": "value1", "param2": 42}

    def test_tool_call_request_empty_arguments(self):
        request = ToolCallRequest(id="call_1", name="tool", arguments={})
        assert request.arguments == {}


class TestRoleAlternationEdgeCases:
    def test_enforce_role_alternation_empty_messages(self):
        from QuantNodes.agent.providers.base import LLMProvider

        result = LLMProvider._enforce_role_alternation([])
        assert result == []

    def test_enforce_role_alternation_single_message(self):
        from QuantNodes.agent.providers.base import LLMProvider

        messages = [{"role": "user", "content": "Hello"}]
        result = LLMProvider._enforce_role_alternation(messages)

        assert len(result) == 1
        assert result[0]["content"] == "Hello"

    def test_enforce_role_alternation_three_consecutive_user(self):
        from QuantNodes.agent.providers.base import LLMProvider

        messages = [
            {"role": "user", "content": "First"},
            {"role": "user", "content": "Second"},
            {"role": "user", "content": "Third"},
        ]
        result = LLMProvider._enforce_role_alternation(messages)

        assert len(result) == 1
        assert "First" in result[0]["content"]
        assert "Second" in result[0]["content"]
        assert "Third" in result[0]["content"]

    def test_enforce_role_alternation_system_kept_separate(self):
        from QuantNodes.agent.providers.base import LLMProvider

        messages = [
            {"role": "system", "content": "Sys1"},
            {"role": "system", "content": "Sys2"},
            {"role": "user", "content": "Hello"},
        ]
        result = LLMProvider._enforce_role_alternation(messages)

        assert len(result) == 3

    def test_enforce_role_alternation_assistant_merged(self):
        from QuantNodes.agent.providers.base import LLMProvider

        messages = [
            {"role": "assistant", "content": "First part"},
            {"role": "assistant", "content": "Second part"},
        ]
        result = LLMProvider._enforce_role_alternation(messages)

        assert len(result) == 1
        assert "First part" in result[0]["content"]
        assert "Second part" in result[0]["content"]

    def test_enforce_role_alternation_none_content(self):
        from QuantNodes.agent.providers.base import LLMProvider

        messages = [
            {"role": "user", "content": None},
            {"role": "user", "content": "actual content"},
        ]
        result = LLMProvider._enforce_role_alternation(messages)

        assert len(result) == 1


class TestToolCallParsingEdgeCases:
    def test_parse_tool_call_with_whitespace(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)

        content = """```tool_call
        {
            "id": "call_1",
            "name": "echo",
            "arguments": {"message": "test"}
        }
        ```"""

        result = provider._parse_tool_calls(content)
        assert len(result) == 1

    def test_parse_tool_call_with_trailing_text(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)

        content = """```tool_call
{"id": "call_1", "name": "echo", "arguments": {}}
```

Some more text after the tool call."""

        result = provider._parse_tool_calls(content)
        assert len(result) == 1

    def test_parse_tool_call_with_leading_text(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)

        content = """Some text before.

```tool_call
{"id": "call_1", "name": "echo", "arguments": {}}
```"""

        result = provider._parse_tool_calls(content)
        assert len(result) == 1

    def test_parse_tool_call_none_content(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)

        result = provider._parse_tool_calls(None)
        assert len(result) == 0


class TestMessageConversionEdgeCases:
    def test_convert_messages_empty_list(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)

        result = provider._convert_messages([])
        assert result == []

    def test_convert_message_unknown_role_defaults_to_user(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)

        messages = [{"role": "unknown_role", "content": "test"}]
        result = provider._convert_messages(messages)

        assert len(result) == 1
        assert result[0].role == MessageRole.USER

    def test_convert_message_none_content(self):
        mock_client = MockLLMClient()
        provider = QuantNodesLLMProvider(mock_client)

        messages = [{"role": "user", "content": None}]
        result = provider._convert_messages(messages)

        assert len(result) == 1
        assert result[0].content == ""
