# coding=utf-8
"""
测试执行引擎 (AgentRunner)
"""

import asyncio
from typing import Any, Dict, List
from QuantNodes.agent.core.runner import AgentRunner, AgentRunSpec, AgentRunResult
from QuantNodes.agent.providers.base import LLMProvider, LLMResponse
from QuantNodes.agent.tools.registry import ToolRegistry
from QuantNodes.agent.tools.echo import EchoTool


class MockProvider(LLMProvider):
    def __init__(self, responses: List[str] | None = None):
        super().__init__()
        self.responses = responses or ["Hello from mock"]
        self.call_count = 0

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        tool_choice: str | Dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.call_count += 1
        content = self.responses[self.call_count - 1] if self.call_count <= len(self.responses) else "Done"
        return LLMResponse(content=content)


class TestAgentRunSpec:
    def test_init(self):
        spec = AgentRunSpec(
            initial_messages=[{"role": "user", "content": "hello"}],
            tools=ToolRegistry(),
        )
        assert spec.max_iterations == 5
        assert spec.concurrent_tools is False


class TestAgentRunResult:
    def test_init(self):
        result = AgentRunResult(
            final_content="Hello",
            messages=[{"role": "user", "content": "hi"}],
            tools_used=[],
            usage={},
            stop_reason="completed",
        )
        assert result.final_content == "Hello"
        assert result.stop_reason == "completed"


class TestAgentRunner:
    def test_init(self):
        provider = MockProvider()
        runner = AgentRunner(provider)
        assert runner.provider == provider

    def test_run_simple(self):
        async def _test():
            provider = MockProvider(["Hello world"])
            runner = AgentRunner(provider)
            registry = ToolRegistry()

            spec = AgentRunSpec(
                initial_messages=[{"role": "user", "content": "Hello"}],
                tools=registry,
                max_iterations=1,
            )

            result = await runner.run(spec)

            assert result.final_content == "Hello world"
            assert result.stop_reason == "completed"

        asyncio.run(_test())

    def test_run_multiple_iterations(self):
        async def _test():
            provider = MockProvider(["Response 1", "Response 2", "Final"])
            runner = AgentRunner(provider)
            registry = ToolRegistry()

            spec = AgentRunSpec(
                initial_messages=[{"role": "user", "content": "Hello"}],
                tools=registry,
                max_iterations=3,
            )

            result = await runner.run(spec)

            assert result.stop_reason == "completed"

        asyncio.run(_test())

    def test_run_completes_before_max_iterations(self):
        async def _test():
            provider = MockProvider(["r1", "r2", "r3", "r4", "r5", "r6", "r7"])
            runner = AgentRunner(provider)
            registry = ToolRegistry()

            spec = AgentRunSpec(
                initial_messages=[{"role": "user", "content": "Hello"}],
                tools=registry,
                max_iterations=3,
            )

            result = await runner.run(spec)

            # Without tool calls, it completes after first iteration
            assert result.stop_reason == "completed"
            assert provider.call_count == 1

        asyncio.run(_test())

    def test_run_with_tool_execution(self):
        from QuantNodes.agent.providers.base import ToolCallRequest

        class MockToolProvider(LLMProvider):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            async def chat(self, messages, tools=None, model=None, **kwargs):
                self.call_count += 1
                if self.call_count == 1:
                    return LLMResponse(
                        content="Let me check that",
                        tool_calls=[ToolCallRequest(id="call_1", name="echo", arguments={"message": "test"})],
                        finish_reason="tool_calls",
                    )
                return LLMResponse(content="Done with tool execution")

        async def _test():
            provider = MockToolProvider()
            runner = AgentRunner(provider)
            registry = ToolRegistry()
            registry.register(EchoTool())

            spec = AgentRunSpec(
                initial_messages=[{"role": "user", "content": "Hello"}],
                tools=registry,
                max_iterations=2,
            )

            result = await runner.run(spec)

            assert result.stop_reason == "completed"
            assert "echo" in result.tools_used
            assert len(result.messages) == 4  # user, assistant, tool, assistant
            assert provider.call_count == 2

        asyncio.run(_test())

    def test_run_with_error_response(self):
        class MockErrorProvider(LLMProvider):
            async def chat(self, messages, tools=None, model=None, **kwargs):
                return LLMResponse(content=None, error="API Error")

        async def _test():
            provider = MockErrorProvider()
            runner = AgentRunner(provider)
            registry = ToolRegistry()

            spec = AgentRunSpec(
                initial_messages=[{"role": "user", "content": "Hello"}],
                tools=registry,
                max_iterations=2,
            )

            result = await runner.run(spec)

            assert result.stop_reason == "error"
            assert result.error == "API Error"

        asyncio.run(_test())

    def test_run_with_concurrent_tools(self):
        from QuantNodes.agent.providers.base import ToolCallRequest

        class MockConcurrentProvider(LLMProvider):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            async def chat(self, messages, tools=None, model=None, **kwargs):
                self.call_count += 1
                if self.call_count == 1:
                    return LLMResponse(
                        content="Checking concurrently",
                        tool_calls=[
                            ToolCallRequest(id="call_1", name="echo", arguments={"message": "test1"}),
                            ToolCallRequest(id="call_2", name="echo", arguments={"message": "test2"}),
                        ],
                        finish_reason="tool_calls",
                    )
                return LLMResponse(content="All done")

        async def _test():
            provider = MockConcurrentProvider()
            runner = AgentRunner(provider)
            registry = ToolRegistry()
            registry.register(EchoTool())

            spec = AgentRunSpec(
                initial_messages=[{"role": "user", "content": "Hello"}],
                tools=registry,
                max_iterations=2,
                concurrent_tools=True,
            )

            result = await runner.run(spec)

            assert result.stop_reason == "completed"
            assert "echo" in result.tools_used
            assert provider.call_count == 2

        asyncio.run(_test())

    def test_run_with_checkpoint_callback(self):
        checkpoint_calls = []

        async def checkpoint_callback(state):
            checkpoint_calls.append(state)

        async def _test():
            provider = MockProvider(["Hello"])
            runner = AgentRunner(provider)
            registry = ToolRegistry()

            spec = AgentRunSpec(
                initial_messages=[{"role": "user", "content": "Hello"}],
                tools=registry,
                max_iterations=1,
                checkpoint_callback=checkpoint_callback,
            )

            await runner.run(spec)

        asyncio.run(_test())

    def test_run_with_injection_callback(self):
        from QuantNodes.agent.providers.base import ToolCallRequest

        injection_messages = [{"role": "user", "content": "Additional context"}]

        async def injection_callback():
            return injection_messages

        class MockInjectionProvider(LLMProvider):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            async def chat(self, messages, tools=None, model=None, **kwargs):
                self.call_count += 1
                if self.call_count == 1:
                    return LLMResponse(
                        content="Check",
                        tool_calls=[ToolCallRequest(id="call_1", name="echo", arguments={"message": "test"})],
                        finish_reason="tool_calls",
                    )
                return LLMResponse(content="Final")

        async def _test():
            provider = MockInjectionProvider()
            runner = AgentRunner(provider)
            registry = ToolRegistry()
            registry.register(EchoTool())

            spec = AgentRunSpec(
                initial_messages=[{"role": "user", "content": "Hello"}],
                tools=registry,
                max_iterations=3,
                injection_callback=injection_callback,
            )

            result = await runner.run(spec)

            assert result.had_injections is True
            assert provider.call_count == 2

        asyncio.run(_test())

    def test_tool_result_truncation(self):
        from QuantNodes.agent.providers.base import ToolCallRequest

        class MockTruncateProvider(LLMProvider):
            def __init__(self):
                super().__init__()
                self.call_count = 0

            async def chat(self, messages, tools=None, model=None, **kwargs):
                self.call_count += 1
                if self.call_count == 1:
                    return LLMResponse(
                        content="Processing",
                        tool_calls=[ToolCallRequest(id="call_1", name="echo", arguments={"message": "x" * 10000})],
                        finish_reason="tool_calls",
                    )
                return LLMResponse(content="Done")

        async def _test():
            provider = MockTruncateProvider()
            runner = AgentRunner(provider)
            registry = ToolRegistry()
            registry.register(EchoTool())

            spec = AgentRunSpec(
                initial_messages=[{"role": "user", "content": "Hello"}],
                tools=registry,
                max_iterations=2,
                max_tool_result_chars=1000,
            )

            result = await runner.run(spec)

            tool_msg = next(m for m in result.messages if m["role"] == "tool")
            assert len(tool_msg["content"]) <= 1000 + len("... (truncated)")

        asyncio.run(_test())

    def test_merge_usage(self):
        runner = AgentRunner(MockProvider())
        total = {"prompt_tokens": 100, "completion_tokens": 50}
        usage = {"prompt_tokens": 200, "completion_tokens": 150}

        merged = runner._merge_usage(total, usage)

        assert merged["prompt_tokens"] == 300
        assert merged["completion_tokens"] == 200

    def test_build_assistant_message_with_tools(self):
        from QuantNodes.agent.providers.base import ToolCallRequest

        runner = AgentRunner(MockProvider())
        response = LLMResponse(
            content="Let me check",
            tool_calls=[ToolCallRequest(id="call_1", name="test_tool", arguments={"param": "value"})],
        )

        msg = runner._build_assistant_message(response)

        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me check"
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "test_tool"

    def test_build_assistant_message_without_tools(self):
        runner = AgentRunner(MockProvider())
        response = LLMResponse(content="Just text")

        msg = runner._build_assistant_message(response)

        assert msg["role"] == "assistant"
        assert msg["content"] == "Just text"
        assert "tool_calls" not in msg
