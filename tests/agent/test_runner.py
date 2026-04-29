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
