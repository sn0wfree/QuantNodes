# coding=utf-8
"""
测试Hook系统
"""

import asyncio
from QuantNodes.agent.core.hook import AgentHook, CompositeHook, AgentHookContext


class SimpleHook(AgentHook):
    def __init__(self):
        self.before_calls = []
        self.after_calls = []

    async def before_iteration(self, context: AgentHookContext) -> None:
        self.before_calls.append(context.iteration)

    async def after_iteration(self, context: AgentHookContext) -> None:
        self.after_calls.append(context.iteration)


class TestAgentHookContext:
    def test_init_basic(self):
        ctx = AgentHookContext(iteration=1, messages=[])
        assert ctx.iteration == 1
        assert ctx.messages == []

    def test_init_with_defaults(self):
        ctx = AgentHookContext(iteration=1, messages=[])
        assert ctx.usage == {}
        assert ctx.tool_calls == []
        assert ctx.error is None


class TestAgentHook:
    def test_wants_streaming_default(self):
        hook = AgentHook()
        assert hook.wants_streaming() is False

    def test_finalize_content_default(self):
        hook = AgentHook()
        ctx = AgentHookContext(iteration=1, messages=[])
        result = hook.finalize_content(ctx, "test")
        assert result == "test"


class TestCompositeHook:
    def test_add_hook(self):
        composite = CompositeHook()
        hook1 = AgentHook()
        hook2 = AgentHook()
        composite.add_hook(hook1)
        composite.add_hook(hook2)
        assert len(composite.hooks) == 2

    def test_wants_streaming_any_true(self):
        class StreamingHook(AgentHook):
            def wants_streaming(self) -> bool:
                return True

        composite = CompositeHook([AgentHook(), StreamingHook()])
        assert composite.wants_streaming() is True

    def test_wants_streaming_all_false(self):
        composite = CompositeHook([AgentHook(), AgentHook()])
        assert composite.wants_streaming() is False

    def test_before_iteration_calls_all_hooks(self):
        async def _test():
            hook1 = SimpleHook()
            hook2 = SimpleHook()
            composite = CompositeHook([hook1, hook2])
            ctx = AgentHookContext(iteration=1, messages=[])

            await composite.before_iteration(ctx)

            assert hook1.before_calls == [1]
            assert hook2.before_calls == [1]

        asyncio.run(_test())

    def test_after_iteration_calls_all_hooks(self):
        async def _test():
            hook1 = SimpleHook()
            hook2 = SimpleHook()
            composite = CompositeHook([hook1, hook2])
            ctx = AgentHookContext(iteration=1, messages=[])

            await composite.after_iteration(ctx)

            assert hook1.after_calls == [1]
            assert hook2.after_calls == [1]

        asyncio.run(_test())

    def test_finalize_content_chains(self):
        class AddPrefixHook(AgentHook):
            def finalize_content(self, ctx, content):
                return f"PREFIX: {content}"

        class AddSuffixHook(AgentHook):
            def finalize_content(self, ctx, content):
                return f"{content} :SUFFIX"

        composite = CompositeHook([AddPrefixHook(), AddSuffixHook()])
        ctx = AgentHookContext(iteration=1, messages=[])

        result = composite.finalize_content(ctx, "test")
        assert result == "PREFIX: test :SUFFIX"

    def test_on_stream_calls_all_hooks(self):
        class StreamCaptureHook(AgentHook):
            def __init__(self):
                self.deltas = []

            async def on_stream(self, context, delta):
                self.deltas.append(delta)

        async def _test():
            hook1 = StreamCaptureHook()
            hook2 = StreamCaptureHook()
            composite = CompositeHook([hook1, hook2])
            ctx = AgentHookContext(iteration=1, messages=[])

            await composite.on_stream(ctx, "token1")
            await composite.on_stream(ctx, "token2")

            assert hook1.deltas == ["token1", "token2"]
            assert hook2.deltas == ["token1", "token2"]

        asyncio.run(_test())

    def test_on_stream_end_calls_all_hooks(self):
        class StreamEndHook(AgentHook):
            def __init__(self):
                self.end_calls = []

            async def on_stream_end(self, context, *, resuming=False):
                self.end_calls.append(resuming)

        async def _test():
            hook1 = StreamEndHook()
            hook2 = StreamEndHook()
            composite = CompositeHook([hook1, hook2])
            ctx = AgentHookContext(iteration=1, messages=[])

            await composite.on_stream_end(ctx, resuming=False)
            await composite.on_stream_end(ctx, resuming=True)

            assert hook1.end_calls == [False, True]
            assert hook2.end_calls == [False, True]

        asyncio.run(_test())

    def test_before_execute_tools_calls_all_hooks(self):
        class ToolHook(AgentHook):
            def __init__(self):
                self.calls = []

            async def before_execute_tools(self, context):
                self.calls.append(context.iteration)

        async def _test():
            hook1 = ToolHook()
            hook2 = ToolHook()
            composite = CompositeHook([hook1, hook2])
            ctx = AgentHookContext(iteration=42, messages=[])

            await composite.before_execute_tools(ctx)

            assert hook1.calls == [42]
            assert hook2.calls == [42]

        asyncio.run(_test())

    def test_hook_error_isolation(self):
        class BadHook(AgentHook):
            async def before_iteration(self, context):
                raise RuntimeError("Hook error")

        class GoodHook(SimpleHook):
            pass

        async def _test():
            bad_hook = BadHook()
            good_hook = GoodHook()
            composite = CompositeHook([bad_hook, good_hook])
            ctx = AgentHookContext(iteration=1, messages=[])

            try:
                await composite.before_iteration(ctx)
                assert False, "Should have raised"
            except RuntimeError:
                pass

        asyncio.run(_test())

    def test_context_with_response_and_usage(self):
        from QuantNodes.agent.providers.base import LLMResponse

        response = LLMResponse(content="test")
        ctx = AgentHookContext(
            iteration=1,
            messages=[{"role": "user", "content": "hi"}],
            response=response,
            usage={"prompt_tokens": 100},
        )

        assert ctx.response.content == "test"
        assert ctx.usage["prompt_tokens"] == 100

    def test_context_with_tool_info(self):
        ctx = AgentHookContext(
            iteration=1,
            messages=[],
            tool_calls=[{"name": "echo"}],
            tool_results=["hello"],
            tool_events=[{"type": "tool_success"}],
        )

        assert len(ctx.tool_calls) == 1
        assert len(ctx.tool_results) == 1
        assert len(ctx.tool_events) == 1

    def test_context_final_state(self):
        ctx = AgentHookContext(
            iteration=0,
            messages=[],
            final_content="Final answer",
            stop_reason="completed",
            error=None,
        )

        assert ctx.final_content == "Final answer"
        assert ctx.stop_reason == "completed"
        assert ctx.error is None
