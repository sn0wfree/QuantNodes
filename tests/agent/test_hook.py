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
