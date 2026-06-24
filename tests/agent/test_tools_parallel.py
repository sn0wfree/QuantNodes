# coding=utf-8
"""Tests for ``ToolRegistry.execute_tools_parallel`` (parallel tool execution).

v3.0.0 contract: ``ToolRegistry.execute_tools_parallel`` groups calls by
``tool.read_only`` — read-only tools run concurrently via
``asyncio.gather``, while write-side tools run sequentially to avoid
race conditions. This is the same strategy the v2.x implementation
used; we just retarget the test to the v3.0.0 ``ToolRegistry``.
"""

import asyncio
from typing import Any, Dict, List

import pytest

from QuantNodes.agent.tools import Tool, ToolRegistry
from QuantNodes.agent.tools.base import ToolExecutionResult


# ----------------------------------------------------------------------------
# Test tools
# ----------------------------------------------------------------------------

class _SlowEchoTool(Tool):
    """A read-only tool that sleeps a configurable time before echoing."""

    def __init__(self, delay: float = 0.01):
        self._delay = delay

    @property
    def name(self) -> str:
        return "slow_echo"

    @property
    def description(self) -> str:
        return "Slow echo for testing parallel execution"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        }

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, message: str, **kwargs: Any) -> str:
        await asyncio.sleep(self._delay)
        return message


class _WriteTool(Tool):
    """A non-read-only tool that records call count."""

    def __init__(self) -> None:
        self.call_count = 0

    @property
    def name(self) -> str:
        return "write_tool"

    @property
    def description(self) -> str:
        return "Non-read-only tool"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    @property
    def read_only(self) -> bool:
        return False

    async def execute(self, **kwargs: Any) -> str:
        self.call_count += 1
        return f"write_{self.call_count}"


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

class TestParallelToolExecution:
    def test_all_read_only_runs_concurrently(self):
        """3 read-only tools run concurrently (results all succeed)."""
        async def _test():
            registry = ToolRegistry()
            registry.register(_SlowEchoTool(delay=0.001))

            calls = [
                {"name": "slow_echo", "arguments": {"message": "msg1"}},
                {"name": "slow_echo", "arguments": {"message": "msg2"}},
                {"name": "slow_echo", "arguments": {"message": "msg3"}},
            ]
            results = await registry.execute_tools_parallel(calls)
            assert len(results) == 3
            assert all(isinstance(r, ToolExecutionResult) for r in results)
            assert all(r.success for r in results)
            assert [r.content for r in results] == ["msg1", "msg2", "msg3"]

        asyncio.run(_test())

    def test_mixed_read_only_and_write(self):
        """Read-only tools concurrent; write tools serial. All succeed."""
        async def _test():
            registry = ToolRegistry()
            echo = _SlowEchoTool(delay=0.001)
            write = _WriteTool()
            registry.register(echo)
            registry.register(write)

            calls = [
                {"name": "slow_echo", "arguments": {"message": "msg1"}},
                {"name": "write_tool", "arguments": {}},
                {"name": "slow_echo", "arguments": {"message": "msg2"}},
                {"name": "write_tool", "arguments": {}},
            ]
            results = await registry.execute_tools_parallel(calls)
            assert len(results) == 4
            assert all(r.success for r in results)
            # write_tool was called exactly twice (serial, not 4 times)
            assert write.call_count == 2

        asyncio.run(_test())

    def test_missing_tool_in_parallel_returns_failure(self):
        """Unknown tool in parallel call returns failure result, not raises."""
        async def _test():
            registry = ToolRegistry()
            registry.register(_SlowEchoTool(delay=0.001))

            calls = [
                {"name": "slow_echo", "arguments": {"message": "hello"}},
                {"name": "nonexistent_tool", "arguments": {}},
            ]
            results = await registry.execute_tools_parallel(calls)
            assert len(results) == 2
            assert results[0].success is True
            assert results[0].content == "hello"
            assert results[1].success is False
            assert "not found" in (results[1].error or "").lower()

        asyncio.run(_test())

    def test_empty_call_list_returns_empty(self):
        """Calling with no tools returns an empty list (no errors)."""
        async def _test():
            registry = ToolRegistry()
            results = await registry.execute_tools_parallel([])
            assert results == []

        asyncio.run(_test())

    def test_preserves_call_order_in_results(self):
        """Results are returned in the same order as the input calls."""
        async def _test():
            registry = ToolRegistry()
            registry.register(_SlowEchoTool(delay=0.001))

            calls = [
                {"name": "slow_echo", "arguments": {"message": "alpha"}},
                {"name": "slow_echo", "arguments": {"message": "beta"}},
                {"name": "slow_echo", "arguments": {"message": "gamma"}},
            ]
            results = await registry.execute_tools_parallel(calls)
            assert [r.content for r in results] == ["alpha", "beta", "gamma"]

        asyncio.run(_test())

    def test_concurrent_execution_faster_than_serial(self):
        """Sanity check: 5 read-only calls with 50ms each run in ~50ms (parallel)
        not ~250ms (serial). We use a generous 80ms threshold to avoid CI flakes.
        """
        import time
        async def _test():
            registry = ToolRegistry()
            registry.register(_SlowEchoTool(delay=0.05))

            calls = [
                {"name": "slow_echo", "arguments": {"message": f"m{i}"}}
                for i in range(5)
            ]
            t0 = time.monotonic()
            await registry.execute_tools_parallel(calls)
            elapsed = time.monotonic() - t0
            # 5 × 50ms = 250ms serial, ~50ms parallel; allow 200ms ceiling
            assert elapsed < 0.2, f"parallel execution took {elapsed:.2f}s (expected <0.2s)"

        asyncio.run(_test())
