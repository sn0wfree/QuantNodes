# coding=utf-8
"""
测试工具并发执行
"""

import asyncio
from typing import Any, Dict
from QuantNodes.agent.tools import ToolRegistry, Tool


class SlowEchoTool(Tool):
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
            "properties": {
                "message": {"type": "string", "description": "Message to echo"}
            },
            "required": ["message"]
        }

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, message: str, **kwargs: Any) -> str:
        await asyncio.sleep(self._delay)
        return message


class WriteTool(Tool):
    def __init__(self):
        self.call_count = 0

    @property
    def name(self) -> str:
        return "write_tool"

    @property
    def description(self) -> str:
        return "Non-read-only tool"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    @property
    def read_only(self) -> bool:
        return False

    async def execute(self, **kwargs: Any) -> str:
        self.call_count += 1
        return f"write_{self.call_count}"


class TestParallelToolExecution:
    def test_execute_tools_parallel_read_only(self):
        async def _test():
            registry = ToolRegistry()
            registry.register(SlowEchoTool(delay=0.001))

            tool_calls = [
                {"name": "slow_echo", "arguments": {"message": "msg1"}},
                {"name": "slow_echo", "arguments": {"message": "msg2"}},
                {"name": "slow_echo", "arguments": {"message": "msg3"}},
            ]

            results = await registry.execute_tools_parallel(tool_calls)

            assert len(results) == 3
            assert all(r.success for r in results)

        asyncio.run(_test())

    def test_execute_tools_mixed_read_only_and_write(self):
        async def _test():
            registry = ToolRegistry()
            echo_tool = SlowEchoTool(delay=0.001)
            write_tool = WriteTool()
            registry.register(echo_tool)
            registry.register(write_tool)

            tool_calls = [
                {"name": "slow_echo", "arguments": {"message": "msg1"}},
                {"name": "write_tool", "arguments": {}},
                {"name": "slow_echo", "arguments": {"message": "msg2"}},
                {"name": "write_tool", "arguments": {}},
            ]

            results = await registry.execute_tools_parallel(tool_calls)

            assert len(results) == 4
            assert all(r.success for r in results)

        asyncio.run(_test())
