# coding=utf-8
"""
执行引擎 (AgentRunner)

处理单轮对话 + 工具调用闭环
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Callable, Awaitable, Optional
import asyncio

from ..providers.base import LLMProvider, LLMResponse
from ..tools.registry import ToolRegistry
from ..utils.helpers import truncate_text
from .hook import AgentHook, AgentHookContext, CompositeHook


@dataclass(frozen=True)
class AgentRunSpec:
    """执行规范"""
    initial_messages: List[Dict[str, Any]]
    tools: ToolRegistry
    model: str | None = None
    max_iterations: int = 5
    max_tool_result_chars: int = 4000
    concurrent_tools: bool = False
    checkpoint_callback: Callable[[Dict], Awaitable[None]] | None = None
    injection_callback: Callable[[], Awaitable[List[Dict]]] | None = None


@dataclass
class AgentRunResult:
    """执行结果"""
    final_content: str | None
    messages: List[Dict[str, Any]]
    tools_used: List[str]
    usage: Dict[str, int]
    stop_reason: str
    error: str | None = None
    had_injections: bool = False


class AgentRunner:
    """Agent执行引擎"""

    def __init__(self, provider: LLMProvider, hook: AgentHook | None = None):
        self.provider = provider
        self.hook = hook or CompositeHook()

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        """执行主循环"""
        messages = list(spec.initial_messages)
        tools_used: List[str] = []
        total_usage: Dict[str, int] = {}
        error: str | None = None
        stop_reason = "max_iterations"
        had_injections = False

        for iteration in range(spec.max_iterations):
            context = AgentHookContext(
                iteration=iteration,
                messages=list(messages),
                usage=dict(total_usage),
            )
            await self.hook.before_iteration(context)

            tool_schemas = spec.tools.get_tool_schemas()

            response = await self.provider.chat(
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
                model=spec.model,
            )

            context.response = response
            total_usage = self._merge_usage(total_usage, response.usage)

            if response.error:
                error = response.error
                stop_reason = "error"
                assistant_msg = self._build_assistant_message(response)
                messages.append(assistant_msg)
                break

            if not response.should_execute_tools:
                stop_reason = "completed"
                assistant_msg = self._build_assistant_message(response)
                messages.append(assistant_msg)
                break

            await self.hook.before_execute_tools(context)

            tool_results, tool_events = await self._execute_tools(
                response.tool_calls,
                spec.tools,
                spec.concurrent_tools,
                spec.max_tool_result_chars,
            )

            tools_used.extend([tc.name for tc in response.tool_calls])
            context.tool_calls = response.tool_calls
            context.tool_results = tool_results
            context.tool_events = tool_events

            assistant_msg = self._build_assistant_message(response)
            messages.append(assistant_msg)

            for i, (result, tool_call) in enumerate(zip(tool_results, response.tool_calls)):
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.name,
                    "content": str(result),
                })

            await self.hook.after_iteration(context)

            if spec.injection_callback:
                injections = await spec.injection_callback()
                if injections:
                    messages.extend(injections)
                    had_injections = True

        final_content = None
        if messages and messages[-1].get("role") == "assistant":
            final_content = messages[-1].get("content")

        context = AgentHookContext(
            iteration=spec.max_iterations,
            messages=messages,
            final_content=final_content,
            stop_reason=stop_reason,
            error=error,
        )
        final_content = self.hook.finalize_content(context, final_content)

        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=list(set(tools_used)),
            usage=total_usage,
            stop_reason=stop_reason,
            error=error,
            had_injections=had_injections,
        )

    async def _execute_tools(
        self,
        tool_calls: List[Any],
        tool_registry: ToolRegistry,
        concurrent_tools: bool,
        max_result_chars: int,
    ) -> tuple[List[Any], List[Dict[str, str]]]:
        """执行工具调用"""
        results = []
        events = []

        if concurrent_tools:
            tasks = []
            for tc in tool_calls:
                task = tool_registry.execute_tool(tc.name, **tc.arguments)
                tasks.append(task)
            execution_results = await asyncio.gather(*tasks)
        else:
            execution_results = []
            for tc in tool_calls:
                result = await tool_registry.execute_tool(tc.name, **tc.arguments)
                execution_results.append(result)

        for result, tc in zip(execution_results, tool_calls):
            if result.success:
                content = truncate_text(str(result.content), max_result_chars)
                results.append(content)
                events.append({
                    "type": "tool_success",
                    "tool": tc.name,
                    "content": content[:100],
                })
            else:
                error_msg = f"Error: {result.error}"
                results.append(error_msg)
                events.append({
                    "type": "tool_error",
                    "tool": tc.name,
                    "error": result.error,
                })

        return results, events

    def _build_assistant_message(self, response: LLMResponse) -> Dict[str, Any]:
        """构建assistant消息"""
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": response.content,
        }
        if response.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in response.tool_calls
            ]
        return msg

    def _merge_usage(self, total: Dict[str, int], usage: Dict[str, int]) -> Dict[str, int]:
        """合并token用量统计"""
        result = dict(total)
        for key, value in usage.items():
            result[key] = result.get(key, 0) + value
        return result
