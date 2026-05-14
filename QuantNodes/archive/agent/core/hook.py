# coding=utf-8
"""
Hook系统

Agent生命周期钩子，支持流式输出、进度回调、自定义扩展
"""

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..providers.base import LLMResponse


@dataclass
class AgentHookContext:
    """Hook执行上下文"""
    iteration: int
    messages: List[Dict[str, Any]]
    response: Optional[LLMResponse] = None
    usage: Dict[str, int] = None
    tool_calls: List[Any] = None
    tool_results: List[Any] = None
    tool_events: List[Dict[str, str]] = None
    streamed_content: bool = False
    final_content: Optional[str] = None
    stop_reason: Optional[str] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.usage is None:
            self.usage = {}
        if self.tool_calls is None:
            self.tool_calls = []
        if self.tool_results is None:
            self.tool_results = []
        if self.tool_events is None:
            self.tool_events = []


class AgentHook(ABC):
    """Agent生命周期钩子基类"""

    def wants_streaming(self) -> bool:
        """是否需要流式输出"""
        return False

    async def before_iteration(self, context: AgentHookContext) -> None:
        """在每轮迭代前调用"""
        pass

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        """流式输出每个token时调用"""
        pass

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool = False) -> None:
        """流式输出结束时调用"""
        pass

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        """在执行工具前调用"""
        pass

    async def after_iteration(self, context: AgentHookContext) -> None:
        """在每轮迭代后调用"""
        pass

    def finalize_content(self, context: AgentHookContext, content: Optional[str]) -> Optional[str]:
        """最终内容处理（可修改返回给用户的内容）"""
        return content


class CompositeHook(AgentHook):
    """组合多个Hook"""

    def __init__(self, hooks: List[AgentHook] | None = None):
        self.hooks = list(hooks) if hooks else []

    def add_hook(self, hook: AgentHook) -> None:
        """添加Hook"""
        self.hooks.append(hook)

    def wants_streaming(self) -> bool:
        return any(h.wants_streaming() for h in self.hooks)

    async def before_iteration(self, context: AgentHookContext) -> None:
        for hook in self.hooks:
            await hook.before_iteration(context)

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        for hook in self.hooks:
            await hook.on_stream(context, delta)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool = False) -> None:
        for hook in self.hooks:
            await hook.on_stream_end(context, resuming=resuming)

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        for hook in self.hooks:
            await hook.before_execute_tools(context)

    async def after_iteration(self, context: AgentHookContext) -> None:
        for hook in self.hooks:
            await hook.after_iteration(context)

    def finalize_content(self, context: AgentHookContext, content: Optional[str]) -> Optional[str]:
        for hook in self.hooks:
            content = hook.finalize_content(context, content)
        return content
