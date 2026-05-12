# coding=utf-8
"""
QuantNodes LLM Provider适配器

适配现有LLMClientBase到Agent Provider接口
"""

from typing import Any, Dict, List, Callable, Awaitable
import asyncio
import json
import re

from .base import LLMProvider, LLMResponse, ToolCallRequest
from QuantNodes.ai.llm.base import LLMClientBase, Message as QNMessage, MessageRole


class QuantNodesLLMProvider(LLMProvider):
    """适配QuantNodes现有LLM客户端的Provider"""

    def __init__(self, client: LLMClientBase, default_model: str | None = None, default_max_tokens: int = 102400):
        super().__init__()
        self.client = client
        self.default_model = default_model
        self.default_max_tokens = default_max_tokens

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[QNMessage]:
        """将OpenAI格式消息转换为QuantNodes格式"""
        result = []
        for msg in messages:
            role_str = msg.get("role", "user")
            try:
                role = MessageRole(role_str)
            except ValueError:
                role = MessageRole.USER
            content = msg.get("content", "")
            if content is None:
                content = ""
            result.append(QNMessage(role=role, content=content))
        return result

    def _parse_tool_calls(self, response_content: str | None) -> List[ToolCallRequest]:
        """从响应中解析工具调用"""
        tool_calls = []
        if response_content is None:
            return tool_calls
        content = response_content.strip()

        if "```tool_call" in content:
            import re
            pattern = r"```tool_call\s*([\s\S]*?)\s*```"
            matches = re.findall(pattern, content)
            for match in matches:
                try:
                    data = json.loads(match.strip())
                    tool_calls.append(ToolCallRequest(
                        id=data.get("id", "tc_0"),
                        name=data.get("name", ""),
                        arguments=data.get("arguments", {}),
                    ))
                except (json.JSONDecodeError, ValueError):
                    continue

        return tool_calls

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        tool_choice: str | Dict[str, Any] | None = None,
    ) -> LLMResponse:
        """调用QuantNodes LLM客户端"""
        messages = self._enforce_role_alternation(messages)
        qn_messages = self._convert_messages(messages)

        if tools:
            tools_desc = "\n".join([
                f"- {t['function']['name']}: {t['function']['description']}"
                for t in tools
            ])
            system_msg = next((m for m in qn_messages if m.role == MessageRole.SYSTEM), None)
            if system_msg:
                system_msg.content += f"\n\n可用工具:\n{tools_desc}"
                system_msg.content += "\n\n如果需要调用工具，请使用```tool_call```代码块输出JSON格式的工具调用。"

        effective_max_tokens = max_tokens or self.default_max_tokens

        def _call():
            return self.client.chat(
                messages=qn_messages,
                model=model or self.default_model,
                temperature=temperature,
                max_tokens=effective_max_tokens
            )

        try:
            qn_response = await asyncio.get_event_loop().run_in_executor(None, _call)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "LLM chat failed: model=%s, error=%s", model or self.default_model, e, exc_info=True
            )
            raise

        content = qn_response.content
        tool_calls = self._parse_tool_calls(content)

        if tool_calls:
            content = content.split("```tool_call")[0].strip()

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage=qn_response.usage or {},
        )

    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """流式调用QuantNodes LLM客户端"""
        messages = self._enforce_role_alternation(messages)
        qn_messages = self._convert_messages(messages)

        if tools:
            tools_desc = "\n".join([
                f"- {t['function']['name']}: {t['function']['description']}"
                for t in tools
            ])
            system_msg = next((m for m in qn_messages if m.role == MessageRole.SYSTEM), None)
            if system_msg:
                system_msg.content += f"\n\n可用工具:\n{tools_desc}"
                system_msg.content += "\n\n如果需要调用工具，请使用```tool_call```代码块输出JSON格式的工具调用。"

        effective_max_tokens = max_tokens or self.default_max_tokens

        full_content = ""
        tool_call_buffer = ""
        in_tool_call = False
        streamed_content = ""

        def _iter_chunks():
            return self.client.chat_stream(
                messages=qn_messages,
                model=model or self.default_model,
                temperature=temperature,
                max_tokens=effective_max_tokens,
            )

        loop = asyncio.get_event_loop()
        try:
            chunks = await loop.run_in_executor(None, lambda: list(_iter_chunks()))
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(
                "LLM stream failed: model=%s, error=%s", model or self.default_model, e, exc_info=True
            )
            raise

        for chunk in chunks:
            delta = chunk.content or ""
            if not delta:
                continue

            full_content += delta

            if in_tool_call:
                tool_call_buffer += delta
                if "```" in tool_call_buffer:
                    in_tool_call = False
                    tool_call_buffer = ""
                continue

            if "```tool_call" in full_content:
                parts = full_content.split("```tool_call", 1)
                before = parts[0]
                if before[len(streamed_content):].strip():
                    new_text = before[len(streamed_content):]
                    streamed_content = before
                    if on_content_delta:
                        await on_content_delta(new_text)
                in_tool_call = True
                tool_call_buffer = delta
                continue

            new_text = full_content[len(streamed_content):]
            if new_text:
                streamed_content = full_content
                if on_content_delta:
                    await on_content_delta(new_text)

        tool_calls = self._parse_tool_calls(full_content)
        content = full_content.split("```tool_call")[0].strip() if tool_calls else full_content.strip()

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage={},
        )
