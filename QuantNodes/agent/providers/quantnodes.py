# coding=utf-8
"""
QuantNodes LLM Provider适配器

适配现有LLMClientBase到Agent Provider接口。
支持单client模式（向后兼容）和ProviderRegistry动态路由模式。
"""

import logging
from typing import Any, Dict, List, Callable, Awaitable, Optional
import asyncio
import json
import re

from .base import LLMProvider, LLMResponse, ToolCallRequest
from QuantNodes.ai.llm.base import LLMClientBase, Message as QNMessage, MessageRole

logger = logging.getLogger(__name__)


class QuantNodesLLMProvider(LLMProvider):
    """适配QuantNodes现有LLM客户端的Provider

    支持两种初始化模式：
    1. 旧模式：QuantNodesLLMProvider(client) — 绑定单个client（向后兼容）
    2. 新模式：QuantNodesLLMProvider(registry=registry) — 按model动态路由
    """

    def __init__(
        self,
        client: LLMClientBase | None = None,
        default_model: str | None = None,
        default_max_tokens: int = 102400,
        registry=None,
        fallback_providers: list[str] | None = None,
    ):
        super().__init__()
        self.client = client
        self.default_model = default_model
        self.default_max_tokens = default_max_tokens
        self.registry = registry
        self.fallback_providers = fallback_providers or []

    def _get_client_for_model(self, model: str | None) -> tuple[LLMClientBase, str]:
        """根据model找到对应client和实际model名

        旧模式（无registry）：返回绑定的单个client
        新模式（有registry）：按model动态路由
        """
        if self.registry is None:
            return self.client, model or self.default_model

        config = self.registry.resolve(model)
        if config:
            actual_model = model or self.default_model
            return self.registry.get_client(config), actual_model

        # 兜底：返回默认client
        if self.client:
            return self.client, model or self.default_model
        default_client = self.registry.get_default_client()
        return default_client, model or self.default_model

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
        """调用LLM"""
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
        client, actual_model = self._get_client_for_model(model)

        def _call():
            return client.chat(
                messages=qn_messages,
                model=actual_model,
                temperature=temperature,
                max_tokens=effective_max_tokens,
            )

        try:
            qn_response = await asyncio.get_event_loop().run_in_executor(None, _call)
        except Exception as e:
            logger.error(
                "LLM chat failed: model=%s, client=%s, error=%s",
                actual_model, type(client).__name__, e, exc_info=True,
            )
            # Fallback logic
            if self.registry and self.fallback_providers:
                for fb_name in self.fallback_providers:
                    fb_config = self.registry.get(fb_name)
                    if fb_config and fb_config.name != getattr(client, '_provider_name', None):
                        try:
                            fb_client = self.registry.get_client(fb_config)
                            qn_response = await asyncio.get_event_loop().run_in_executor(
                                None, lambda: fb_client.chat(
                                    messages=qn_messages,
                                    model=actual_model,
                                    temperature=temperature,
                                    max_tokens=effective_max_tokens,
                                )
                            )
                            logger.info("Fallback to provider %s succeeded", fb_name)
                            break
                        except Exception:
                            continue
                else:
                    raise
            else:
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
        """流式调用LLM"""
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
        client, actual_model = self._get_client_for_model(model)

        full_content = ""
        tool_call_buffer = ""
        in_tool_call = False
        streamed_content = ""

        def _iter_chunks():
            return client.chat_stream(
                messages=qn_messages,
                model=actual_model,
                temperature=temperature,
                max_tokens=effective_max_tokens,
            )

        loop = asyncio.get_event_loop()
        try:
            chunks = await loop.run_in_executor(None, lambda: list(_iter_chunks()))
        except Exception as e:
            logger.error(
                "LLM stream failed: model=%s, client=%s, error=%s",
                actual_model, type(client).__name__, e, exc_info=True,
            )
            # Fallback logic
            if self.registry and self.fallback_providers:
                for fb_name in self.fallback_providers:
                    fb_config = self.registry.get(fb_name)
                    if fb_config and fb_config.name != getattr(client, '_provider_name', None):
                        try:
                            fb_client = self.registry.get_client(fb_config)
                            chunks = await loop.run_in_executor(
                                None, lambda: list(fb_client.chat_stream(
                                    messages=qn_messages,
                                    model=actual_model,
                                    temperature=temperature,
                                    max_tokens=effective_max_tokens,
                                ))
                            )
                            logger.info("Fallback stream to provider %s succeeded", fb_name)
                            break
                        except Exception:
                            continue
                else:
                    raise
            else:
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
