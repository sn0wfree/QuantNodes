# coding=utf-8
"""
QuantNodes LLM Provider适配器

适配现有LLMClientBase到Agent Provider接口。
支持两种模式：
1. LiteLLM SDK模式（默认）：内置重试、连接池、速率限制
2. 旧模式：使用LLMClientBase（向后兼容）

LiteLLM集成提供以下功能：
- 内置指数退避重试（区分429/500）
- httpx连接池（连接复用）
- 可配置的速率限制（Token Bucket）
- 多模型路由和Fallback支持
"""

import logging
from typing import Any, Dict, List, Callable, Awaitable, Optional
import asyncio
import json
import re

from .base import LLMProvider, LLMResponse, ToolCallRequest
from .rate_limiter import AsyncTokenBucket
from QuantNodes.ai.llm.base import LLMClientBase, Message as QNMessage, MessageRole

logger = logging.getLogger(__name__)

# 尝试导入 LiteLLM SDK
try:
    from litellm import acompletion, RateLimitError, APIError
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logger.warning("LiteLLM SDK not installed. Falling back to legacy LLMClientBase.")


class QuantNodesLLMProvider(LLMProvider):
    """适配QuantNodes现有LLM客户端的Provider

    支持两种初始化模式：
    1. LiteLLM模式（默认）：QuantNodesLLMProvider(api_key, api_base, model)
       - 内置重试、连接池、速率限制
    2. 旧模式：QuantNodesLLMProvider(client=client)
       - 使用绑定的单个LLMClientBase（向后兼容）

    初始化参数:
        api_key: API密钥（用于LiteLLM模式）
        api_base: API基础URL（用于LiteLLM模式）
        client: LLMClientBase实例（旧模式）
        default_model: 默认模型名
        default_max_tokens: 默认最大token数
        registry: ProviderRegistry实例（多模型路由）
        fallback_providers: fallback provider名称列表
        use_litellm: 是否使用LiteLLM SDK（默认True）
        rate_limit_rps: 每秒请求数（默认0.5，用于免费账号）
        max_retries: LiteLLM最大重试次数（默认3）
        timeout: 请求超时秒数（默认60）
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        client: LLMClientBase | None = None,
        default_model: str | None = None,
        default_max_tokens: int = 102400,
        registry=None,
        fallback_providers: list[str] | None = None,
        use_litellm: bool = True,
        rate_limit_rps: float = 0.5,
        max_retries: int = 3,
        timeout: float = 60.0,
    ):
        """
        Args:
            api_key: API密钥（LiteLLM模式）
            api_base: API基础URL（LiteLLM模式）
            client: LLMClientBase实例（旧模式）
            default_model: 默认模型名
            default_max_tokens: 默认最大token数
            registry: ProviderRegistry实例
            fallback_providers: fallback provider列表
            use_litellm: 是否启用LiteLLM SDK（默认True）
            rate_limit_rps: 每秒请求速率（免费账号建议0.5）
            max_retries: 最大重试次数
            timeout: 请求超时（秒）
        """
        super().__init__(api_key=api_key, api_base=api_base)
        self.client = client
        self.default_model = default_model
        self.default_max_tokens = default_max_tokens
        self.registry = registry
        self.fallback_providers = fallback_providers or []
        self.use_litellm = use_litellm and LITELLM_AVAILABLE
        self.rate_limit_rps = rate_limit_rps
        self.max_retries = max_retries
        self.timeout = timeout

        # 速率限制器（用于LiteLLM模式）
        self._rate_limiter: Optional[AsyncTokenBucket] = None
        if self.use_litellm:
            self._rate_limiter = AsyncTokenBucket(
                requests_per_second=rate_limit_rps,
                burst=1,
            )

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

    def _convert_litellm_response(self, response: Any) -> LLMResponse:
        """将LiteLLM响应转换为LLMResponse格式"""
        content = ""
        tool_calls = []
        usage = {}

        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            if hasattr(choice, 'message'):
                content = choice.message.content or ""
                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                    for tc in choice.message.tool_calls:
                        if hasattr(tc, 'id') and hasattr(tc, 'function'):
                            tool_calls.append(ToolCallRequest(
                                id=tc.id,
                                name=tc.function.name,
                                arguments=tc.function.arguments or {},
                            ))
                if hasattr(choice.message, 'finish_reason'):
                    finish_reason = choice.message.finish_reason
                else:
                    finish_reason = "tool_calls" if tool_calls else "stop"
            else:
                finish_reason = "stop"
        else:
            finish_reason = "stop"

        if hasattr(response, 'usage') and response.usage:
            usage = {
                'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0),
                'completion_tokens': getattr(response.usage, 'completion_tokens', 0),
                'total_tokens': getattr(response.usage, 'total_tokens', 0),
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason if tool_calls else "stop",
            usage=usage,
        )

    async def _call_litellm(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None,
        model: str | None,
        max_tokens: int | None,
        temperature: float,
        tool_choice: str | Dict[str, Any] | None,
    ) -> LLMResponse:
        """使用LiteLLM SDK调用LLM"""
        if self._rate_limiter:
            await self._rate_limiter.acquire()

        actual_model = model or self.default_model

        if self.api_base and actual_model and "/" in actual_model:
            parts = actual_model.split("/", 1)
            known_prefixes = {
                "openrouter", "anthropic", "huggingface", "bedrock", "vertex_ai",
                "ollama", "deepseek", "groq", "fireworks_ai", "mistral",
                "perplexity", "together_ai", "replicate",
            }
            if parts[0] in known_prefixes:
                actual_model = parts[1]
                logger.info(f"Stripped provider prefix for LiteLLM, model: {actual_model}")

        try:
            response = await acompletion(
                model=actual_model,
                messages=messages,
                api_key=self.api_key,
                base_url=self.api_base,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
            return self._convert_litellm_response(response)
        except RateLimitError as e:
            logger.warning(f"Rate limit hit, trying fallback: {e}")
            raise
        except APIError as e:
            logger.error(f"LiteLLM API error: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'url'):
                logger.error(f"Request URL: {e.response.url}")
            raise
        except Exception as e:
            logger.error(f"LiteLLM call failed: {e}")
            logger.error(f"Model: {actual_model}, base_url: {self.api_base}")
            raise

    async def _fallback_to_legacy(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None,
        model: str | None,
        max_tokens: int | None,
        temperature: float,
    ) -> LLMResponse:
        """LiteLLM失败时降级到原有LLMClientBase"""
        logger.info("Falling back to legacy LLMClientBase")

        qn_messages = self._convert_messages(messages)

        if tools:
            tools_desc = "\n".join([
                f"- {t['function']['name']}: {t['function']['description']}"
                for t in tools
            ])
            system_msg = next((m for m in qn_messages if m.role == MessageRole.SYSTEM), None)
            if system_msg:
                system_msg.content += f"\n\n可用工具:\n{tools_desc}"
                system_msg.content += (
    "\n\n如果需要调用工具，请使用```tool_call```代码块输出JSON格式的工具调用。"
)

        effective_max_tokens = max_tokens or self.default_max_tokens
        client, actual_model = self._get_client_for_model(model)

        # Strip litellm provider prefix for legacy client
# (e.g. openrouter/google/gemini -> google/gemini)
        # LiteLLM needs the prefix to route correctly, but legacy OpenAI client doesn't use it
        if actual_model and "/" in actual_model:
            parts = actual_model.split("/", 1)
            known_prefixes = {
                "openrouter", "anthropic", "huggingface", "bedrock", "vertex_ai",
                "ollama", "deepseek", "groq", "fireworks_ai", "mistral",
                "perplexity", "together_ai", "replicate",
            }
            if parts[0] in known_prefixes:
                actual_model = parts[1]
                logger.info(f"Stripped provider prefix for legacy client, model: {actual_model}")

        if client is None:
            raise RuntimeError(
                "No LLM client available for legacy fallback. "
                "Configure api_key and api_base in settings.json."
            )

        def _call():
            return client.chat(
                messages=qn_messages,
                model=actual_model,
                temperature=temperature,
                max_tokens=effective_max_tokens,
            )

        loop = asyncio.get_event_loop()
        qn_response = await loop.run_in_executor(None, _call)

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

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        tool_choice: str | Dict[str, Any] | None = None,
    ) -> LLMResponse:
        """调用LLM

        优先使用LiteLLM SDK，失败时降级到legacy client。
        """
        messages = self._enforce_role_alternation(messages)

        # 如果禁用了LiteLLM或LiteLLM不可用，直接使用legacy
        if not self.use_litellm:
            return await self._fallback_to_legacy(
                messages, tools, model, max_tokens, temperature
            )

        try:
            return await self._call_litellm(
                messages, tools, model, max_tokens, temperature, tool_choice
            )
        except Exception as e:
            logger.warning(f"LiteLLM call failed, trying legacy: {e}")

            # 检查是否有legacy client可用
            if self.client is None and self.registry is None:
                raise RuntimeError(
                    "LiteLLM failed and no legacy client available. "
                    f"Original error: {e}"
                )

            try:
                return await self._fallback_to_legacy(
                    messages, tools, model, max_tokens, temperature
                )
            except Exception as fallback_error:
                logger.error(f"Legacy fallback also failed: {fallback_error}")
                raise fallback_error

    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        on_content_delta: Callable[[str], Awaitable[None]] | None = None,
    ) -> LLMResponse:
        """流式调用LLM

        优先使用LiteLLM SDK，失败时降级到legacy client。
        注意：LiteLLM的流式响应需要特殊处理。
        """
        messages = self._enforce_role_alternation(messages)

        if not self.use_litellm:
            return await self._stream_legacy(
                messages, tools, model, max_tokens, temperature, on_content_delta
            )

        # 速率限制
        if self._rate_limiter:
            await self._rate_limiter.acquire()

        actual_model = model or self.default_model

        try:
            full_content = ""
            tool_call_buffer = ""
            in_tool_call = False
            streamed_content = ""
            tool_calls = []

            # LiteLLM 流式调用
            async for chunk in await acompletion(
                model=actual_model,
                messages=messages,
                api_key=self.api_key,
                base_url=self.api_base,
                max_tokens=max_tokens or self.default_max_tokens,
                temperature=temperature,
                tools=tools,
                stream=True,
            ):
                # 解析chunk内容
                delta = ""
                if hasattr(chunk, 'choices') and chunk.choices:
                    choice = chunk.choices[0]
                    if hasattr(choice, 'delta') and choice.delta:
                        delta = choice.delta.content or ""
                        if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                            for tc in choice.delta.tool_calls:
                                if hasattr(tc, 'function'):
                                    tool_call_buffer += tc.function.arguments or ""

                if not delta and not tool_call_buffer:
                    continue

                full_content += delta

                if in_tool_call:
                    tool_call_buffer += delta
                    if "```" in tool_call_buffer:
                        in_tool_call = False
                        try:
                            data = json.loads(tool_call_buffer.replace("```", "").strip())
                            tool_calls.append(ToolCallRequest(
                                id=data.get("id", f"tc_{len(tool_calls)}"),
                                name=data.get("name", ""),
                                arguments=data.get("arguments", {}),
                            ))
                        except (json.JSONDecodeError, ValueError):
                            pass
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

            # 解析最终的tool_calls
            if not tool_calls and "```tool_call" in full_content:
                parsed = self._parse_tool_calls(full_content)
                tool_calls.extend(parsed)

            content = (
                full_content.split("```tool_call")[0].strip()
                if tool_calls else full_content.strip()
            )

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason="tool_calls" if tool_calls else "stop",
                usage={},
            )

        except Exception as e:
            logger.warning(f"LiteLLM stream failed, trying legacy: {e}")

            if self.client is None and self.registry is None:
                raise RuntimeError(
                    f"LiteLLM stream failed and no legacy client available. Original error: {e}"
                )

            try:
                return await self._stream_legacy(
                    messages, tools, model, max_tokens, temperature, on_content_delta
                )
            except Exception as fallback_error:
                logger.error(f"Legacy stream fallback also failed: {fallback_error}")
                raise fallback_error

    async def _stream_legacy(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None,
        model: str | None,
        max_tokens: int | None,
        temperature: float,
        on_content_delta: Callable[[str], Awaitable[None]] | None,
    ) -> LLMResponse:
        """使用legacy LLMClientBase进行流式调用"""
        qn_messages = self._convert_messages(messages)

        if tools:
            tools_desc = "\n".join([
                f"- {t['function']['name']}: {t['function']['description']}"
                for t in tools
            ])
            system_msg = next((m for m in qn_messages if m.role == MessageRole.SYSTEM), None)
            if system_msg:
                system_msg.content += f"\n\n可用工具:\n{tools_desc}"
                system_msg.content += (
    "\n\n如果需要调用工具，请使用```tool_call```代码块输出JSON格式的工具调用。"
)

        effective_max_tokens = max_tokens or self.default_max_tokens
        client, actual_model = self._get_client_for_model(model)

        # Strip litellm provider prefix for legacy client
# (e.g. openrouter/google/gemini -> google/gemini)
        # LiteLLM needs the prefix to route correctly, but legacy OpenAI client doesn't use it
        if actual_model and "/" in actual_model:
            parts = actual_model.split("/", 1)
            known_prefixes = {
                "openrouter", "anthropic", "huggingface", "bedrock", "vertex_ai",
                "ollama", "deepseek", "groq", "fireworks_ai", "mistral",
                "perplexity", "together_ai", "replicate",
            }
            if parts[0] in known_prefixes:
                actual_model = parts[1]
                logger.info(
                    f"Stripped provider prefix for legacy stream client, model: {actual_model}"
                )

        if client is None:
            raise RuntimeError(
                "No LLM client available for legacy stream fallback. "
                "Configure api_key and api_base in settings.json."
            )

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
        chunks = await loop.run_in_executor(None, lambda: list(_iter_chunks()))

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
        content = (
            full_content.split("```tool_call")[0].strip()
            if tool_calls else full_content.strip()
        )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
            usage={},
        )
