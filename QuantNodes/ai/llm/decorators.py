# coding=utf-8
"""LLM Client Decorators (Phase 1.1).

实现 4 个跨切关注点的装饰器, 可链式组合:
  RetryingLLMClient(LoggingLLMClient(OpenAIClient(...)))  # 推荐顺序
  TokenCountingLLMClient(RetryingLLMClient(...))
  CachedLLMClient(RetryingLLMClient(LoggingLLMClient(...)))

每个装饰器只关注一件事, 通过组合实现横切关注分离。
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from QuantNodes.ai.llm.base import (
    ChatCompletion,
    LLMClientBase,
    Message,
    RateLimitError,
    APIError,
)


# ============================================================================
# Decorator 1: RetryingLLMClient
# ============================================================================

class RetryingLLMClient(LLMClientBase):
    """装饰器: 在 RateLimitError / APIError 时自动重试, 指数退避。

    Args:
        inner: 被包装的 LLM 客户端
        max_retries: 最大重试次数 (覆盖 LLMClientBase 字段)
        initial_backoff: 首次重试等待秒数
        backoff_factor: 退避倍数 (e.g. 2.0 → 1s, 2s, 4s, 8s)
        retry_on: 触发重试的异常类型元组, 默认 (RateLimitError, APIError)
    """

    def __init__(
        self,
        inner: LLMClientBase,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        backoff_factor: float = 2.0,
        retry_on: tuple = (RateLimitError, APIError),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            api_key=inner.api_key, base_url=inner.base_url,
            timeout=inner.timeout, max_retries=max_retries, **kwargs,
        )
        self.inner = inner
        self.initial_backoff = initial_backoff
        self.backoff_factor = backoff_factor
        self.retry_on = retry_on
        self.total_retries = 0

    def _call_api(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        backoff = self.initial_backoff
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.inner._call_api(messages, model, **kwargs)
            except self.retry_on as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break
                self.total_retries += 1
                self.logger.warning(
                    f"LLM call attempt {attempt + 1} failed ({type(e).__name__}: {e}), "
                    f"retrying in {backoff:.1f}s..."
                )
                time.sleep(backoff)
                backoff *= self.backoff_factor
        assert last_exc is not None
        raise last_exc

    def _call_api_stream(self, messages, model=None, **kwargs):
        # 流式不支持重试 (用户体验差), 透传
        yield from self.inner._call_api_stream(messages, model, **kwargs)

    def get_model_list(self) -> List[str]:
        return self.inner.get_model_list()


# ============================================================================
# Decorator 2: LoggingLLMClient
# ============================================================================

class LoggingLLMClient(LLMClientBase):
    """装饰器: 记录每条 LLM 请求/响应 + latency 到 logger.

    不会修改 prompt 或 response, 仅观察。日志格式:
      [LLMCall] model=foo latency=123ms prompt_tokens=N response_len=M
    """

    def __init__(self, inner: LLMClientBase, log_level: int = logging.INFO, **kwargs: Any) -> None:
        super().__init__(
            api_key=inner.api_key, base_url=inner.base_url,
            timeout=inner.timeout, max_retries=inner.max_retries, **kwargs,
        )
        self.inner = inner
        self.log_level = log_level
        self.call_log: List[Dict[str, Any]] = []

    def _call_api(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        start = time.perf_counter()
        prompt_tokens = self.count_messages_tokens(messages)
        try:
            response = self.inner._call_api(messages, model, **kwargs)
            latency_ms = (time.perf_counter() - start) * 1000
            entry = {
                "model": model or "default",
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "response_len": len(response.content),
                "error": None,
            }
            self.call_log.append(entry)
            self.logger.log(
                self.log_level,
                f"[LLMCall] model={entry['model']} latency={latency_ms:.1f}ms "
                f"prompt_tokens={prompt_tokens} response_len={entry['response_len']}",
            )
            return response
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            entry = {
                "model": model or "default",
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "response_len": 0,
                "error": f"{type(e).__name__}: {e}",
            }
            self.call_log.append(entry)
            self.logger.error(
                f"[LLMCall] model={entry['model']} latency={latency_ms:.1f}ms "
                f"ERROR={entry['error']}"
            )
            raise

    def _call_api_stream(self, messages, model=None, **kwargs):
        yield from self.inner._call_api_stream(messages, model, **kwargs)

    def get_model_list(self) -> List[str]:
        return self.inner.get_model_list()


# ============================================================================
# Decorator 3: TokenCountingLLMClient (Phase 1.2)
# ============================================================================

class TokenCountingLLMClient(LLMClientBase):
    """装饰器: 累计 LLM 调用的 token 消耗, 用于 cost tracking。

    读取 ChatCompletion.usage.total_tokens (若有), 累加到 self.total_tokens_used。
    同时记录 prompt / completion 各自的累计。
    """

    def __init__(self, inner: LLMClientBase, **kwargs: Any) -> None:
        super().__init__(
            api_key=inner.api_key, base_url=inner.base_url,
            timeout=inner.timeout, max_retries=inner.max_retries, **kwargs,
        )
        self.inner = inner
        self.total_tokens_used = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0

    def _call_api(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        response = self.inner._call_api(messages, model, **kwargs)
        self.call_count += 1
        if response.usage:
            pt = response.usage.get("prompt_tokens", 0)
            ct = response.usage.get("completion_tokens", 0)
            tt = response.usage.get("total_tokens", pt + ct)
            self.total_prompt_tokens += pt
            self.total_completion_tokens += ct
            self.total_tokens_used += tt
        return response

    def _call_api_stream(self, messages, model=None, **kwargs):
        # 流式响应通常无 usage 统计, 透传
        yield from self.inner._call_api_stream(messages, model, **kwargs)

    def get_model_list(self) -> List[str]:
        return self.inner.get_model_list()

    def reset(self) -> None:
        self.total_tokens_used = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0


# ============================================================================
# Decorator 4: CachedLLMClient (Phase 1.2)
# ============================================================================

class CachedLLMClient(LLMClientBase):
    """装饰器: 缓存 LLM 响应 (prompt_hash, model) → ChatCompletion。

    用于:
      - 单元测试中避免重复 API 调用
      - 相同 prompt 的批量处理提速
      - 离线 replay

    Args:
        inner: 被包装的 LLM 客户端
        max_size: LRU 缓存最大条目数, 默认 128
    """

    def __init__(self, inner: LLMClientBase, max_size: int = 128, **kwargs: Any) -> None:
        super().__init__(
            api_key=inner.api_key, base_url=inner.base_url,
            timeout=inner.timeout, max_retries=inner.max_retries, **kwargs,
        )
        self.inner = inner
        self.max_size = max_size
        self._cache: "OrderedDict[str, ChatCompletion]" = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0

    @staticmethod
    def _make_key(messages: List[Message], model: Optional[str], kwargs: Dict[str, Any]) -> str:
        payload = (
            model or "",
            tuple((m.role.value, m.content) for m in messages),
            tuple(sorted(kwargs.items())),
        )
        return hashlib.sha256(repr(payload).encode()).hexdigest()

    def _call_api(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        key = self._make_key(messages, model, kwargs)
        if key in self._cache:
            self.hit_count += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.miss_count += 1
        response = self.inner._call_api(messages, model, **kwargs)
        self._cache[key] = response
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
        return response

    def _call_api_stream(self, messages, model=None, **kwargs):
        # 流式不缓存
        yield from self.inner._call_api_stream(messages, model, **kwargs)

    def get_model_list(self) -> List[str]:
        return self.inner.get_model_list()

    def clear(self) -> None:
        self._cache.clear()
        self.hit_count = 0
        self.miss_count = 0
