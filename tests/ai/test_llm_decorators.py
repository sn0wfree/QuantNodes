# coding=utf-8
"""Tests for LLM Client Decorators (Phase 1.1 + 1.2).

Covers:
  - NullLLMClient (Null Object pattern)
  - LLMError inheritance from QuantNodesError
  - RetryingLLMClient (retry with backoff)
  - LoggingLLMClient (call observation)
  - TokenCountingLLMClient (cumulative token tracking)
  - CachedLLMClient (LRU cache)
  - Decorator composition (chained decorators)
"""
import logging
import time
from typing import List

import pytest

from QuantNodes.ai.llm.base import (
    LLMClientBase,
    LLMError,
    RateLimitError,
    APIError,
    AuthenticationError,
    Message,
    MessageRole,
    ChatCompletion,
)
from QuantNodes.ai.llm.null import NullLLMClient
from QuantNodes.ai.llm.decorators import (
    RetryingLLMClient,
    LoggingLLMClient,
    TokenCountingLLMClient,
    CachedLLMClient,
)
from QuantNodes.core.base import QuantNodesError


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class FakeLLMClient(LLMClientBase):
    """测试用 fake LLM 客户端, 可配置响应/异常。"""

    def __init__(
        self,
        response_content: str = "fake response",
        fail_times: int = 0,
        fail_with: type = RateLimitError,
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        **kwargs,
    ):
        super().__init__(api_key="fake", max_retries=0, **kwargs)
        self.response_content = response_content
        self.fail_times = fail_times
        self.fail_with = fail_with
        self.call_count = 0
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    def _call_api(self, messages, model=None, **kwargs):
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise self.fail_with(f"simulated failure #{self.call_count}")
        return ChatCompletion(
            content=self.response_content,
            role=MessageRole.ASSISTANT,
            finish_reason="stop",
            usage={
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            },
        )

    def get_model_list(self):
        return ["fake-model"]


def make_messages(content: str = "hello") -> List[Message]:
    return [Message(role=MessageRole.USER, content=content)]


# ---------------------------------------------------------------------------
# Exception hierarchy (Phase 1.1)
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    def test_llm_error_inherits_from_quantnodes_error(self):
        assert issubclass(LLMError, QuantNodesError)
        assert issubclass(LLMError, Exception)

    def test_rate_limit_inherits_llm_error(self):
        assert issubclass(RateLimitError, LLMError)
        assert issubclass(RateLimitError, QuantNodesError)

    def test_authentication_inherits_llm_error(self):
        assert issubclass(AuthenticationError, LLMError)

    def test_api_error_inherits_llm_error(self):
        assert issubclass(APIError, LLMError)

    def test_catchable_as_quantnodes_error(self):
        """Phase 1.1 关键收益: 顶层 except QuantNodesError 现在能捕获 LLM 错误。"""
        try:
            raise RateLimitError("test")
        except QuantNodesError as e:
            assert e.code == "LLM_RATE_LIMIT"
            assert e.message == "test"

    def test_error_has_details(self):
        """LLMError 继承 QuantNodesError 后, 也获得 details 字段。"""
        err = APIError("bad", details={"status": 500})
        assert err.details == {"status": 500}


# ---------------------------------------------------------------------------
# NullLLMClient (Null Object)
# ---------------------------------------------------------------------------

class TestNullLLMClient:
    def test_returns_canned_response(self):
        client = NullLLMClient()
        resp = client.chat(make_messages("hi"))
        assert resp.role == MessageRole.ASSISTANT
        assert "[NullLLMClient]" in resp.content
        assert resp.finish_reason == "null"

    def test_custom_canned_response(self):
        client = NullLLMClient(canned_response="custom placeholder")
        resp = client.chat(make_messages())
        assert resp.content == "custom placeholder"

    def test_no_network_access(self):
        """NullLLMClient 不应访问网络或抛错。"""
        client = NullLLMClient()
        # 即使没有 api_key 也能跑
        for _ in range(3):
            resp = client.chat(make_messages())
            assert resp is not None

    def test_call_count_increments(self):
        client = NullLLMClient()
        assert client.call_count == 0
        client.chat(make_messages())
        client.chat(make_messages())
        assert client.call_count == 2

    def test_warning_logged_once(self, caplog):
        client = NullLLMClient()
        with caplog.at_level(logging.WARNING, logger="llm.NullLLMClient"):
            client.chat(make_messages())
            client.chat(make_messages())
            client.chat(make_messages())
        warnings = [r for r in caplog.records if "NullLLMClient" in r.message]
        assert len(warnings) == 1  # 只警告一次

    def test_chat_accepts_dict_messages(self):
        client = NullLLMClient()
        resp = client.chat([{"role": "user", "content": "hi"}])
        assert resp.content.startswith("[NullLLMClient]")


# ---------------------------------------------------------------------------
# RetryingLLMClient (Decorator 1)
# ---------------------------------------------------------------------------

class TestRetryingLLMClient:
    def test_no_retry_on_success(self):
        inner = FakeLLMClient(response_content="ok")
        client = RetryingLLMClient(inner, max_retries=3, initial_backoff=0.001)
        resp = client.chat(make_messages())
        assert resp.content == "ok"
        assert inner.call_count == 1
        assert client.total_retries == 0

    def test_retry_on_rate_limit_eventually_succeeds(self):
        inner = FakeLLMClient(fail_times=2, fail_with=RateLimitError)
        client = RetryingLLMClient(inner, max_retries=3, initial_backoff=0.001)
        resp = client.chat(make_messages())
        assert resp.content == "fake response"
        assert inner.call_count == 3  # 2 fails + 1 success
        assert client.total_retries == 2

    def test_retry_exhausted_raises_last_exception(self):
        inner = FakeLLMClient(fail_times=10, fail_with=RateLimitError)
        client = RetryingLLMClient(inner, max_retries=2, initial_backoff=0.001)
        with pytest.raises(RateLimitError):
            client.chat(make_messages())
        assert inner.call_count == 3  # initial + 2 retries
        assert client.total_retries == 2

    def test_retry_on_api_error(self):
        inner = FakeLLMClient(fail_times=1, fail_with=APIError)
        client = RetryingLLMClient(
            inner, max_retries=2, initial_backoff=0.001,
            retry_on=(RateLimitError, APIError),
        )
        client.chat(make_messages())
        assert inner.call_count == 2

    def test_no_retry_on_non_retryable_exception(self):
        """AuthenticationError 不在 retry_on 中, 直接抛。"""
        class AuthOnlyClient(LLMClientBase):
            def __init__(self):
                super().__init__(api_key="x", max_retries=0)
            def _call_api(self, messages, model=None, **kwargs):
                raise AuthenticationError("auth fail")
        inner = AuthOnlyClient()
        client = RetryingLLMClient(inner, max_retries=3, initial_backoff=0.001)
        with pytest.raises(AuthenticationError):
            client.chat(make_messages())

    def test_exponential_backoff_timing(self):
        inner = FakeLLMClient(fail_times=10, fail_with=RateLimitError)
        client = RetryingLLMClient(
            inner, max_retries=3,
            initial_backoff=0.01, backoff_factor=10.0,
        )
        start = time.perf_counter()
        with pytest.raises(RateLimitError):
            client.chat(make_messages())
        elapsed = time.perf_counter() - start
        # 3 retries × ~exponential backoff ≈ 0.01 + 0.1 + 1.0 = 1.11s
        # max_retries=3 意味着初次 + 3 retries = 4 calls, 3 sleeps
        assert elapsed > 0.5  # 至少 0.01 + 0.1 + 1.0 - epsilon
        assert elapsed < 3.0  # 合理上限

    def test_get_model_list_passthrough(self):
        inner = FakeLLMClient()
        client = RetryingLLMClient(inner, max_retries=0)
        assert client.get_model_list() == ["fake-model"]


# ---------------------------------------------------------------------------
# LoggingLLMClient (Decorator 2)
# ---------------------------------------------------------------------------

class TestLoggingLLMClient:
    def test_logs_successful_call(self, caplog):
        inner = FakeLLMClient(response_content="hello world")
        client = LoggingLLMClient(inner, log_level=logging.INFO)
        with caplog.at_level(logging.INFO, logger="llm.LoggingLLMClient"):
            client.chat(make_messages(), model="test-model")
        log_text = "\n".join(r.message for r in caplog.records)
        assert "[LLMCall]" in log_text
        assert "model=test-model" in log_text
        # response_len is len("hello world") = 11
        assert "response_len=11" in log_text
        # prompt_tokens 来自 messages 内容, 不来自 FakeLLMClient
        assert "prompt_tokens=" in log_text

    def test_records_call_log(self):
        inner = FakeLLMClient()
        client = LoggingLLMClient(inner)
        client.chat(make_messages("first"), model="m1")
        client.chat(make_messages("second longer"), model="m2")
        assert len(client.call_log) == 2
        assert client.call_log[0]["model"] == "m1"
        assert client.call_log[1]["model"] == "m2"
        assert client.call_log[0]["error"] is None
        assert client.call_log[1]["latency_ms"] > 0

    def test_records_error_in_log(self):
        class FailingClient(LLMClientBase):
            def __init__(self):
                super().__init__(api_key="x", max_retries=0)
            def _call_api(self, messages, model=None, **kwargs):
                raise APIError("down")
        client = LoggingLLMClient(FailingClient())
        with pytest.raises(APIError):
            client.chat(make_messages())
        assert len(client.call_log) == 1
        assert "APIError" in client.call_log[0]["error"]

    def test_latency_measured(self):
        class SlowClient(LLMClientBase):
            def __init__(self):
                super().__init__(api_key="x", max_retries=0)
            def _call_api(self, messages, model=None, **kwargs):
                time.sleep(0.05)
                return ChatCompletion(content="slow", role=MessageRole.ASSISTANT)
        client = LoggingLLMClient(SlowClient())
        client.chat(make_messages())
        assert client.call_log[0]["latency_ms"] >= 50


# ---------------------------------------------------------------------------
# TokenCountingLLMClient (Decorator 3, Phase 1.2)
# ---------------------------------------------------------------------------

class TestTokenCountingLLMClient:
    def test_accumulates_tokens(self):
        inner = FakeLLMClient(prompt_tokens=10, completion_tokens=5)
        client = TokenCountingLLMClient(inner)
        client.chat(make_messages())
        client.chat(make_messages())
        client.chat(make_messages())
        assert client.total_prompt_tokens == 30
        assert client.total_completion_tokens == 15
        assert client.total_tokens_used == 45
        assert client.call_count == 3

    def test_no_usage_in_response(self):
        class NoUsageClient(LLMClientBase):
            def __init__(self):
                super().__init__(api_key="x", max_retries=0)
            def _call_api(self, messages, model=None, **kwargs):
                return ChatCompletion(content="x", role=MessageRole.ASSISTANT)  # no usage
        client = TokenCountingLLMClient(NoUsageClient())
        client.chat(make_messages())
        assert client.total_tokens_used == 0
        assert client.call_count == 1

    def test_reset(self):
        inner = FakeLLMClient()
        client = TokenCountingLLMClient(inner)
        client.chat(make_messages())
        assert client.total_tokens_used > 0
        client.reset()
        assert client.total_tokens_used == 0
        assert client.call_count == 0


# ---------------------------------------------------------------------------
# CachedLLMClient (Decorator 4, Phase 1.2)
# ---------------------------------------------------------------------------

class TestCachedLLMClient:
    def test_cache_miss_then_hit(self):
        inner = FakeLLMClient()
        client = CachedLLMClient(inner, max_size=10)
        client.chat(make_messages("hello"))
        client.chat(make_messages("hello"))  # same prompt → cache hit
        assert inner.call_count == 1
        assert client.hit_count == 1
        assert client.miss_count == 1

    def test_different_prompts_not_cached(self):
        inner = FakeLLMClient()
        client = CachedLLMClient(inner, max_size=10)
        client.chat(make_messages("a"))
        client.chat(make_messages("b"))
        assert inner.call_count == 2
        assert client.miss_count == 2
        assert client.hit_count == 0

    def test_lru_eviction(self):
        inner = FakeLLMClient()
        client = CachedLLMClient(inner, max_size=2)
        client.chat(make_messages("a"))  # miss 1
        client.chat(make_messages("b"))  # miss 2
        client.chat(make_messages("c"))  # miss 3, evicts "a"
        assert len(client._cache) == 2
        client.chat(make_messages("a"))  # miss 4 (a was evicted)
        assert inner.call_count == 4
        assert client.miss_count == 4
        assert client.hit_count == 0

    def test_cache_key_includes_model(self):
        inner = FakeLLMClient()
        client = CachedLLMClient(inner, max_size=10)
        client.chat(make_messages("x"), model="m1")
        client.chat(make_messages("x"), model="m2")
        assert inner.call_count == 2  # different model = different cache entry

    def test_clear(self):
        inner = FakeLLMClient()
        client = CachedLLMClient(inner, max_size=10)
        client.chat(make_messages("x"))
        client.clear()
        assert len(client._cache) == 0
        assert client.hit_count == 0
        assert client.miss_count == 0


# ---------------------------------------------------------------------------
# Decorator composition (chained decorators)
# ---------------------------------------------------------------------------

class TestDecoratorComposition:
    def test_retrying_wraps_logging_wraps_inner(self):
        """推荐组合顺序: Retrying(Logging(Inner))"""
        inner = FakeLLMClient(fail_times=1, fail_with=RateLimitError)
        logging_client = LoggingLLMClient(inner)
        retrying = RetryingLLMClient(logging_client, max_retries=2, initial_backoff=0.001)
        resp = retrying.chat(make_messages())
        assert resp.content == "fake response"
        assert inner.call_count == 2
        # 内部 logging 也记录了 2 次
        assert len(logging_client.call_log) == 2

    def test_caching_short_circuits_retrying(self):
        """Cached 在 Retrying 外面: 第二次调用不触发 inner 也不重试。"""
        inner = FakeLLMClient()
        retrying = RetryingLLMClient(inner, max_retries=3, initial_backoff=0.001)
        cached = CachedLLMClient(retrying, max_size=10)
        cached.chat(make_messages("a"))
        cached.chat(make_messages("a"))  # cache hit, no retry
        assert inner.call_count == 1

    def test_token_counting_with_logging(self):
        inner = FakeLLMClient(prompt_tokens=7, completion_tokens=3)
        token = TokenCountingLLMClient(inner)
        logged = LoggingLLMClient(token)
        logged.chat(make_messages())
        assert token.total_tokens_used == 10

    def test_null_with_decorators(self):
        """NullLLMClient 可以被装饰器包装 (用于测试环境)。"""
        null = NullLLMClient()
        token = TokenCountingLLMClient(null)
        logged = LoggingLLMClient(token)
        resp = logged.chat(make_messages())
        assert "[NullLLMClient]" in resp.content
        # NullLLMClient 返回 usage 0
        assert token.total_tokens_used == 0
