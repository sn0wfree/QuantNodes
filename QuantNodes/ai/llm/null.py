# coding=utf-8
"""NullLLMClient (Null Object pattern, Phase 1.1).

替代 None 检查, 当无 LLM 客户端可用时返回确定性 canned response + warning 日志。
避免调用方散落的 `if self._llm_client is None` 模式 (见 agent/tools/strategy.py)。
"""
from __future__ import annotations

from typing import Any, List, Optional

from QuantNodes.ai.llm.base import (
    ChatCompletion,
    LLMClientBase,
    Message,
    MessageRole,
)


class NullLLMClient(LLMClientBase):
    """LLM 客户端的空对象 (Null Object pattern)。

    行为:
      - chat() 返回固定 canned response, 内容含标识 `[NullLLMClient]`
      - 每次调用 logger.warning 一次 (避免刷屏, 用 _warned 标记)
      - 不抛异常, 不访问网络

    使用场景:
      - 单元测试中无需真实 LLM
      - 配置缺失或 API key 未设置时的 fallback
      - Agent 框架中的 "无 LLM 模式" 占位

    Args:
        canned_response: 固定返回内容, 默认带 [NullLLMClient] 前缀
    """

    def __init__(
        self,
        canned_response: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # 避免父类要求 api_key, base_url 等真实字段
        super().__init__(api_key=None, base_url=None, timeout=0, max_retries=0, **kwargs)
        self.canned_response = canned_response or (
            "[NullLLMClient] no real LLM configured; "
            "this is a deterministic placeholder response."
        )
        self._warned = False
        self.call_count = 0

    def _call_api(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ChatCompletion:
        if not self._warned:
            self.logger.warning(
                "NullLLMClient is in use; downstream code will receive "
                "deterministic placeholder responses."
            )
            self._warned = True
        self.call_count += 1
        return ChatCompletion(
            content=self.canned_response,
            role=MessageRole.ASSISTANT,
            finish_reason="null",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    def _call_api_stream(self, messages: List[Message], model: Optional[str] = None, **kwargs: Any):
        if not self._warned:
            self.logger.warning("NullLLMClient is in use (stream).")
            self._warned = True
        self.call_count += 1
        from QuantNodes.ai.llm.base import ChatCompletionChunk
        yield ChatCompletionChunk(content=self.canned_response, finish_reason="null")
