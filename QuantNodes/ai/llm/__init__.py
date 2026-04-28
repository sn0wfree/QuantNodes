# coding=utf-8
"""
LLM 模块

提供 LLM 客户端接口。
"""

from QuantNodes.ai.llm.base import (
    LLMClientBase,
    LLMError,
    RateLimitError,
    AuthenticationError,
    APIError,
    Message,
    MessageRole,
    ChatCompletion,
    ChatCompletionChunk,
)

from QuantNodes.ai.llm.openai import (
    OpenAIClient,
    AzureOpenAIClient,
)

__all__ = [
    # Base classes
    'LLMClientBase',
    'LLMError',
    'RateLimitError',
    'AuthenticationError',
    'APIError',
    'Message',
    'MessageRole',
    'ChatCompletion',
    'ChatCompletionChunk',

    # Implementations
    'OpenAIClient',
    'AzureOpenAIClient',
]