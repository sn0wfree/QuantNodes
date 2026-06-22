# coding=utf-8
"""
LLM Client 基类

提供 LLM 客户端的统一接口。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

import logging

from QuantNodes.core.base import QuantNodesError


class MessageRole(str, Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """对话消息"""
    role: MessageRole
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


@dataclass
class ChatCompletion:
    """聊天补全结果"""
    content: str
    role: MessageRole = MessageRole.ASSISTANT
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


@dataclass
class ChatCompletionChunk:
    """聊天补全流式块"""
    content: str
    finish_reason: Optional[str] = None


class LLMError(QuantNodesError):
    """LLM 异常基类 (Phase 1.1: 统一异常层次, 继承 QuantNodesError)"""
    code = "LLM_ERROR"


class RateLimitError(LLMError):
    """速率限制异常"""
    code = "LLM_RATE_LIMIT"


class AuthenticationError(LLMError):
    """认证异常"""
    code = "LLM_AUTH"


class APIError(LLMError):
    """API 异常"""
    code = "LLM_API"


class LLMClientBase(ABC):
    """
    LLM 客户端基类

    提供统一的 LLM 调用接口。

    Subclasses must implement:
        _call_api(): 调用具体的 LLM API

    Examples:
        >>> client = OpenAIClient(api_key="sk-...")
        >>> response = client.chat([Message(role="user", content="Hello")])
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs
    ):
        """
        初始化 LLM 客户端

        Args:
            api_key: API 密钥
            base_url: API 基础 URL
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            **kwargs: 额外配置参数
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_config = kwargs
        self.logger = logging.getLogger(f"llm.{self.__class__.__name__}")

    @abstractmethod
    def _call_api(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs
    ) -> ChatCompletion:
        """
        调用具体的 LLM API

        Args:
            messages: 对话消息列表
            model: 模型名称
            **kwargs: 额外参数

        Returns:
            ChatCompletion 聊天补全结果
        """
        pass

    def chat(
        self,
        messages: Union[List[Message], List[Dict[str, str]]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, None]:
        """
        发送聊天请求

        Args:
            messages: 对话消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数
            stream: 是否流式返回
            **kwargs: 额外参数

        Returns:
            ChatCompletion 或 None（流式模式）
        """
        normalized_messages = self._normalize_messages(messages)
        return self._call_api(
            normalized_messages, model,
            temperature=temperature, max_tokens=max_tokens,
            stream=stream, **kwargs,
        )

    def _normalize_messages(
        self,
        messages: Union[List[Message], List[Dict[str, str]]]
    ) -> List[Message]:
        """规范化消息格式"""
        normalized = []
        for msg in messages:
            if isinstance(msg, Message):
                normalized.append(msg)
            elif isinstance(msg, dict):
                role = MessageRole(msg.get('role', 'user'))
                normalized.append(Message(
                    role=role,
                    content=msg.get('content', ''),
                    name=msg.get('name'),
                ))
            else:
                raise ValueError(f"Invalid message format: {type(msg)}")
        return normalized

    def chat_stream(
        self,
        messages: Union[List[Message], List[Dict[str, str]]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        发送聊天请求（流式）

        Args:
            messages: 对话消息列表
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数
            **kwargs: 额外参数

        Yields:
            ChatCompletionChunk 流式块
        """
        normalized_messages = self._normalize_messages(messages)
        for chunk in self._call_api_stream(
            normalized_messages, model,
            temperature=temperature, max_tokens=max_tokens, **kwargs,
        ):
            yield chunk

    def _call_api_stream(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs
    ):
        """流式调用 API（子类可重写）"""
        raise NotImplementedError("Streaming not supported by this client")

    def get_model_list(self) -> List[str]:
        """获取可用模型列表（子类可重写）"""
        return []

    def count_tokens(self, text: str) -> int:
        """估算 token 数量（粗略实现）"""
        return len(text) // 4

    def count_messages_tokens(self, messages: List[Message]) -> int:
        """估算消息列表的 token 数量"""
        total = 0
        for msg in messages:
            total += self.count_tokens(msg.content)
            total += 4
        return total
