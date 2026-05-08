# coding=utf-8
"""
OpenAI 兼容 LLM 客户端

提供 OpenAI API 兼容的 LLM 调用接口。
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Iterator

import requests

from QuantNodes.ai.llm.base import (
    LLMClientBase,
    RateLimitError,
    AuthenticationError,
    APIError,
    Message,
    MessageRole,
    ChatCompletion,
    ChatCompletionChunk,
)


class OpenAIClient(LLMClientBase):
    """
    OpenAI API 客户端

    支持 OpenAI 兼容的 API 接口。

    Examples:
        >>> client = OpenAIClient(api_key="sk-...")
        >>> response = client.chat([Message(role="user", content="Hello")])
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs
    ):
        """
        初始化 OpenAI 客户端

        Args:
            api_key: API 密钥（默认从环境变量 OPENAI_API_KEY 获取）
            base_url: API 基础 URL
            model: 默认模型
            timeout: 请求超时时间
            max_retries: 最大重试次数
            **kwargs: 额外配置
        """
        super().__init__(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url or self.DEFAULT_BASE_URL,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )
        self.model = model or self.DEFAULT_MODEL

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _call_api(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> ChatCompletion:
        """
        调用 OpenAI API

        Args:
            messages: 对话消息
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大 token 数
            stream: 是否流式
            **kwargs: 额外参数

        Returns:
            ChatCompletion 结果
        """
        url = f"{self.base_url}/chat/completions"
        model = model or self.model

        payload = {
            "model": model,
            "messages": [self._message_to_dict(m) for m in messages],
            "temperature": temperature,
            "stream": stream,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        payload.update(kwargs)

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=self.timeout,
                    stream=stream,
                )

                if response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt
                        self.logger.warning(f"Rate limit hit, waiting {wait_time}s")
                        time.sleep(wait_time)
                        continue
                    raise RateLimitError("Rate limit exceeded")
                elif response.status_code == 401:
                    raise AuthenticationError("Invalid API key")
                elif response.status_code != 200:
                    raise APIError(f"API error: {response.status_code} - {response.text}")

                if stream:
                    return None

                data = response.json()
                return self._parse_response(data)

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    continue
                raise APIError("Request timeout")
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    continue
                raise APIError(f"Request failed: {str(e)}")

    def _call_api_stream(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Iterator[ChatCompletionChunk]:
        """流式调用 API"""
        url = f"{self.base_url}/chat/completions"
        model = model or self.model

        payload = {
            "model": model,
            "messages": [self._message_to_dict(m) for m in messages],
            "temperature": temperature,
            "stream": True,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        payload.update(kwargs)

        try:
            response = requests.post(
                url,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout,
                stream=True,
            )

            if response.status_code == 429:
                raise RateLimitError("Rate limit exceeded")
            elif response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif response.status_code != 200:
                raise APIError(f"API error: {response.status_code}")

            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        import json
                        chunk_data = json.loads(data)
                        yield self._parse_stream_chunk(chunk_data)
                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.RequestException as e:
            raise APIError(f"Stream request failed: {str(e)}")

    def _message_to_dict(self, message: Message) -> Dict[str, str]:
        """将 Message 转换为字典"""
        result = {
            "role": message.role.value,
            "content": message.content,
        }
        if message.name:
            result["name"] = message.name
        return result

    def _parse_response(self, data: Dict[str, Any]) -> ChatCompletion:
        """解析 API 响应"""
        choices = data.get("choices", [])
        if not choices:
            raise APIError("No choices in response")

        choice = choices[0]
        message_data = choice.get("message", {})

        return ChatCompletion(
            content=message_data.get("content", ""),
            role=MessageRole(message_data.get("role", "assistant")),
            finish_reason=choice.get("finish_reason"),
            usage=data.get("usage"),
        )

    def _parse_stream_chunk(self, data: Dict[str, Any]) -> ChatCompletionChunk:
        """解析流式 chunk"""
        choices = data.get("choices", [])
        if not choices:
            return ChatCompletionChunk(content="", finish_reason=None)

        choice = choices[0]
        delta = choice.get("delta", {})

        return ChatCompletionChunk(
            content=delta.get("content", ""),
            finish_reason=choice.get("finish_reason"),
        )

    def get_model_list(self) -> List[str]:
        """获取可用模型列表"""
        url = f"{self.base_url}/models"

        try:
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=self.timeout,
            )

            if response.status_code != 200:
                return [self.model]

            data = response.json()
            return [m["id"] for m in data.get("data", [])]

        except requests.exceptions.RequestException:
            return [self.model]


class AzureOpenAIClient(OpenAIClient):
    """Azure OpenAI 客户端"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: str = "2024-02-01",
        deployment_name: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs
    ):
        """
        初始化 Azure OpenAI 客户端

        Args:
            api_key: API 密钥
            azure_endpoint: Azure 端点
            api_version: API 版本
            deployment_name: 部署名称
            timeout: 超时时间
            max_retries: 最大重试次数
        """
        super().__init__(
            api_key=api_key,
            base_url=azure_endpoint,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )
        self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self.api_version = api_version
        self.deployment_name = deployment_name

    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }
        return headers

    def _call_api(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> ChatCompletion:
        """调用 Azure OpenAI API"""
        if self.deployment_name:
            model = self.deployment_name

        url = f"{self.azure_endpoint}/openai/deployments/{model}/chat/completions"
        params = {"api-version": self.api_version}

        payload = {
            "messages": [self._message_to_dict(m) for m in messages],
            "temperature": temperature,
            "stream": stream,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        payload.update(kwargs)

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    headers=self._get_headers(),
                    json=payload,
                    params=params,
                    timeout=self.timeout,
                    stream=stream,
                )

                if response.status_code == 429:
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    raise RateLimitError("Rate limit exceeded")
                elif response.status_code == 401:
                    raise AuthenticationError("Invalid API key")
                elif response.status_code != 200:
                    raise APIError(f"API error: {response.status_code} - {response.text}")

                if stream:
                    return None

                data = response.json()
                return self._parse_response(data)

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    continue
                raise APIError("Request timeout")
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    continue
                raise APIError(f"Request failed: {str(e)}")