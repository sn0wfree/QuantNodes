# coding=utf-8
"""
QuantNodes LLM Provider适配器

适配现有LLMClientBase到Agent Provider接口
"""

from typing import Any, Dict, List
import asyncio
import json

from .base import LLMProvider, LLMResponse, ToolCallRequest
from QuantNodes.ai.llm.base import LLMClientBase, Message as QNMessage, MessageRole


class QuantNodesLLMProvider(LLMProvider):
    """适配QuantNodes现有LLM客户端的Provider"""

    def __init__(self, client: LLMClientBase, default_model: str | None = None):
        super().__init__()
        self.client = client
        self.default_model = default_model

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[QNMessage]:
        """将OpenAI格式消息转换为QuantNodes格式"""
        result = []
        for msg in messages:
            role = MessageRole(msg.get("role", "user"))
            content = msg.get("content", "")
            result.append(QNMessage(role=role, content=content))
        return result

    def _parse_tool_calls(self, response_content: str) -> List[ToolCallRequest]:
        """从响应中解析工具调用"""
        tool_calls = []
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
        max_tokens: int = 1024,
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

        def _call():
            return self.client.chat(
                messages=qn_messages,
                model=model or self.default_model,
                temperature=temperature,
                max_tokens=max_tokens
            )

        qn_response = await asyncio.get_event_loop().run_in_executor(None, _call)

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
