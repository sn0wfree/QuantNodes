# coding=utf-8
"""
上下文构建器

构建系统Prompt与消息上下文
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


class ContextBuilder:
    """构建系统Prompt与消息上下文"""

    def __init__(self, templates_dir: Path | str, timezone: str = "Asia/Shanghai"):
        self.templates_dir = Path(templates_dir)
        self.timezone = timezone
        self._system_prompt_cache: Optional[str] = None

    def load_system_prompt(self) -> str:
        """加载系统Prompt模板"""
        if self._system_prompt_cache is not None:
            return self._system_prompt_cache

        identity_file = self.templates_dir / "identity.md"
        system_file = self.templates_dir / "system_prompt.md"

        parts = []

        if identity_file.exists():
            with open(identity_file, "r", encoding="utf-8") as f:
                parts.append(f.read())

        if system_file.exists():
            with open(system_file, "r", encoding="utf-8") as f:
                parts.append(f.read())

        self._system_prompt_cache = "\n\n".join(parts)
        return self._system_prompt_cache

    def build_messages(
        self,
        history: List[Dict[str, Any]],
        current_message: str,
        media: List[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """构建完整的消息列表（系统Prompt + 历史 + 当前消息）"""
        messages = []

        system_prompt = self.load_system_prompt()
        if system_prompt:
            runtime_context = self._build_runtime_context(channel, chat_id)
            if runtime_context:
                system_prompt = f"{system_prompt}\n\n{runtime_context}"
            messages.append({"role": "system", "content": system_prompt})

        for msg in history:
            if msg.get("role") in ("system", "user", "assistant", "tool"):
                messages.append({"role": msg["role"], "content": msg.get("content", "")})

        if current_message:
            messages.append({"role": "user", "content": current_message})

        return messages

    def _build_runtime_context(
        self,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> str:
        """构建运行时上下文信息"""
        parts = []
        now = datetime.now().isoformat()
        parts.append(f"当前时间: {now}")
        parts.append(f"时区: {self.timezone}")
        if channel:
            parts.append(f"渠道: {channel}")
        if chat_id:
            parts.append(f"会话ID: {chat_id}")
        return "\n".join(parts)
