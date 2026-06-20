# coding=utf-8
"""
Context Compaction - Intelligent context compression using LLM summarization

Phase 5: Replace simple truncation with LLM-powered summarization
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from ..providers.base import LLMProvider


@dataclass
class CompactionConfig:
    """Configuration for context compaction"""
    enabled: bool = True
    target_tokens: int = 8000
    max_messages: int = 50
    min_messages_to_compact: int = 10
    summarization_threshold: float = 0.3
    use_llm_summarization: bool = True
    llm_model: Optional[str] = None


@dataclass
class CompactionResult:
    """Result of compaction operation"""
    messages: List[Dict[str, Any]]
    compacted_count: int
    original_count: int
    used_summarization: bool
    summarization: Optional[str] = None


class ContextCompactor:
    """Intelligent context compactor

    Features:
    1. Preserves system messages
    2. Keeps recent N messages
    3. Summarizes middle messages using LLM (optional)
    4. Falls back to simple truncation if LLM unavailable
    """

    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        config: Optional[CompactionConfig] = None,
    ):
        self._provider = provider
        self._config = config or CompactionConfig()

    async def compact(
        self,
        messages: List[Dict[str, Any]],
        target_tokens: Optional[int] = None,
    ) -> CompactionResult:
        """Compact messages to fit within token budget

        Args:
            messages: List of message dictionaries
            target_tokens: Target token count (uses config default if None)

        Returns:
            CompactionResult with compacted messages
        """
        if not messages:
            return CompactionResult(
                messages=[],
                compacted_count=0,
                original_count=0,
                used_summarization=False,
            )

        original_count = len(messages)

        if not self._should_compact(messages):
            return CompactionResult(
                messages=messages,
                compacted_count=0,
                original_count=original_count,
                used_summarization=False,
            )

        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        if (
            self._config.use_llm_summarization
            and self._provider
            and len(other_msgs) >= self._config.min_messages_to_compact
        ):
            return await self._llm_compact(system_msgs, other_msgs, original_count)

        return self._simple_compact(system_msgs, other_msgs, original_count)

    def _should_compact(self, messages: List[Dict[str, Any]]) -> bool:
        """Determine if compaction is needed"""
        if len(messages) > self._config.max_messages:
            return True
        if len(messages) >= self._config.min_messages_to_compact:
            return True
        return False

    async def _llm_compact(
        self,
        system_msgs: List[Dict[str, Any]],
        other_msgs: List[Dict[str, Any]],
        original_count: int,
    ) -> CompactionResult:
        """Compact using LLM summarization for middle messages"""
        recent_count = self._config.max_messages // 2
        recent_msgs = other_msgs[-recent_count:] if len(other_msgs) > recent_count else other_msgs
        middle_msgs = other_msgs[:-recent_count] if len(other_msgs) > recent_count else []

        summarization = None
        used_summarization = False

        if middle_msgs and self._provider:
            try:
                summarization = await self._summarize_messages(middle_msgs)
                used_summarization = True
            except Exception:
                pass

        if summarization:
            summary_msg = {
                "role": "system",
                "content": f"[Previous context summarized]: {summarization}",
            }
            compacted = system_msgs + [summary_msg] + recent_msgs
        else:
            compacted = self._simple_compact(system_msgs, other_msgs, original_count).messages

        return CompactionResult(
            messages=compacted,
            compacted_count=len(middle_msgs),
            original_count=original_count,
            used_summarization=used_summarization,
            summarization=summarization,
        )

    async def _summarize_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> str:
        """Use LLM to summarize a list of messages"""
        if not self._provider:
            raise ValueError("No LLM provider configured")

        summary_prompt = self._build_summary_prompt(messages)

        response = await self._provider.chat(
            messages=[{"role": "user", "content": summary_prompt}],
            model=self._config.llm_model,
        )

        return response.content if hasattr(response, 'content') else str(response)

    def _build_summary_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Build prompt for summarization"""
        msg_summary = "\n".join([
            f"[{m.get('role', 'unknown')}]: {m.get('content', '')[:200]}"
            for m in messages[:20]
        ])
        return f"""Please summarize the following conversation context concisely:
{msg_summary}

Provide a brief summary (2-3 sentences max) of what was discussed and decided."""

    def _simple_compact(
        self,
        system_msgs: List[Dict[str, Any]],
        other_msgs: List[Dict[str, Any]],
        original_count: int,
    ) -> CompactionResult:
        """Simple truncation-based compaction"""
        max_keep = self._config.max_messages - len(system_msgs)
        kept_msgs = other_msgs[-max_keep:] if len(other_msgs) > max_keep else other_msgs
        result_messages = system_msgs + kept_msgs

        return CompactionResult(
            messages=result_messages,
            compacted_count=len(other_msgs) - len(kept_msgs),
            original_count=original_count,
            used_summarization=False,
        )


async def compact_messages(
    messages: List[Dict[str, Any]],
    provider: Optional[LLMProvider] = None,
    config: Optional[CompactionConfig] = None,
    target_tokens: Optional[int] = None,
) -> CompactionResult:
    """Convenience function for compacting messages"""
    compactor = ContextCompactor(provider=provider, config=config)
    return await compactor.compact(messages, target_tokens=target_tokens)
