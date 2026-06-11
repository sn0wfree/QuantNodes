"""ConsistencyChecker — LLM 验证 hypothesis ↔ description ↔ expression。"""
from __future__ import annotations

from ..feedback import ChannelFeedback, FeedbackChannel, LLMJudge
from .settings import ConsistencySetting


class ConsistencyChecker:
    """一致性门: 复用 FactorFeedback 的 LLMJudge。"""

    def __init__(self, settings: ConsistencySetting | None = None, judge: LLMJudge | None = None):
        self.settings = settings or ConsistencySetting()
        self._judge = judge if judge is not None else LLMJudge(
            model=self.settings.model,
            max_correction_attempts=self.settings.max_correction_attempts,
        )

    def check(self, hypothesis: str, description: str, expression: str) -> ChannelFeedback:
        """返回 ChannelFeedback (LLM 通道)。"""
        if not self.settings.enabled:
            return ChannelFeedback(
                channel=FeedbackChannel.LLM,
                passed=True,
                detail="consistency disabled",
            )
        return self._judge.judge(hypothesis, description, expression)
