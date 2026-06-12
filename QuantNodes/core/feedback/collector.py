"""FeedbackCollector — 聚合多通道反馈的便捷类。"""
from __future__ import annotations

import time
from typing import Optional

from .dataclass import ChannelFeedback, FactorFeedback, FeedbackChannel


class FeedbackCollector:
    """聚合多个通道的反馈信号, finalize() 返回 FactorFeedback。"""

    def __init__(self, factor_id: str, factor_name: str):
        self.factor_id = factor_id
        self.factor_name = factor_name
        self._channels: dict[FeedbackChannel, ChannelFeedback] = {}
        self._t0 = time.perf_counter()

    def add(
        self,
        channel: FeedbackChannel,
        passed: bool,
        detail: str,
        score: float = 1.0,
        **metadata,
    ) -> "FeedbackCollector":
        """添加一个通道的反馈, 支持链式调用。"""
        self._channels[channel] = ChannelFeedback(
            channel=channel,
            passed=passed,
            detail=detail,
            score=score,
            metadata=dict(metadata),
        )
        return self

    def add_feedback(self, fb: ChannelFeedback) -> "FeedbackCollector":
        """添加一个完整的 ChannelFeedback 对象。"""
        self._channels[fb.channel] = fb
        return self

    def has(self, channel: FeedbackChannel) -> bool:
        return channel in self._channels

    def get(self, channel: FeedbackChannel) -> Optional[ChannelFeedback]:
        return self._channels.get(channel)

    def finalize(
        self,
        decision: Optional[bool] = None,
        summary: str = "",
        agg_mode: str = "all",  # M3: "all" (AND) | "any" (OR) | "majority"
        **metadata,
    ) -> FactorFeedback:
        """聚合所有通道, 返回 FactorFeedback。

        Args:
            decision: 显式决策 (None=按 agg_mode 自动)
            summary: 一句话总结 (空=自动生成)
            agg_mode: M3 聚合方式
                - "all" (默认): 全部通过才算通过
                - "any": 任一通过就算通过
                - "majority": 多数通过才算通过
            **metadata: 附加到 FactorFeedback.metadata
        """
        if decision is None:
            if not self._channels:
                decision = True
            elif agg_mode == "all":
                decision = all(fb.passed for fb in self._channels.values())
            elif agg_mode == "any":
                decision = any(fb.passed for fb in self._channels.values())
            elif agg_mode == "majority":
                passed_count = sum(1 for fb in self._channels.values() if fb.passed)
                decision = passed_count > len(self._channels) / 2
            else:
                raise ValueError(f"未知 agg_mode: {agg_mode}")

        if not summary:
            failed = [ch.value for ch, fb in self._channels.items() if not fb.passed]
            summary = f"失败通道: {', '.join(failed)}" if failed else "全部通过"

        return FactorFeedback(
            factor_id=self.factor_id,
            factor_name=self.factor_name,
            channels=dict(self._channels),
            decision=decision,
            summary=summary,
            duration_ms=(time.perf_counter() - self._t0) * 1000,
            metadata=dict(metadata),
        )
