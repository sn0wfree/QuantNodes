"""RedundancyChecker — 与 Zoo 已有因子 AST hash 汉明距离 < 阈值则拒绝。"""
from __future__ import annotations

from ..feedback import ChannelFeedback, FeedbackChannel
from .settings import RedundancySetting
from .zoo import FactorZoo


class RedundancyChecker:
    """冗余门: 与 Zoo 中已有因子的 AST 哈希距离。"""

    def __init__(self, settings: RedundancySetting | None = None, zoo: FactorZoo | None = None):
        self.settings = settings or RedundancySetting()
        self.zoo = zoo if zoo is not None else FactorZoo(self.settings.zoo_path)

    def check(self, expression: str) -> ChannelFeedback:
        """返回 ChannelFeedback (VALUE 通道)。"""
        if not self.settings.enabled:
            return ChannelFeedback(
                channel=FeedbackChannel.VALUE,
                passed=True,
                detail="redundancy disabled",
            )

        if len(self.zoo) == 0:
            return ChannelFeedback(
                channel=FeedbackChannel.VALUE,
                passed=True,
                detail="Zoo 为空, 无需检查",
                metadata={"zoo_size": 0},
            )

        distances = self.zoo.hamming_to(expression)
        min_dist = distances[0][0]
        nearest = distances[0][2]
        passed = min_dist >= self.settings.threshold
        detail = (
            f"min_hamming_dist={min_dist}, threshold={self.settings.threshold}, "
            f"zoo_size={len(self.zoo)}, nearest={nearest[:50]!r}"
        )
        return ChannelFeedback(
            channel=FeedbackChannel.VALUE,
            passed=passed,
            detail=detail,
            score=1.0 if passed else 0.0,
            metadata={
                "min_hamming_dist": int(min_dist),
                "threshold": int(self.settings.threshold),
                "zoo_size": len(self.zoo),
            },
        )
