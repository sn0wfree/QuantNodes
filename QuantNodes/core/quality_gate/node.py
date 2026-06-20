"""QualityGateNode — 集成 3 个 checkers, 输出 passed + FactorFeedback。"""
from __future__ import annotations

import uuid
from typing import Any

from ..feedback import (
    FeedbackCollector,
)
from .complexity import ComplexityChecker
from .consistency import ConsistencyChecker
from .redundancy import RedundancyChecker
from .settings import QualityGateSetting
from .zoo import FactorZoo


class QualityGateNode:
    """质量门节点 — pre-backtest 检查。

    输入: context['FactorCandidate'] = {
        'factor_id': str,
        'name': str,
        'expression': str,
        'hypothesis': str,
        'description': str,
    }
    输出: {
        'passed': bool,
        'feedback': FactorFeedback,
        'channels': dict[FeedbackChannel, ChannelFeedback],
    }
    """

    def __init__(
        self,
        settings: QualityGateSetting | None = None,
        zoo: FactorZoo | None = None,
    ):
        self.settings = settings or QualityGateSetting()
        self._complexity = ComplexityChecker(self.settings.complexity)
        self._redundancy = RedundancyChecker(
            self.settings.redundancy,
            zoo=zoo if zoo is not None else FactorZoo(self.settings.redundancy.zoo_path),
        )
        self._consistency = ConsistencyChecker(self.settings.consistency)

    def check(
        self,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        """同步检查入口, 不依赖 BaseNode 框架 (便于单测)。

        Args:
            candidate: 因子候选 dict (含 expression / hypothesis / description)

        Returns:
            dict: {'passed', 'feedback', 'channels'}
        """
        if "expression" not in candidate:
            raise ValueError("FactorCandidate 缺少 'expression' 字段")

        factor_id = candidate.get("factor_id") or str(uuid.uuid4())
        factor_name = candidate.get("name", "unnamed")
        expression = candidate["expression"]
        hypothesis = candidate.get("hypothesis", "")
        description = candidate.get("description", "")

        collector = FeedbackCollector(factor_id, factor_name)

        if self.settings.complexity.enabled:
            collector.add_feedback(self._complexity.check(expression))
        if self.settings.redundancy.enabled:
            collector.add_feedback(self._redundancy.check(expression))
        if self.settings.consistency.enabled:
            collector.add_feedback(
                self._consistency.check(hypothesis, description, expression)
            )

        feedback = collector.finalize()
        return {
            "passed": feedback.decision,
            "feedback": feedback,
            "channels": feedback.channels,
        }

    def execute(self, context: dict | None = None, **kwargs) -> dict[str, Any]:
        """节点风格入口, 从 context['FactorCandidate'] 读取。

        Raises:
            ValueError: FactorCandidate 缺失
        """
        if context is None:
            context = kwargs.get("context", {})
        candidate = context.get("FactorCandidate")
        if not candidate:
            raise ValueError("FactorCandidate 缺失 (context['FactorCandidate'] 不存在)")
        return self.check(candidate)
