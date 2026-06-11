"""FactorFeedback — 结构化反馈框架。

公开 API:
    - FeedbackChannel (Enum): 5 通道
    - ChannelFeedback (dataclass): 单通道信号
    - FactorFeedback (dataclass): 完整反馈
    - FeedbackCollector: 聚合器
    - LLMJudge: LLM 一致性评判
    - ensure_feedback: dict → FactorFeedback 包装
    - collect_execution / collect_shape / collect_code / collect_value: 4 通道采集器
"""
from .dataclass import (
    ChannelFeedback,
    FactorFeedback,
    FeedbackChannel,
    ensure_feedback,
)
from .collector import FeedbackCollector
from .llm_judge import LLMJudge
from .channels import (
    collect_execution,
    collect_shape,
    collect_code,
    collect_value,
)

__all__ = [
    "FeedbackChannel",
    "ChannelFeedback",
    "FactorFeedback",
    "FeedbackCollector",
    "LLMJudge",
    "ensure_feedback",
    "collect_execution",
    "collect_shape",
    "collect_code",
    "collect_value",
]
