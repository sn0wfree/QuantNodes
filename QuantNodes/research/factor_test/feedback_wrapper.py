# coding: utf-8
"""FactorFeedback 包装器 / Feedback Wrapper.

把 5 个分析节点 (IC/Group/LongShort/Score/RiskCorrelation) 的返回值
统一包装成 ``FactorFeedback`` 对象, 聚合到 ``ctx['Feedback']``,
并可选持久化到 Parquet.

Phase R2 (2026-06-19): 从 pipeline_runner.py 抽出, 单一职责.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from QuantNodes.core.feedback import (
    FactorFeedback,
    FeedbackChannel,
    FeedbackCollector,
    LLMJudge,
    ensure_feedback,
)
from QuantNodes.core.path_utils import ensure_dir
from QuantNodes.research.factor_test.config import SingleFactorTestConfig


ANALYSIS_NODES: tuple[str, ...] = (
    "ICAnalyzer", "GroupAnalyzer", "LongShort", "FactorScore", "RiskCorrelation",
)


def build_feedback(
    ctx: dict,
    factor_id: str,
    factor_name: str,
    cfg: SingleFactorTestConfig,
    judge: Optional[LLMJudge],
) -> dict[str, FactorFeedback]:
    """包装 5 个分析节点返回值为 FactorFeedback.

    包装策略:
        - 节点返回 dict → ``ensure_feedback()`` 创建 metadata-only FactorFeedback
        - 节点已是 FactorFeedback → 直接用其 channels
        - 都通过同一个 FeedbackCollector 聚合, 共享 ``factor_id``
        - ``judge`` 不为 None 时追加 LLM 一致性通道
    """
    feedbacks: dict[str, FactorFeedback] = {}
    for node_name in ANALYSIS_NODES:
        result = ctx.get(node_name)
        if result is None:
            continue
        collector = FeedbackCollector(factor_id, factor_name)
        fb = ensure_feedback(result, factor_id, factor_name)
        for _ch, ch_fb in fb.channels.items():
            collector.add_feedback(ch_fb)
        if not fb.channels:
            collector.add(
                channel=FeedbackChannel.VALUE,
                passed=fb.decision,
                detail=fb.summary or f"{node_name} 节点无显式通道反馈",
                score=1.0 if fb.decision else 0.0,
            )
        if judge is not None:
            hypothesis = getattr(cfg.factor, "hypothesis", "") or ""
            description = getattr(cfg.factor, "description", "") or ""
            expression = getattr(cfg.factor, "expression", "") or ""
            if hypothesis or description or expression:
                llm_fb = judge.judge(hypothesis, description, expression)
                collector.add_feedback(llm_fb)
        feedbacks[node_name] = collector.finalize(
            summary=fb.summary or f"{node_name} 节点执行完成",
        )
    return feedbacks


def maybe_persist_feedback(
    feedbacks: dict[str, FactorFeedback],
    cfg: SingleFactorTestConfig,
) -> None:
    """可选: 持久化 Feedback 到 Parquet.

    Args:
        feedbacks: ``build_feedback`` 返回的 dict
        cfg: SingleFactorTestConfig, 需 ``cfg.feedback.output_dir`` 不为 None
    """
    if cfg.feedback.output_dir is None:
        return
    out = Path(cfg.feedback.output_dir)
    ensure_dir(out)
    parquet_path = out / "feedback.parquet"
    for node_name, fb in feedbacks.items():
        fb.save_parquet(parquet_path)


def maybe_build_judge(cfg: SingleFactorTestConfig) -> Optional[LLMJudge]:
    """若 ``cfg.feedback.judge_enabled`` 为 True, 构建 LLMJudge; 否则 None."""
    if not cfg.feedback.judge_enabled:
        return None
    return LLMJudge(
        model=cfg.feedback.judge_model,
        max_correction_attempts=cfg.feedback.judge_max_attempts,
    )
