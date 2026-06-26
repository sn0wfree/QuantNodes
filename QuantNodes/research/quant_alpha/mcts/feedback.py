# coding=utf-8
"""
mcts/feedback.py - MCTS 5 通道反馈采集（基于 OperatorVocab）

vs 旧 mcts_search.py 缺 5 通道反馈：
- 旧：只用 `dimension_scores: Dict[str, float]` 一个 dict 装所有维度
- 新：复用 core/feedback.FactorFeedback 完整 5 通道框架
       （execution / shape / code / value / llm）

5 通道：
- execution: 公式评估是否成功（无异常）
- shape: 输出形状 vs 预期（长度匹配）
- code: AST 静态检查（防过拟合：长度/特征数/自由参数比例）
- value: 数值分布（NaN 比例/Inf 数量/标准差）
- llm: hypothesis ↔ expression 一致性（M5+ 接入真实 LLM，M2 用 mock）
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import polars as pl

from QuantNodes.core.constants import BASE_FEATURE_NAMES
from QuantNodes.core.feedback import (
    ChannelFeedback,
    FactorFeedback,
    FeedbackChannel,
)
from QuantNodes.research.quant_alpha.operator_vocab import OperatorVocab

logger = logging.getLogger(__name__)


# 默认阈值
DEFAULT_COMPLEXITY_CONFIG = {
    "symbol_length_threshold": 200,
    "base_features_threshold": 5,
    "free_args_ratio_threshold": 0.5,
    "nan_threshold": 0.3,
    "std_threshold": 1e-6,
}


@dataclass
class MCTSFeedbackConfig:
    """MCTS 5 通道反馈配置"""
    # CODE 通道阈值
    symbol_length_threshold: int = 200
    base_features_threshold: int = 5
    free_args_ratio_threshold: float = 0.5
    # VALUE 通道阈值
    nan_threshold: float = 0.3
    std_threshold: float = 1e-6
    # 通道开关（True=启用，False=禁用）
    enable_execution: bool = True
    enable_shape: bool = True
    enable_code: bool = True
    enable_value: bool = True
    enable_llm: bool = False  # M2 暂用 mock，M5+ 接入真实 LLM


# ==============================================================================
# 5 通道采集器
# ==============================================================================


def collect_execution_channel(
    formula: str,
    exception: Optional[Exception] = None,
) -> ChannelFeedback:
    """EXECUTION 通道：公式评估是否成功"""
    passed = exception is None
    if passed:
        detail = "OK (no exception)"
        score = 1.0
    else:
        detail = f"FAIL: {type(exception).__name__}: {str(exception)[:500]}"
        score = 0.0
    return ChannelFeedback(
        channel=FeedbackChannel.EXECUTION,
        passed=passed,
        detail=detail,
        score=score,
        metadata={"exception_type": type(exception).__name__ if exception else None},
    )


def collect_shape_channel(
    result: Optional[pl.Series],
    expected_length: int,
) -> ChannelFeedback:
    """SHAPE 通道：输出形状 vs 预期"""
    if result is None:
        return ChannelFeedback(
            channel=FeedbackChannel.SHAPE,
            passed=False,
            detail="result is None",
            score=0.0,
        )
    actual_length = len(result)
    passed = actual_length == expected_length
    detail = f"actual_length={actual_length}, expected={expected_length}"
    score = 1.0 if passed else 0.0
    return ChannelFeedback(
        channel=FeedbackChannel.SHAPE,
        passed=passed,
        detail=detail,
        score=score,
        metadata={"actual_length": actual_length, "expected_length": expected_length},
    )


def collect_code_channel(
    formula: str,
    config: MCTSFeedbackConfig,
) -> ChannelFeedback:
    """CODE 通道：AST 静态检查（防过拟合）

    注意：公式使用 OperatorVocab 语法（如 rank(ts_mean(close, 20))），
    不是标准 Python 语法。因此只做基本检查，不做 AST 解析。
    """
    symbol_length = len(formula)
    base_features = _count_base_features(formula)
    free_args_ratio = _calc_free_args_ratio(formula)

    violations: List[str] = []
    if symbol_length > config.symbol_length_threshold:
        violations.append(
            f"length={symbol_length}>{config.symbol_length_threshold}"
        )
    if base_features > config.base_features_threshold:
        violations.append(
            f"features={base_features}>{config.base_features_threshold}"
        )
    if free_args_ratio > config.free_args_ratio_threshold:
        violations.append(
            f"free_args={free_args_ratio:.2f}>{config.free_args_ratio_threshold}"
        )

    # 基本括号匹配检查
    open_parens = formula.count("(")
    close_parens = formula.count(")")
    if open_parens != close_parens:
        violations.append(f"parentheses mismatch: {open_parens} vs {close_parens}")

    passed = len(violations) == 0
    detail = (
        "; ".join(violations) if violations else
        f"OK (length={symbol_length}, features={base_features}, "
        f"free_args={free_args_ratio:.2f})"
    )
    score = 1.0 if passed else 0.0
    return ChannelFeedback(
        channel=FeedbackChannel.CODE,
        passed=passed,
        detail=detail,
        score=score,
        metadata={
            "symbol_length": symbol_length,
            "base_features": base_features,
            "free_args_ratio": free_args_ratio,
        },
    )


def collect_value_channel(
    result: Optional[pl.Series],
    config: MCTSFeedbackConfig,
) -> ChannelFeedback:
    """VALUE 通道：数值分布合理性"""
    if result is None:
        return ChannelFeedback(
            channel=FeedbackChannel.VALUE,
            passed=False,
            detail="result is None",
            score=0.0,
        )

    # 转为 numpy 数组
    arr = result.to_numpy()
    nan_count = int(np.isnan(arr).sum()) if arr.dtype.kind == 'f' else 0
    total = len(arr)
    nan_pct = nan_count / total if total > 0 else 0.0

    # Inf
    inf_count = 0
    if arr.dtype.kind == 'f':
        inf_count = int(np.isinf(arr).sum())

    # 标准差（非 NaN/Inf）
    clean = arr[~np.isnan(arr) & ~np.isinf(arr)] if arr.dtype.kind == 'f' else arr
    std_val = float(np.std(clean)) if len(clean) > 1 else 0.0
    mean_val = float(np.mean(clean)) if len(clean) > 0 else 0.0

    violations: List[str] = []
    if nan_pct > config.nan_threshold:
        violations.append(f"NaN={nan_pct:.2%}>{config.nan_threshold:.0%}")
    if inf_count > 0:
        violations.append(f"Inf={inf_count}>0")
    if std_val <= config.std_threshold:
        violations.append(f"std={std_val:.6f}<={config.std_threshold}")

    passed = len(violations) == 0
    detail = (
        "; ".join(violations) if violations else
        f"OK (NaN={nan_pct:.2%}, mean={mean_val:.4f}, std={std_val:.4f})"
    )
    score = 1.0 if passed else 0.0
    return ChannelFeedback(
        channel=FeedbackChannel.VALUE,
        passed=passed,
        detail=detail,
        score=score,
        metadata={
            "nan_pct": nan_pct,
            "inf_count": inf_count,
            "mean": mean_val,
            "std": std_val,
        },
    )


def collect_llm_channel(
    formula: str,
    hypothesis: Optional[str] = None,
    description: Optional[str] = None,
) -> ChannelFeedback:
    """LLM 通道：hypothesis ↔ expression 一致性

    M2: 使用 mock 简单实现（关键字匹配）
    M5+: 接入真实 LLM judge
    """
    # Mock: 简单关键字匹配
    if hypothesis is None or description is None:
        # 没有 hypothesis/description 时默认 pass
        return ChannelFeedback(
            channel=FeedbackChannel.LLM,
            passed=True,
            detail="no hypothesis/description (mock: pass)",
            score=1.0,
        )

    # 提取 hypothesis 关键词
    hyp_lower = hypothesis.lower()
    expr_lower = formula.lower()
    # 至少 formula 中应包含 hypothesis 中提到的关键算子或变量
    keywords = [
        w for w in hyp_lower.split()
        if len(w) > 3 and w.isalpha()
    ]
    matches = sum(1 for kw in keywords if kw in expr_lower)
    match_ratio = matches / len(keywords) if keywords else 1.0

    passed = match_ratio >= 0.3
    detail = (
        f"keyword match: {matches}/{len(keywords)} ({match_ratio:.0%})"
    )
    score = match_ratio
    return ChannelFeedback(
        channel=FeedbackChannel.LLM,
        passed=passed,
        detail=detail,
        score=score,
        metadata={"match_ratio": match_ratio, "matches": matches},
    )


# ==============================================================================
# 聚合器
# ==============================================================================


def collect_all_channels(
    formula: str,
    result: Optional[pl.Series],
    expected_length: int,
    config: MCTSFeedbackConfig,
    exception: Optional[Exception] = None,
    hypothesis: Optional[str] = None,
    description: Optional[str] = None,
) -> FactorFeedback:
    """一次性采集 5 通道反馈，构造 FactorFeedback

    Args:
        formula: 因子公式
        result: 评估结果（pl.Series 或 None）
        expected_length: 预期长度（数据行数）
        config: 5 通道配置
        exception: 评估异常（None=成功）
        hypothesis: 研究假设（用于 LLM 通道）
        description: 因子描述（用于 LLM 通道）

    Returns:
        FactorFeedback（含 5 通道 + decision + summary）
    """
    channels: Dict[FeedbackChannel, ChannelFeedback] = {}

    if config.enable_execution:
        channels[FeedbackChannel.EXECUTION] = collect_execution_channel(formula, exception)
    if config.enable_shape and exception is None:
        channels[FeedbackChannel.SHAPE] = collect_shape_channel(result, expected_length)
    if config.enable_code:
        channels[FeedbackChannel.CODE] = collect_code_channel(formula, config)
    if config.enable_value and exception is None and result is not None:
        channels[FeedbackChannel.VALUE] = collect_value_channel(result, config)
    if config.enable_llm:
        channels[FeedbackChannel.LLM] = collect_llm_channel(
            formula, hypothesis, description,
        )

    # decision: 所有启用的通道都通过
    enabled_channels = list(channels.values())
    decision = all(ch.passed for ch in enabled_channels) if enabled_channels else True

    # 综合评分（5 通道平均）
    if enabled_channels:
        score = sum(ch.score for ch in enabled_channels) / len(enabled_channels)
    else:
        score = 0.0

    # summary
    failed = [ch.channel.value for ch in enabled_channels if not ch.passed]
    summary = (
        "OK" if decision else f"FAIL: {','.join(failed)}"
    )

    return FactorFeedback(
        factor_name=formula[:100],
        channels=channels,
        decision=decision,
        summary=summary,
        metadata={
            "score": score,
            "enabled_channels": [ch.value for ch in channels.keys()],
        },
    )


# ==============================================================================
# 辅助函数
# ==============================================================================


def _count_base_features(formula: str) -> int:
    """统计表达式中唯一的基础特征名数量

    Args:
        formula: 公式字符串
    """
    try:
        tree = ast.parse(formula)
    except SyntaxError:
        return 0
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return len(names & BASE_FEATURE_NAMES)


def _calc_free_args_ratio(formula: str) -> float:
    """计算自由参数（非基础特征）占比

    注意：只计算数据特征（如 close, open, high, low, vol, amount, returns），
    不计算函数名（如 rank, ts_mean, ts_std）。
    """
    import re

    # 已知函数名（不算作自由参数）
    known_ops = {
        "rank", "zscore", "winsorize", "ts_mean", "ts_std", "ts_sum",
        "ts_max", "ts_min", "ts_median", "ts_rank", "ts_zscore",
        "ts_skew", "ts_kurt", "ts_delta", "ts_corr", "ts_cov",
        "ts_decay_linear", "abs", "log", "sqrt", "sign", "signedpower",
        "delta", "delay", "sub", "add", "mul", "div", "greater", "less",
        "IndNeutralize", "returns", "scale",
    }

    # 提取所有标识符
    names = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', formula)

    total_names = 0
    free_args = 0
    for name in names:
        # 跳过已知函数名
        if name in known_ops:
            continue
        # 跳过数字
        if name.isdigit():
            continue
        total_names += 1
        if name not in BASE_FEATURE_NAMES:
            free_args += 1

    if total_names == 0:
        return 0.0
    return free_args / total_names
