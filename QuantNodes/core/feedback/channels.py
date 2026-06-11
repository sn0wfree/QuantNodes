"""4 通道反馈采集器: execution / shape / code / value。

每个函数返回 ChannelFeedback, 可被 FeedbackCollector.add() 直接消费。
"""
from __future__ import annotations

import ast
from typing import Iterable

import numpy as np
import pandas as pd

from .dataclass import ChannelFeedback, FeedbackChannel


_BASE_FEATURE_NAMES = frozenset({
    "open", "high", "low", "close", "volume", "amount",
    "vwap", "turnover", "mv_float", "total_mv", "circ_mv",
    "returns", "vwap_adj",
})


def collect_execution(stdout: str, stderr: str, exit_code: int) -> ChannelFeedback:
    """EXECUTION 通道: 沙箱执行结果。"""
    passed = exit_code == 0
    detail = f"exit={exit_code}\nstdout: {str(stdout)[:500]}\nstderr: {str(stderr)[:500]}"
    score = 1.0 if passed else 0.0
    return ChannelFeedback(
        channel=FeedbackChannel.EXECUTION,
        passed=passed,
        detail=detail,
        score=score,
        metadata={"exit_code": int(exit_code)},
    )


def collect_shape(actual_shape: tuple, expected_shape: tuple) -> ChannelFeedback:
    """SHAPE 通道: 形状一致性。"""
    passed = tuple(actual_shape) == tuple(expected_shape)
    detail = f"actual={tuple(actual_shape)}, expected={tuple(expected_shape)}"
    score = 1.0 if passed else 0.0
    return ChannelFeedback(
        channel=FeedbackChannel.SHAPE,
        passed=passed,
        detail=detail,
        score=score,
    )


def collect_code(
    expression: str,
    symbol_length_threshold: int = 200,
    base_features_threshold: int = 5,
    free_args_ratio_threshold: float = 0.5,
) -> ChannelFeedback:
    """CODE 通道: AST 静态检查 (防过拟合)。

    检查项:
        - 表达式长度 <= symbol_length_threshold
        - 基础特征数 <= base_features_threshold
        - 自由参数 (Name) 占比 <= free_args_ratio_threshold
    """
    try:
        tree = ast.parse(expression)
    except SyntaxError as e:
        return ChannelFeedback(
            channel=FeedbackChannel.CODE,
            passed=False,
            detail=f"语法错误: {e}",
            score=0.0,
        )

    symbol_length = len(expression)
    base_features = _count_base_features(tree)
    free_args_ratio = _calc_free_args_ratio(tree, base_features)

    violations: list[str] = []
    if symbol_length > symbol_length_threshold:
        violations.append(f"length={symbol_length}>{symbol_length_threshold}")
    if base_features > base_features_threshold:
        violations.append(f"features={base_features}>{base_features_threshold}")
    if free_args_ratio > free_args_ratio_threshold:
        violations.append(f"free_args={free_args_ratio:.2f}>{free_args_ratio_threshold}")

    passed = len(violations) == 0
    detail = "; ".join(violations) if violations else \
        f"OK (length={symbol_length}, features={base_features}, free_args={free_args_ratio:.2f})"
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


def collect_value(
    values: pd.Series,
    nan_threshold: float = 0.3,
    std_threshold: float = 1e-6,
) -> ChannelFeedback:
    """VALUE 通道: 数值分布合理性。

    检查项:
        - NaN 比例 <= nan_threshold (默认 30%)
        - Inf 数量 == 0
        - 标准差 > std_threshold
    """
    s = pd.Series(values).dropna()
    if len(s) == 0:
        return ChannelFeedback(
            channel=FeedbackChannel.VALUE,
            passed=False,
            detail="全部 NaN, 无有效数值",
            score=0.0,
        )

    nan_pct = float(pd.Series(values).isna().mean())
    inf_count = int(np.isinf(pd.Series(values).fillna(0)).sum())
    mean_val = float(s.mean())
    std_val = float(s.std()) if len(s) > 1 else 0.0

    violations: list[str] = []
    if nan_pct > nan_threshold:
        violations.append(f"NaN={nan_pct:.2%}>{nan_threshold:.0%}")
    if inf_count > 0:
        violations.append(f"Inf={inf_count}>0")
    if std_val <= std_threshold:
        violations.append(f"std={std_val:.6f}<={std_threshold}")

    passed = len(violations) == 0
    detail = "; ".join(violations) if violations else \
        f"OK (NaN={nan_pct:.2%}, mean={mean_val:.4f}, std={std_val:.4f})"
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


def _count_base_features(tree: ast.AST) -> int:
    """统计表达式中唯一的基础特征名数量。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return len(names & _BASE_FEATURE_NAMES)


def _calc_free_args_ratio(tree: ast.AST, base_features: int) -> float:
    """计算自由参数 (非基础特征) 占比。"""
    total_names = 0
    free_args = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            total_names += 1
            if node.id not in _BASE_FEATURE_NAMES:
                free_args += 1
    if total_names == 0:
        return 0.0
    return free_args / total_names
