"""ComplexityChecker — AST 静态检查, 防过拟合。"""
from __future__ import annotations

import ast

from ..constants import BASE_FEATURE_NAMES as _BASE_FEATURE_NAMES
from ..feedback import ChannelFeedback, FeedbackChannel
from .settings import ComplexitySetting


class ComplexityChecker:
    """复杂度门: AST 静态检查 (length / base features / free args ratio)。"""

    def __init__(self, settings: ComplexitySetting | None = None):
        self.settings = settings or ComplexitySetting()

    def check(self, expression: str) -> ChannelFeedback:
        """返回 ChannelFeedback (CODE 通道)。"""
        if not self.settings.enabled:
            return ChannelFeedback(
                channel=FeedbackChannel.CODE,
                passed=True,
                detail="complexity disabled",
            )

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
        free_args_ratio = _calc_free_args_ratio(tree)

        violations: list[str] = []
        if symbol_length > self.settings.symbol_length_threshold:
            violations.append(
                f"length={symbol_length}>{self.settings.symbol_length_threshold}"
            )
        if base_features > self.settings.base_features_threshold:
            violations.append(
                f"features={base_features}>{self.settings.base_features_threshold}"
            )
        if free_args_ratio > self.settings.free_args_ratio_threshold:
            violations.append(
                f"free_args={free_args_ratio:.2f}>{self.settings.free_args_ratio_threshold}"
            )

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


def _count_base_features(tree: ast.AST) -> int:
    """统计表达式中唯一的基础特征名数量。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return len(names & _BASE_FEATURE_NAMES)


def _calc_free_args_ratio(tree: ast.AST) -> float:
    """计算自由参数 (非基础特征) 占比。"""
    total = 0
    free = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            total += 1
            if node.id not in _BASE_FEATURE_NAMES:
                free += 1
    return (free / total) if total > 0 else 0.0
