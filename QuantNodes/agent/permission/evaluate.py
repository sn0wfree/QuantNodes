# coding=utf-8
"""
权限规则评估引擎
"""

from .models import PermissionRule, Action, Ruleset


def evaluate(
    permission: str,
    target: str,
    *rulesets: Ruleset,
) -> PermissionRule:
    """评估权限规则

    规则评估逻辑：
    1. 将所有 ruleset 扁平化
    2. 从后向前查找第一个匹配的规则（后定义的规则优先）
    3. 如果没有匹配规则，默认返回 ask（安全默认）

    Args:
        permission: 权限类别（如 "bash", "edit"）
        target: 目标模式（如 "git commit", "/path/to/file.py"）
        *rulesets: 规则集（按优先级从低到高排列）

    Returns:
        匹配的规则（包含 action）
    """
    all_rules = []
    for rs in rulesets:
        all_rules.extend(rs)

    for rule in reversed(all_rules):
        if rule.matches(permission, target):
            return rule

    return PermissionRule(
        permission=permission,
        pattern="*",
        action=Action.ASK,
    )
