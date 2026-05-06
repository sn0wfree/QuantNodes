# coding=utf-8
"""
operators.proxy - 统一代理层

本模块从 factor_functions 导出注册表 API，
作为 operators 模块的统一代理入口。

Agent 系统应从本模块导入，而非直接从 factor_functions 导入。
"""

from QuantNodes.factor_node.factor_functions import (
    list_operators,
    get_operator,
    register_operator,
    operator_info,
    generate_documentation,
    OperatorCategory,
    _OPERATOR_REGISTRY,
)

__all__ = [
    "list_operators",
    "get_operator",
    "register_operator",
    "operator_info",
    "generate_documentation",
    "OperatorCategory",
    "_OPERATOR_REGISTRY",
]
