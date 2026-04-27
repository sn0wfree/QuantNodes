# coding=utf-8
"""
AST 安全表达式解析器

基于 Python 标准库的 ast 模块，安全解析表达式字符串，
不使用 eval，避免代码注入风险。
"""

from __future__ import annotations

import ast
import logging

from QuantNodes.core.expression import (
    Expression,
    InputExpr,
    ConstantExpr,
    VariableExpr,
    AttributeExpr,
    SubscriptExpr,
    MethodCallExpr,
    BinaryOpExpr,
    UnaryOpExpr,
    ComparisonExpr,
    LogicalOpExpr,
    ALLOWED_AST_NODES,
    FORBIDDEN_METHODS,
)


logger = logging.getLogger(__name__)


def parse_expression(expr_str: str) -> Expression:
    """
    安全解析字符串表达式

    Args:
        expr_str: 表达式字符串，如 "df['close'] > 50"

    Returns:
        Expression 对象

    Raises:
        SyntaxError: 表达式语法错误
        ValueError: 包含不支持的操作
    """
    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError as e:
        raise SyntaxError(f"Invalid expression syntax: {e}") from e

    # 安全检查
    _validate_ast(tree)

    # 转换为表达式对象
    return _ast_to_expr(tree.body)


def _validate_ast(node: ast.AST) -> None:
    """递归验证 AST 节点是否在白名单中"""
    if type(node) not in ALLOWED_AST_NODES:
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")

    for child in ast.iter_child_nodes(node):
        _validate_ast(child)


def _ast_to_expr(node: ast.AST) -> Expression:
    """将 AST 节点转换为 Expression 对象"""

    # 常量
    if isinstance(node, ast.Constant):
        return ConstantExpr(node.value)

    # 变量名 - 假设是输入数据的属性或列名
    if isinstance(node, ast.Name):
        # 特殊处理常见的输入变量名
        if node.id in ("df", "input", "data", "x", "result"):
            return InputExpr()
        # 其他名称视为属性访问
        return VariableExpr(node.id)

    # 属性访问
    if isinstance(node, ast.Attribute):
        expr = _ast_to_expr(node.value)
        return AttributeExpr(expr, node.attr)

    # 下标访问
    if isinstance(node, ast.Subscript):
        expr = _ast_to_expr(node.value)
        key = _ast_to_expr(node.slice)
        return SubscriptExpr(expr, key)

    # 方法调用
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            expr = _ast_to_expr(node.func.value)
            method_name = node.func.attr
            
            if method_name in FORBIDDEN_METHODS:
                raise ValueError(f"Forbidden method: {method_name}")
            
            args = tuple(_ast_to_expr(arg) for arg in node.args)
            kwargs = {
                kw.arg: _ast_to_expr(kw.value)
                for kw in node.keywords
                if kw.arg is not None
            }
            return MethodCallExpr(expr, method_name, args, kwargs)
        raise ValueError("Unsupported call type")

    # 二元运算
    if isinstance(node, ast.BinOp):
        left = _ast_to_expr(node.left)
        right = _ast_to_expr(node.right)
        op = _bin_op_to_str(node.op)
        return BinaryOpExpr(left, op, right)

    # 一元运算
    if isinstance(node, ast.UnaryOp):
        operand = _ast_to_expr(node.operand)
        op = _unary_op_to_str(node.op)
        return UnaryOpExpr(op, operand)

    # 比较运算
    if isinstance(node, ast.Compare):
        left = _ast_to_expr(node.left)
        # 简化：只处理单个比较
        op = _cmp_op_to_str(node.ops[0])
        right = _ast_to_expr(node.comparators[0])
        return ComparisonExpr(left, op, right)

    # 逻辑运算
    if isinstance(node, ast.BoolOp):
        op = _bool_op_to_str(node.op)
        operands = tuple(_ast_to_expr(v) for v in node.values)
        return LogicalOpExpr(op, *operands)

    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


def _bin_op_to_str(op: ast.operator) -> str:
    """将二元运算 AST 节点转换为运算符字符串"""
    mapping = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.FloorDiv: "//",
        ast.Mod: "%",
        ast.Pow: "**",
    }
    result = mapping.get(type(op))
    if result is None:
        raise ValueError(f"Unsupported binary operator: {type(op).__name__}")
    return result


def _unary_op_to_str(op: ast.unaryop) -> str:
    """将一元运算 AST 节点转换为运算符字符串"""
    mapping = {
        ast.UAdd: "+",
        ast.USub: "-",
        ast.Not: "not",
    }
    result = mapping.get(type(op))
    if result is None:
        raise ValueError(f"Unsupported unary operator: {type(op).__name__}")
    return result


def _cmp_op_to_str(op: ast.cmpop) -> str:
    """将比较运算 AST 节点转换为运算符字符串"""
    mapping = {
        ast.Gt: ">",
        ast.GtE: ">=",
        ast.Lt: "<",
        ast.LtE: "<=",
        ast.Eq: "==",
        ast.NotEq: "!=",
    }
    result = mapping.get(type(op))
    if result is None:
        raise ValueError(f"Unsupported comparison operator: {type(op).__name__}")
    return result


def _bool_op_to_str(op: ast.boolop) -> str:
    """将逻辑运算 AST 节点转换为运算符字符串"""
    mapping = {
        ast.And: "and",
        ast.Or: "or",
    }
    result = mapping.get(type(op))
    if result is None:
        raise ValueError(f"Unsupported logical operator: {type(op).__name__}")
    return result
