# coding=utf-8
"""
符号计算引擎 - AST 优化器

对 SQLExpression AST 进行优化，如常量折叠、死代码消除等。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Set

if TYPE_CHECKING:
    from QuantNodes.symbolic.expression import SQLExpression


class SQLOptimizer:
    """
    SQL AST 优化器

    对表达式进行各种优化：
    - 常量折叠
    - 冗余括号消除
    - 恒等变换
    """

    def __init__(self):
        self._visited: Set[int] = set()

    def optimize(self, expr: "SQLExpression") -> "SQLExpression":
        """
        优化表达式

        Args:
            expr: 输入表达式

        Returns:
            优化后的表达式
        """
        from QuantNodes.symbolic.expression import (
            ColumnRef, LiteralValue, SQLBinaryOp, SQLUnaryOp,
            SQLComparison, SQLLogicalOp, SQLFunction, SQLCase,
        )

        if id(expr) in self._visited:
            return expr

        self._visited.add(id(expr))

        if isinstance(expr, ColumnRef):
            return expr
        if isinstance(expr, LiteralValue):
            return expr
        if isinstance(expr, SQLBinaryOp):
            return self._optimize_binary_op(expr)
        if isinstance(expr, SQLUnaryOp):
            return self._optimize_unary_op(expr)
        if isinstance(expr, SQLComparison):
            return self._optimize_comparison(expr)
        if isinstance(expr, SQLLogicalOp):
            return self._optimize_logical_op(expr)
        if isinstance(expr, SQLFunction):
            return self._optimize_function(expr)
        if isinstance(expr, SQLCase):
            return self._optimize_case(expr)

        return expr

    def _optimize_binary_op(self, expr: "SQLBinaryOp") -> "SQLExpression":  # noqa: F821
        """优化二元运算"""
        left = self.optimize(expr.left)
        right = self.optimize(expr.right)

        from QuantNodes.symbolic.expression import LiteralValue, SQLBinaryOp

        if isinstance(left, LiteralValue) and isinstance(right, LiteralValue):
            return self._fold_binary_op(expr.op, left.value, right.value)

        if expr.op == "+" and isinstance(right, LiteralValue) and right.value == 0:
            return left
        if expr.op == "-" and isinstance(right, LiteralValue) and right.value == 0:
            return left
        if expr.op == "*" and isinstance(right, LiteralValue) and right.value == 1:
            return left
        if expr.op == "*" and isinstance(left, LiteralValue) and left.value == 1:
            return right
        if expr.op == "/" and isinstance(right, LiteralValue) and right.value == 1:
            return left

        return SQLBinaryOp(left, expr.op, right)

    def _optimize_unary_op(self, expr: "SQLUnaryOp") -> "SQLExpression":  # noqa: F821
        """优化一元运算"""
        from QuantNodes.symbolic.expression import LiteralValue, SQLUnaryOp
        operand = self.optimize(expr.operand)

        if isinstance(operand, LiteralValue):
            return self._fold_unary_op(expr.op, operand.value)

        return SQLUnaryOp(expr.op, operand)

    def _optimize_comparison(self, expr: "SQLComparison") -> "SQLExpression":  # noqa: F821
        """优化比较运算"""
        left = self.optimize(expr.left)  # noqa: F841 (placeholder for future folding)
        right = self.optimize(expr.right)  # noqa: F841
        return expr

    def _optimize_logical_op(self, expr: "SQLLogicalOp") -> "SQLExpression":  # noqa: F821
        """优化逻辑运算"""
        from QuantNodes.symbolic.expression import LiteralValue, SQLLogicalOp
        operands = [self.optimize(op) for op in expr.operands]

        if expr.op == "AND":
            for op in operands:
                if isinstance(op, LiteralValue) and not op.value:
                    return LiteralValue(False)
        if expr.op == "OR":
            for op in operands:
                if isinstance(op, LiteralValue) and op.value:
                    return LiteralValue(True)

        return SQLLogicalOp(expr.op, *operands)

    def _optimize_function(self, expr: "SQLFunction") -> "SQLExpression":  # noqa: F821
        """优化函数调用"""
        args = [self.optimize(arg) for arg in expr.args]  # noqa: F841 (placeholder)
        kwargs = {k: self.optimize(v) for k, v in expr.kwargs.items()}  # noqa: F841
        return expr

    def _optimize_case(self, expr: "SQLCase") -> "SQLExpression":  # noqa: F821
        """优化 CASE 表达式"""
        return expr

    def _fold_binary_op(self, op: str, a: Any, b: Any) -> "SQLExpression":
        """常量折叠 - 二元运算"""
        from QuantNodes.symbolic.expression import LiteralValue
        try:
            if op == "+":
                return LiteralValue(a + b)
            if op == "-":
                return LiteralValue(a - b)
            if op == "*":
                return LiteralValue(a * b)
            if op == "/":
                return LiteralValue(a / b)
        except Exception:
            pass
        return None

    def _fold_unary_op(self, op: str, a: Any) -> "SQLExpression":
        """常量折叠 - 一元运算"""
        from QuantNodes.symbolic.expression import LiteralValue
        try:
            if op == "-":
                return LiteralValue(-a)
            if op == "+":
                return LiteralValue(+a)
        except Exception:
            pass
        return None


def optimize_expression(expr: "SQLExpression") -> "SQLExpression":
    """便捷函数：优化表达式"""
    optimizer = SQLOptimizer()
    return optimizer.optimize(expr)
