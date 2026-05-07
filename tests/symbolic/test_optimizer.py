# coding=utf-8
"""SQLOptimizer 单元测试"""
import pytest

from QuantNodes.symbolic.optimizer import SQLOptimizer, optimize_expression
from QuantNodes.symbolic.expression import (
    ColumnRef, LiteralValue, SQLBinaryOp, SQLUnaryOp,
    SQLComparison, SQLLogicalOp, SQLFunction, SQLCase,
)
from QuantNodes.symbolic.dialect import ClickHouseDialect


class TestSQLOptimizer:
    """SQLOptimizer 测试"""

    def test_optimize_column_ref(self):
        """列引用保持不变"""
        col = ColumnRef("close")
        optimizer = SQLOptimizer()
        result = optimizer.optimize(col)
        assert result is col

    def test_optimize_literal(self):
        """字面量保持不变"""
        lit = LiteralValue(100)
        optimizer = SQLOptimizer()
        result = optimizer.optimize(lit)
        assert result is lit

    def test_optimize_binary_fold_add(self):
        """常量折叠 - 加法"""
        expr = LiteralValue(1) + LiteralValue(2)
        optimizer = SQLOptimizer()
        result = optimizer.optimize(expr)
        assert isinstance(result, LiteralValue)
        assert result.value == 3

    def test_optimize_binary_fold_sub(self):
        """常量折叠 - 减法"""
        expr = LiteralValue(5) - LiteralValue(3)
        optimizer = SQLOptimizer()
        result = optimizer.optimize(expr)
        assert isinstance(result, LiteralValue)
        assert result.value == 2

    def test_optimize_binary_fold_mul(self):
        """常量折叠 - 乘法"""
        expr = LiteralValue(3) * LiteralValue(4)
        optimizer = SQLOptimizer()
        result = optimizer.optimize(expr)
        assert isinstance(result, LiteralValue)
        assert result.value == 12

    def test_optimize_binary_fold_div(self):
        """常量折叠 - 除法"""
        expr = LiteralValue(10) / LiteralValue(2)
        optimizer = SQLOptimizer()
        result = optimizer.optimize(expr)
        assert isinstance(result, LiteralValue)
        assert result.value == 5

    def test_optimize_binary_identity_add(self):
        """恒等变换 - 加 0"""
        expr = ColumnRef("close") + LiteralValue(0)
        optimizer = SQLOptimizer()
        result = optimizer.optimize(expr)
        assert isinstance(result, ColumnRef)

    def test_optimize_binary_identity_sub(self):
        """恒等变换 - 减 0"""
        expr = ColumnRef("close") - LiteralValue(0)
        optimizer = SQLOptimizer()
        result = optimizer.optimize(expr)
        assert isinstance(result, ColumnRef)

    def test_optimize_binary_identity_mul_1(self):
        """恒等变换 - 乘 1 (左)"""
        expr = LiteralValue(1) * ColumnRef("close")
        optimizer = SQLOptimizer()
        result = optimizer.optimize(expr)
        assert isinstance(result, ColumnRef)

    def test_optimize_binary_identity_mul_2(self):
        """恒等变换 - 乘 1 (右)"""
        expr = ColumnRef("close") * LiteralValue(1)
        optimizer = SQLOptimizer()
        result = optimizer.optimize(expr)
        assert isinstance(result, ColumnRef)

    def test_optimize_binary_identity_div(self):
        """恒等变换 - 除以 1"""
        expr = ColumnRef("close") / LiteralValue(1)
        optimizer = SQLOptimizer()
        result = optimizer.optimize(expr)
        assert isinstance(result, ColumnRef)

    def test_optimize_unary_neg(self):
        """一元负号折叠"""
        expr = -LiteralValue(5)
        optimizer = SQLOptimizer()
        result = optimizer.optimize(expr)
        assert isinstance(result, LiteralValue)
        assert result.value == -5

    def test_optimize_logical_and_false(self):
        """逻辑 AND - 左侧为 False"""
        expr = SQLLogicalOp("AND", LiteralValue(False), ColumnRef("close"))
        optimizer = SQLOptimizer()
        result = optimizer.optimize(expr)
        assert isinstance(result, LiteralValue)
        assert result.value is False

    def test_optimize_logical_or_true(self):
        """逻辑 OR - 左侧为 True"""
        expr = SQLLogicalOp("OR", LiteralValue(True), ColumnRef("close"))
        optimizer = SQLOptimizer()
        result = optimizer.optimize(expr)
        assert isinstance(result, LiteralValue)
        assert result.value is True

    def test_optimize_visited_set(self):
        """访问集合防止重复访问"""
        col = ColumnRef("close")
        optimizer = SQLOptimizer()
        optimizer._visited.add(id(col))
        result = optimizer.optimize(col)
        assert result is col

    def test_optimize_case(self):
        """CASE 表达式"""
        case = SQLCase([(ColumnRef("close") > LiteralValue(100), LiteralValue("high"))])
        optimizer = SQLOptimizer()
        result = optimizer.optimize(case)
        assert isinstance(result, SQLCase)

    def test_optimize_function(self):
        """函数表达式"""
        func = SQLFunction("avg", ColumnRef("close"))
        optimizer = SQLOptimizer()
        result = optimizer.optimize(func)
        assert isinstance(result, SQLFunction)


class TestOptimizeExpression:
    """optimize_expression 便捷函数测试"""

    def test_optimize_expression_basic(self):
        """基本用法"""
        expr = LiteralValue(1) + LiteralValue(2)
        result = optimize_expression(expr)
        assert isinstance(result, LiteralValue)
        assert result.value == 3

    def test_optimize_expression_column(self):
        """列引用"""
        expr = ColumnRef("close")
        result = optimize_expression(expr)
        assert result is expr
