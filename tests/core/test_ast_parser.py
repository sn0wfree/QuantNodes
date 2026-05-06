# coding=utf-8
"""QuantNodes.core.ast_parser 单元测试"""
import pytest

from QuantNodes.core.ast_parser import parse_expression
from QuantNodes.core.expression import (
    ConstantExpr,
    VariableExpr,
    InputExpr,
    AttributeExpr,
    SubscriptExpr,
    MethodCallExpr,
    BinaryOpExpr,
    UnaryOpExpr,
    ComparisonExpr,
    LogicalOpExpr,
)


class TestParseExpressionConstants:
    def test_integer(self):
        expr = parse_expression("42")
        assert isinstance(expr, ConstantExpr)
        assert expr.value == 42

    def test_float(self):
        expr = parse_expression("3.14")
        assert isinstance(expr, ConstantExpr)
        assert expr.value == 3.14

    def test_string(self):
        expr = parse_expression("'hello'")
        assert isinstance(expr, ConstantExpr)
        assert expr.value == "hello"

    def test_negative_number(self):
        expr = parse_expression("-5")
        assert isinstance(expr, UnaryOpExpr)

    def test_boolean_true(self):
        expr = parse_expression("True")
        assert isinstance(expr, ConstantExpr)
        assert expr.value is True

    def test_boolean_false(self):
        expr = parse_expression("False")
        assert isinstance(expr, ConstantExpr)
        assert expr.value is False


class TestParseExpressionVariables:
    def test_simple_name(self):
        expr = parse_expression("price")
        assert isinstance(expr, VariableExpr)
        assert expr.name == "price"

    def test_input_df(self):
        expr = parse_expression("df")
        assert isinstance(expr, InputExpr)

    def test_input_data(self):
        expr = parse_expression("data")
        assert isinstance(expr, InputExpr)

    def test_input_x(self):
        expr = parse_expression("x")
        assert isinstance(expr, InputExpr)

    def test_input_result(self):
        expr = parse_expression("result")
        assert isinstance(expr, InputExpr)

    def test_input_input(self):
        expr = parse_expression("input")
        assert isinstance(expr, InputExpr)


class TestParseExpressionAttributes:
    def test_attribute_access(self):
        expr = parse_expression("obj.attr")
        assert isinstance(expr, AttributeExpr)
        assert expr.attr == "attr"

    def test_nested_attribute(self):
        expr = parse_expression("a.b.c")
        assert isinstance(expr, AttributeExpr)
        assert expr.attr == "c"


class TestParseExpressionBinaryOps:
    def test_add(self):
        expr = parse_expression("a + b")
        assert isinstance(expr, BinaryOpExpr)
        assert expr.op == "+"

    def test_sub(self):
        expr = parse_expression("a - b")
        assert isinstance(expr, BinaryOpExpr)
        assert expr.op == "-"

    def test_mul(self):
        expr = parse_expression("a * b")
        assert isinstance(expr, BinaryOpExpr)
        assert expr.op == "*"

    def test_div(self):
        expr = parse_expression("a / b")
        assert isinstance(expr, BinaryOpExpr)
        assert expr.op == "/"

    def test_floor_div(self):
        expr = parse_expression("a // b")
        assert isinstance(expr, BinaryOpExpr)
        assert expr.op == "//"

    def test_mod(self):
        expr = parse_expression("a % b")
        assert isinstance(expr, BinaryOpExpr)
        assert expr.op == "%"

    def test_pow(self):
        expr = parse_expression("a ** b")
        assert isinstance(expr, BinaryOpExpr)
        assert expr.op == "**"


class TestParseExpressionUnaryOps:
    def test_negate(self):
        expr = parse_expression("-x")
        assert isinstance(expr, UnaryOpExpr)
        assert expr.op == "-"


class TestParseExpressionComparisons:
    def test_gt(self):
        expr = parse_expression("a > b")
        assert isinstance(expr, ComparisonExpr)
        assert expr.op == ">"

    def test_gte(self):
        expr = parse_expression("a >= b")
        assert isinstance(expr, ComparisonExpr)
        assert expr.op == ">="

    def test_lt(self):
        expr = parse_expression("a < b")
        assert isinstance(expr, ComparisonExpr)
        assert expr.op == "<"

    def test_lte(self):
        expr = parse_expression("a <= b")
        assert isinstance(expr, ComparisonExpr)
        assert expr.op == "<="

    def test_eq(self):
        expr = parse_expression("a == b")
        assert isinstance(expr, ComparisonExpr)
        assert expr.op == "=="

    def test_neq(self):
        expr = parse_expression("a != b")
        assert isinstance(expr, ComparisonExpr)
        assert expr.op == "!="


class TestParseExpressionLogicalOps:
    def test_and(self):
        expr = parse_expression("a and b")
        assert isinstance(expr, LogicalOpExpr)
        assert expr.op == "and"

    def test_or(self):
        expr = parse_expression("a or b")
        assert isinstance(expr, LogicalOpExpr)
        assert expr.op == "or"


class TestParseExpressionSubscript:
    def test_subscript(self):
        expr = parse_expression("df['close']")
        assert isinstance(expr, SubscriptExpr)

    def test_subscript_number(self):
        expr = parse_expression("df[0]")
        assert isinstance(expr, SubscriptExpr)


class TestParseExpressionMethodCall:
    def test_method_call(self):
        expr = parse_expression("x.upper()")
        assert isinstance(expr, MethodCallExpr)
        assert expr.method == "upper"

    def test_method_with_args(self):
        expr = parse_expression("x.strip('a')")
        assert isinstance(expr, MethodCallExpr)


class TestParseExpressionSecurity:
    def test_forbidden_eval(self):
        with pytest.raises(ValueError, match="Forbidden method"):
            parse_expression("x.eval('1+1')")

    def test_forbidden_exec(self):
        with pytest.raises(ValueError, match="Forbidden method"):
            parse_expression("x.exec('pass')")

    def test_forbidden_import(self):
        with pytest.raises(ValueError, match="Forbidden method"):
            parse_expression("x.__import__('os')")

    def test_forbidden_open(self):
        with pytest.raises(ValueError, match="Forbidden method"):
            parse_expression("x.open('file')")

    def test_forbidden_compile(self):
        with pytest.raises(ValueError, match="Forbidden method"):
            parse_expression("x.compile('code')")


class TestParseExpressionSyntaxErrors:
    def test_empty_string(self):
        with pytest.raises(SyntaxError):
            parse_expression("")

    def test_invalid_syntax(self):
        with pytest.raises(SyntaxError):
            parse_expression("if")

    def test_double_operator(self):
        with pytest.raises(SyntaxError):
            parse_expression("a @")


class TestParseExpressionComplex:
    def test_complex_expression(self):
        expr = parse_expression("df['close'] > 50 and df['volume'] > 1000")
        assert isinstance(expr, LogicalOpExpr)

    def test_nested_binary(self):
        expr = parse_expression("(a + b) * c")
        assert isinstance(expr, BinaryOpExpr)
        assert expr.op == "*"
