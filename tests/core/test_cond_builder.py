# -*- coding: utf-8 -*-
"""QuantNodes.core.cond_builder 单元测试"""

from QuantNodes.core.cond_builder import Cond, _CondBuilder


class TestCondBuilder:
    def test_cond_call(self):
        result = Cond('close')
        assert result is not None

    def test_cond_attr(self):
        result = Cond.attr('metrics')
        assert result is not None

    def test_cond_constant(self):
        result = Cond.constant(42)
        assert result is not None

    def test_cond_getattr(self):
        result = Cond.sharpe
        assert result is not None

    def test_cond_getitem(self):
        result = Cond['close']
        assert result is not None

    def test_cond_input(self):
        result = Cond.input
        assert result is not None

    def test_cond_is_instance(self):
        assert isinstance(Cond, _CondBuilder)


class TestCondBuilderBuildsCorrectExpression:
    def test_cond_call_builds_variable_expr(self):
        from QuantNodes.core.expression import VariableExpr
        result = Cond('close')
        assert isinstance(result._expr, VariableExpr)
        assert result._expr.name == 'close'

    def test_cond_attr_builds_attribute_expr(self):
        from QuantNodes.core.expression import AttributeExpr, InputExpr
        result = Cond.attr('metrics')
        assert isinstance(result._expr, AttributeExpr)
        assert isinstance(result._expr.expr, InputExpr)
        assert result._expr.attr == 'metrics'

    def test_cond_constant_builds_constant_expr(self):
        from QuantNodes.core.expression import ConstantExpr
        result = Cond.constant(3.14)
        assert isinstance(result._expr, ConstantExpr)
        assert result._expr.value == 3.14

    def test_cond_getattr_builds_attribute_expr(self):
        from QuantNodes.core.expression import AttributeExpr
        result = Cond.sharpe
        assert isinstance(result._expr, AttributeExpr)
        assert result._expr.attr == 'sharpe'

    def test_cond_getitem_builds_variable_expr(self):
        from QuantNodes.core.expression import VariableExpr
        result = Cond['volume']
        assert isinstance(result._expr, VariableExpr)
        assert result._expr.name == 'volume'


class TestCondBuilderChaining:
    def test_cond_supports_comparison(self):
        result = Cond('close') > 50
        assert result is not None

    def test_cond_supports_arithmetic(self):
        result = Cond('close') + Cond('volume')
        assert result is not None

    def test_cond_supports_method_chaining(self):
        result = Cond.attr('metrics').sharpe
        assert result is not None
