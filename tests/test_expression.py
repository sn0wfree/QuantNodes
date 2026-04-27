# coding=utf-8
"""
表达式系统单元测试
"""

import pytest
from QuantNodes.core import (
    Cond,
    Expression,
    IfNode,
    WhileNode,
    BaseNode,
)


class MultiplyNode(BaseNode):
    """乘法节点"""
    def __init__(self, factor: int = 2, name=None):
        super().__init__(name=name or "Multiply")
        self.factor = factor

    def _execute(self, input_data, **kwargs):
        return input_data * self.factor


class TestDSLBuilder:
    """测试 Cond DSL 构建器"""

    def test_cond_call(self):
        """测试 Cond('name') 语法"""
        expr = Cond('value') > 50
        assert expr({'value': 60}) is True
        assert expr({'value': 40}) is False

    def test_cond_getitem(self):
        """测试 Cond['name'] 语法"""
        expr = Cond['value'] > 50
        assert expr({'value': 60}) is True

    def test_cond_attr(self):
        """测试 Cond.attr 语法"""
        class Obj:
            def __init__(self):
                self.metrics = type('Metrics', (), {'sharpe': 2.0})()

        expr = Cond.attr('metrics').sharpe >= 1.5
        assert expr(Obj()) is True

    def test_cond_nested_attr(self):
        """测试嵌套属性访问"""
        expr = Cond.attr('a').attr('b').c == 3
        obj = type('Root', (), {'a': type('A', (), {'b': type('B', (), {'c': 3})()})()})()
        assert expr(obj) is True

    def test_cond_property(self):
        """测试 Cond.name 属性语法"""
        expr = Cond.age > 18
        obj = type('Person', (), {'age': 20})()
        assert expr(obj) is True

    def test_cond_constant(self):
        """测试常量表达式"""
        expr = Cond.constant(True)
        assert expr(None) is True


class TestLogicalOperations:
    """测试逻辑运算"""

    def test_and(self):
        """测试 AND 运算"""
        expr = (Cond('a') > 0) & (Cond('b') < 10)
        assert expr({'a': 5, 'b': 5}) is True
        assert expr({'a': 5, 'b': 15}) is False

    def test_or(self):
        """测试 OR 运算"""
        expr = (Cond('a') > 10) | (Cond('b') > 10)
        assert expr({'a': 5, 'b': 15}) is True
        assert expr({'a': 5, 'b': 5}) is False

    def test_not(self):
        """测试 NOT 运算"""
        expr = ~(Cond('a') > 5)
        assert expr({'a': 3}) is True
        assert expr({'a': 10}) is False

    def test_complex_logic(self):
        """测试复杂逻辑组合"""
        expr = ((Cond('a') > 0) & (Cond('b') > 0)) | (Cond('c') > 100)
        assert expr({'a': 1, 'b': 1, 'c': 0}) is True
        assert expr({'a': -1, 'b': -1, 'c': 200}) is True
        assert expr({'a': -1, 'b': -1, 'c': 0}) is False


class TestArithmeticOperations:
    """测试算术运算"""

    def test_add(self):
        """测试加法"""
        expr = Cond('a') + Cond('b')
        assert expr({'a': 2, 'b': 3}) == 5

    def test_sub(self):
        """测试减法"""
        expr = Cond('a') - 1
        assert expr({'a': 10}) == 9

    def test_mul(self):
        """测试乘法"""
        expr = Cond('a') * 2
        assert expr({'a': 5}) == 10

    def test_div(self):
        """测试除法"""
        expr = Cond('a') / 2
        assert expr({'a': 10}) == 5.0

    def test_neg(self):
        """测试负号"""
        expr = -Cond('a')
        assert expr({'a': 5}) == -5


class TestExpressionParse:
    """测试字符串表达式解析"""

    def test_parse_simple_compare(self):
        """测试解析简单比较表达式"""
        expr = Expression.parse("df['value'] > 50")
        assert expr({'value': 60}) is True

    def test_parse_attribute(self):
        """测试解析属性访问表达式"""
        expr = Expression.parse("result.status == 'ok'")
        obj = type('Result', (), {'status': 'ok'})()
        assert expr(obj) is True

    def test_parse_arithmetic(self):
        """测试解析算术运算表达式"""
        expr = Expression.parse("(a + b) * 2")
        assert expr({'a': 2, 'b': 3}) == 10


class TestIfNodeWithExpression:
    """测试 IfNode 集成表达式"""

    def test_if_with_dsl_condition(self):
        """测试使用 DSL 构建条件"""
        node = IfNode(
            condition=Cond.input > 10,
            true_branch=MultiplyNode(2),
            false_branch=MultiplyNode(3),
        )
        assert node.execute(20) == 40
        assert node.execute(5) == 15

    def test_if_with_string_condition(self):
        """测试使用字符串表达式条件"""
        node = IfNode(
            condition="x > 10",
            true_branch=MultiplyNode(2),
            false_branch=MultiplyNode(3),
        )
        assert node.execute(20) == 40
        assert node.execute(5) == 15

    def test_if_to_dict(self):
        """测试 to_dict 包含可读的条件"""
        node = IfNode(
            condition=Cond('value') > 10,
            true_branch=MultiplyNode(2),
        )
        d = node.to_info()
        assert 'condition' in d
        assert 'condition_dict' in d
        assert '(Cond' in d['condition'] or 'VariableExpr' in d['condition_dict']['type']


class TestWhileNodeWithExpression:
    """测试 WhileNode 集成表达式"""

    def test_while_with_dsl_condition(self):
        """测试使用 DSL 构建循环条件"""
        loop = WhileNode(
            condition=Cond.input < 10,
            body=MultiplyNode(2),
            max_iterations=10,
        )
        result = loop.execute(1)
        # 1 -> 2 -> 4 -> 8 -> 16 (停止)
        assert result == 16
        assert loop.iteration_count == 4

    def test_while_with_string_condition(self):
        """测试使用字符串表达式作为循环条件"""
        loop = WhileNode(
            condition="x < 10",
            body=MultiplyNode(2),
            max_iterations=10,
        )
        result = loop.execute(1)
        assert result == 16


class TestSerialization:
    """测试表达式序列化"""

    def test_serialize_basic(self):
        """测试基础表达式序列化"""
        expr = Cond('a') > 5
        d = expr.serialize()
        assert d['type'] == 'ComparisonExpr'
        assert d['op'] == '>'

    def test_constant_expr_serialize(self):
        """测试常量表达式序列化"""
        from QuantNodes.core.expression import ConstantExpr
        expr = ConstantExpr(42)
        d = expr.serialize()
        assert d['value'] == 42
        restored = ConstantExpr.deserialize(d)
        assert restored.evaluate(None) == 42


class TestBackwardCompatibility:
    """测试向后兼容 - lambda 仍然可用"""

    def test_if_with_lambda(self):
        """测试 IfNode 仍然支持 lambda 条件"""
        node = IfNode(
            condition=lambda x: x > 10,
            true_branch=MultiplyNode(2),
            false_branch=MultiplyNode(3),
        )
        assert node.execute(20) == 40
        assert node.execute(5) == 15

    def test_while_with_lambda(self):
        """测试 WhileNode 仍然支持 lambda 条件"""
        loop = WhileNode(
            condition=lambda x: x < 10,
            body=MultiplyNode(2),
            max_iterations=10,
        )
        result = loop.execute(1)
        assert result == 16
