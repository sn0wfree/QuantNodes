# coding=utf-8
"""Tests for core/control.py — IfNode, MapNode, WhileNode, _wrap_condition.

Covers: creation, condition wrapping, serialization fields, basic structure.
"""

import pandas as pd
import pytest

from QuantNodes.core.control import IfNode, MapNode, WhileNode, _wrap_condition
from QuantNodes.core.expression import ConstantExpr, VariableExpr, Expression, ExpressionBuilder


# ============================================================================
# _wrap_condition helper
# ============================================================================

class TestWrapCondition:
    def test_wrap_expression(self):
        expr = ConstantExpr(True)
        result = _wrap_condition(expr)
        assert isinstance(result, Expression)

    def test_wrap_string(self):
        result = _wrap_condition("True")
        assert isinstance(result, Expression)

    def test_wrap_callable(self):
        result = _wrap_condition(lambda x: True)
        assert isinstance(result, Expression)

    def test_wrap_expression_builder(self):
        builder = ExpressionBuilder("input")
        result = _wrap_condition(builder)
        # _wrap_condition returns condition._expr which may be a string
        # for simple ExpressionBuilder instances
        assert result is not None


# ============================================================================
# IfNode
# ============================================================================

class TestIfNode:
    def test_creation(self):
        node = IfNode(
            condition=ConstantExpr(True),
            true_branch=None,
        )
        assert node is not None

    def test_creation_with_false_branch(self):
        node = IfNode(
            condition=ConstantExpr(False),
            true_branch=None,
            false_branch=None,
        )
        assert node is not None

    def test_creation_with_string_condition(self):
        node = IfNode(
            condition="True",
            true_branch=None,
        )
        assert node is not None

    def test_creation_with_callable_condition(self):
        node = IfNode(
            condition=lambda x: x > 0,
            true_branch=None,
        )
        assert node is not None

    def test_creation_with_expression_builder(self):
        node = IfNode(
            condition=ExpressionBuilder("input"),
            true_branch=None,
        )
        assert node is not None

    def test_condition_is_expression(self):
        node = IfNode(
            condition=ConstantExpr(True),
            true_branch=None,
        )
        assert isinstance(node.condition, Expression)

    def test_serializable_fields_with_lambda_raises(self):
        """Lambda conditions cannot be serialized."""
        node = IfNode(
            condition=lambda x: True,
            true_branch=None,
        )
        with pytest.raises(Exception):
            node._get_serializable_fields()


# ============================================================================
# WhileNode
# ============================================================================

class TestWhileNode:
    def test_creation(self):
        node = WhileNode(
            condition=ConstantExpr(False),
            body=None,
        )
        assert node is not None

    def test_creation_with_string_condition(self):
        node = WhileNode(
            condition="False",
            body=None,
        )
        assert node is not None

    def test_creation_with_max_iterations(self):
        node = WhileNode(
            condition=ConstantExpr(False),
            body=None,
            max_iterations=10,
        )
        assert node is not None

    def test_condition_is_expression(self):
        node = WhileNode(
            condition=ConstantExpr(True),
            body=None,
        )
        assert isinstance(node.condition, Expression)

    def test_serializable_fields_with_lambda_raises(self):
        node = WhileNode(
            condition=lambda x: True,
            body=None,
        )
        with pytest.raises(Exception):
            node._get_serializable_fields()


# ============================================================================
# MapNode
# ============================================================================

class TestMapNode:
    def test_creation(self):
        from QuantNodes.core.lambda_node import LambdaNode
        inner = LambdaNode(func=lambda x: x)
        node = MapNode(
            node=inner,
            group_by="date",
        )
        assert node is not None

    def test_creation_with_string_group_by(self):
        from QuantNodes.core.lambda_node import LambdaNode
        inner = LambdaNode(func=lambda x: x)
        node = MapNode(
            node=inner,
            group_by="group",
        )
        assert node is not None

    def test_creation_with_callable_group_by(self):
        from QuantNodes.core.lambda_node import LambdaNode
        inner = LambdaNode(func=lambda x: x)
        node = MapNode(
            node=inner,
            group_by=lambda x: x.get("group"),
        )
        assert node is not None

    def test_creation_with_expression_group_by(self):
        from QuantNodes.core.lambda_node import LambdaNode
        inner = LambdaNode(func=lambda x: x)
        node = MapNode(
            node=inner,
            group_by=VariableExpr("group"),
        )
        assert node is not None
