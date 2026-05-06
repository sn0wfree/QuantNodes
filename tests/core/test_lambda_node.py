# coding=utf-8
"""QuantNodes.core.lambda_node 单元测试"""
import pytest

from QuantNodes.core.lambda_node import LambdaNode
from QuantNodes.core.node import SerializationError


def _add_one(input_data, node):
    return input_data + 1 if input_data is not None else 1


def _multiply(input_data, node):
    config = node.config or {}
    factor = config.get("factor", 2)
    return input_data * factor if input_data is not None else 0


def _named_func(x, ctx):
    return x


class TestLambdaNodeExecution:
    def test_execute(self):
        node = LambdaNode(_add_one)
        assert node.execute(5) == 6

    def test_execute_none(self):
        node = LambdaNode(_add_one)
        assert node.execute(None) == 1

    def test_execute_with_config(self):
        node = LambdaNode(_multiply, config={"factor": 10})
        assert node.execute(5) == 50

    def test_execute_returns_node(self):
        node = LambdaNode(_add_one)
        result = node(3)
        assert result == 4


class TestLambdaNodeName:
    def test_default_name_from_func(self):
        node = LambdaNode(_add_one)
        assert node.name == "_add_one"

    def test_default_name_lambda(self):
        node = LambdaNode(lambda x, n: x)
        # lambda __name__ is "<lambda>" which is truthy
        assert node.name == "<lambda>"

    def test_custom_name(self):
        node = LambdaNode(_add_one, name="MyNode")
        assert node.name == "MyNode"


class TestLambdaNodeSerialization:
    def test_get_serializable_fields_named(self):
        node = LambdaNode(_named_func, name="TestNode", config={"k": "v"})
        fields = node._get_serializable_fields()
        assert fields["func"]["type"] == "named_function"
        assert fields["func"]["module"] == _named_func.__module__
        assert fields["func"]["qualname"] == "_named_func"

    def test_get_serializable_fields_lambda_raises(self):
        node = LambdaNode(lambda x, n: x)
        with pytest.raises(SerializationError):
            node._get_serializable_fields()

    def test_from_dict_impl(self):
        data = {
            "name": "TestNode",
            "config": {"key": "val"},
            "func": {
                "type": "named_function",
                "module": _named_func.__module__,
                "qualname": "_named_func",
            },
        }
        node = LambdaNode._from_dict_impl(data)
        assert isinstance(node, LambdaNode)
        assert node.name == "TestNode"
        assert node.config == {"key": "val"}

    def test_roundtrip(self):
        node = LambdaNode(_named_func, name="RT", config={"a": 1})
        fields = node._get_serializable_fields()
        restored = LambdaNode._from_dict_impl({
            "name": node.name,
            "config": node.config,
            **fields,
        })
        assert restored.name == "RT"
        assert restored.config == {"a": 1}
