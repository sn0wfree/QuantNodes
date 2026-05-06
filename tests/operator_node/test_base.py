# coding=utf-8
"""QuantNodes.operator_node.base 单元测试"""
from typing import Any, Dict

import pandas as pd
import pytest

from QuantNodes.core.node import BaseNode
from QuantNodes.operator_node.base import ChainOperator, OperatorNode


class DummyOperator(OperatorNode):
    def __init__(self, name: str = "Dummy", return_value: Any = None):
        super().__init__(name=name)
        self.return_value = return_value

    def _execute_operation(self, input_data: Any = None, **kwargs) -> Any:
        if self.return_value is not None:
            return self.return_value
        return f"{self.name}_result"


class TestOperatorNode:
    def test_operator_node_creation(self):
        op = DummyOperator(name="TestOp")
        assert op.name == "TestOp"

    def test_operator_node_execute(self):
        op = DummyOperator(name="TestOp", return_value="test_result")
        result = op.execute()
        assert result == "test_result"

    def test_operator_node_execute_with_input(self):
        op = DummyOperator(name="TestOp")
        result = op.execute("input_data")
        assert result == "TestOp_result"

    def test_operator_node_chain(self):
        op1 = DummyOperator(name="Op1")
        op2 = DummyOperator(name="Op2")
        chain = op1.then(op2)
        assert isinstance(chain, ChainOperator)
        assert len(chain.operators) == 2

    def test_operator_node_rshift(self):
        op1 = DummyOperator(name="Op1")
        op2 = DummyOperator(name="Op2")
        chain = op1 >> op2
        assert isinstance(chain, ChainOperator)


class TestChainOperator:
    def test_chain_operator_creation(self):
        op1 = DummyOperator(name="Op1")
        op2 = DummyOperator(name="Op2")
        chain = ChainOperator([op1, op2], name="TestChain")
        assert chain.name == "TestChain"
        assert len(chain.operators) == 2

    def test_chain_operator_execute(self):
        op1 = DummyOperator(name="Op1", return_value="result1")
        op2 = DummyOperator(name="Op2", return_value="result2")
        chain = ChainOperator([op1, op2])
        result = chain.execute("input")
        assert result == "result2"

    def test_chain_operator_chaining(self):
        op1 = DummyOperator(name="Op1")
        op2 = DummyOperator(name="Op2")
        op3 = DummyOperator(name="Op3")
        chain = ChainOperator([op1, op2, op3])
        assert len(chain.operators) == 3

    def test_chain_operator_serialize(self):
        op1 = DummyOperator(name="Op1")
        op2 = DummyOperator(name="Op2")
        chain = ChainOperator([op1, op2])
        serialized = chain.serialize()
        assert "operators" in serialized

    def test_chain_operator_serialize_deserialize(self):
        op1 = DummyOperator(name="Op1")
        op2 = DummyOperator(name="Op2")
        chain = ChainOperator([op1, op2])
        serialized = chain.serialize()
        assert len(serialized["operators"]) == 2

    def test_chain_operator_empty(self):
        chain = ChainOperator([])
        result = chain.execute("input")
        assert result == "input"
