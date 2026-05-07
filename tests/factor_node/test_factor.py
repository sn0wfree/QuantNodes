# coding=utf-8
"""QuantNodes.factor_node.factor 单元测试"""
import pytest
import numpy as np

from QuantNodes.factor_node.factor import DataType
from QuantNodes.factor_node.factor import _UnitaryOperator, _BinaryOperator
from QuantNodes.core.base import FactorError


class TestDataType:
    def test_dtype_enum_values(self):
        assert DataType.DOUBLE.value == "double"
        assert DataType.STRING.value == "string"
        assert DataType.OBJECT.value == "object"


class TestUnitaryOperator:
    def test_neg_operator(self):
        x = np.array([1.0, 2.0, 3.0])
        args = {"OperatorType": "neg"}
        result = _UnitaryOperator(None, None, None, [x], args)
        assert np.allclose(result, [-1.0, -2.0, -3.0])

    def test_abs_operator(self):
        x = np.array([-1.0, 2.0, -3.0])
        args = {"OperatorType": "abs"}
        result = _UnitaryOperator(None, None, None, [x], args)
        assert np.allclose(result, [1.0, 2.0, 3.0])

    def test_not_operator(self):
        x = np.array([True, False, True])
        args = {"OperatorType": "not"}
        result = _UnitaryOperator(None, None, None, [x], args)
        assert np.array_equal(result, [False, True, False])

    def test_unsupported_operator_raises_error(self):
        x = np.array([1.0, 2.0, 3.0])
        args = {"OperatorType": "unsupported"}
        with pytest.raises(FactorError, match="尚不支持的单因子运算符"):
            _UnitaryOperator(None, None, None, [x], args)


class TestBinaryOperator:
    def test_add_operator(self):
        x = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        args = {"OperatorType": "add", "SepInd": 1}
        result = _BinaryOperator(None, None, None, x, args)
        assert np.allclose(result, [4.0, 6.0])

    def test_sub_operator(self):
        x = [np.array([3.0, 4.0]), np.array([1.0, 2.0])]
        args = {"OperatorType": "sub", "SepInd": 1}
        result = _BinaryOperator(None, None, None, x, args)
        assert np.allclose(result, [2.0, 2.0])

    def test_mul_operator(self):
        x = [np.array([2.0, 3.0]), np.array([4.0, 5.0])]
        args = {"OperatorType": "mul", "SepInd": 1}
        result = _BinaryOperator(None, None, None, x, args)
        assert np.allclose(result, [8.0, 15.0])

    def test_div_operator(self):
        x = [np.array([6.0, 8.0]), np.array([2.0, 4.0])]
        args = {"OperatorType": "div", "SepInd": 1}
        result = _BinaryOperator(None, None, None, x, args)
        assert np.allclose(result, [3.0, 2.0])

    def test_div_by_zero_returns_nan(self):
        x = [np.array([6.0, 8.0]), np.array([0.0, 2.0])]
        args = {"OperatorType": "div", "SepInd": 1}
        result = _BinaryOperator(None, None, None, x, args)
        assert np.isnan(result[0])

    def test_pow_operator(self):
        x = [np.array([2.0, 3.0]), np.array([3.0, 2.0])]
        args = {"OperatorType": "pow", "SepInd": 1}
        result = _BinaryOperator(None, None, None, x, args)
        assert np.allclose(result, [8.0, 9.0])

    def test_comparison_lt(self):
        x = [np.array([1.0, 3.0]), np.array([2.0, 2.0])]
        args = {"OperatorType": "<", "SepInd": 1}
        result = _BinaryOperator(None, None, None, x, args)
        assert np.array_equal(result, [True, False])

    def test_comparison_eq(self):
        x = [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0])]
        args = {"OperatorType": "==", "SepInd": 1}
        result = _BinaryOperator(None, None, None, x, args)
        assert np.array_equal(result, [True, True, False])

    def test_unsupported_operator_raises_error(self):
        x = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        args = {"OperatorType": "unsupported", "SepInd": 1}
        with pytest.raises(FactorError, match="尚不支持的多因子运算符"):
            _BinaryOperator(None, None, None, x, args)
