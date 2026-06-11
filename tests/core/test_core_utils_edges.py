"""serializable.py + cond_builder + lookback 边界测试 (15 tests)。"""
from __future__ import annotations

import pytest

from QuantNodes.core.serializable import (
    Serializable,
    _REGISTRY,
    serializable,
)


# ============================================================================
# 1. Serializable mixin (5 tests)
# ============================================================================

class TestSerializable:
    def test_registry_registered(self):
        @serializable
        class Foo(Serializable):
            def _get_serializable_fields(self):
                return {"x": 1}

            @classmethod
            def _from_dict_impl(cls, data):
                return cls()

        assert "Foo" in _REGISTRY
        assert _REGISTRY["Foo"] is Foo

    def test_serialize_roundtrip(self):
        @serializable
        class Bar(Serializable):
            def __init__(self, x=0, y=""):
                self.x = x
                self.y = y

            def _get_serializable_fields(self):
                return {"x": self.x, "y": self.y}

            @classmethod
            def _from_dict_impl(cls, data):
                return cls(x=data["x"], y=data["y"])

        obj = Bar(x=42, y="hello")
        d = obj.serialize()
        assert d["type"] == "Bar"
        assert d["x"] == 42
        assert d["y"] == "hello"
        assert d["_schema_version"] == "1.0"
        obj2 = Bar.deserialize(d)
        assert obj2.x == 42

    def test_missing_type_raises(self):
        with pytest.raises(ValueError, match="Missing 'type'"):
            Serializable.deserialize({})

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown serializable type"):
            Serializable.deserialize({"type": "NotRegistered"})

    def test_serializable_decorator_returns_class(self):
        @serializable
        class Baz:
            pass
        assert Baz.__name__ == "Baz"


# ============================================================================
# 2. ast_parser.parse_expression (5 tests)
# ============================================================================

class TestParseExpression:
    def test_simple_compare(self):
        from QuantNodes.core.ast_parser import parse_expression
        expr = parse_expression("close > 5")
        assert expr is not None

    def test_arith(self):
        from QuantNodes.core.ast_parser import parse_expression
        expr = parse_expression("close - open")
        assert expr is not None

    def test_function_call(self):
        from QuantNodes.core.ast_parser import parse_expression
        # method call on attribute: data.clip(0, 1)
        expr = parse_expression("x.clip(0, 1)")
        assert expr is not None

    def test_invalid_syntax_raises(self):
        from QuantNodes.core.ast_parser import parse_expression
        with pytest.raises(Exception):
            parse_expression("close >")

    def test_forbidden_node_raises(self):
        from QuantNodes.core.ast_parser import parse_expression
        with pytest.raises(Exception):
            parse_expression("__import__('os')")


# ============================================================================
# 3. cond_builder / lookback 边界 (5 tests)
# ============================================================================

class TestCondBuilder:
    def test_cond_call(self):
        from QuantNodes.core.cond_builder import Cond
        c = Cond("close") > 5
        assert c is not None

    def test_cond_attr(self):
        from QuantNodes.core.cond_builder import Cond
        c = Cond.attr("metrics")
        assert c is not None

    def test_cond_getitem(self):
        from QuantNodes.core.cond_builder import Cond
        c = Cond["close"]
        assert c is not None

    def test_cond_input(self):
        from QuantNodes.core.cond_builder import Cond
        c = Cond.input
        assert c is not None


class TestLookbackHelpers:
    def test_compute_lookback_params(self):
        from QuantNodes.core._lookback_helpers import compute_lookback_params
        p = compute_lookback_params([20], "rolling")
        assert p is not None

    def test_extend_dt_ruler(self):
        from QuantNodes.core._lookback_helpers import extend_dt_ruler
        import pandas as pd
        ruler = pd.DatetimeIndex(["2025-01-01"])
        dts = pd.DatetimeIndex(["2024-12-15"])
        out = extend_dt_ruler(ruler, dts, max_lookback=30)
        assert len(out) >= 1

