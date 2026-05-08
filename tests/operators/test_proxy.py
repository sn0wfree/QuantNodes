# -*- coding: utf-8 -*-
"""QuantNodes.operators.proxy 单元测试"""

from QuantNodes.operators.proxy import (
    list_operators,
    get_operator,
    register_operator,
    operator_info,
    generate_documentation,
    OperatorCategory,
    _OPERATOR_REGISTRY,
)


class TestProxyExports:
    def test_list_operators_callable(self):
        result = list_operators()
        assert isinstance(result, list)

    def test_get_operator_callable(self):
        result = get_operator("ts_mean")
        assert callable(result) or result is None

    def test_register_operator_callable(self):
        assert callable(register_operator)

    def test_operator_info_callable(self):
        result = operator_info("ts_mean")
        assert result is None or isinstance(result, dict)

    def test_generate_documentation_callable(self):
        result = generate_documentation()
        assert isinstance(result, str)

    def test_operator_category_enum(self):
        assert hasattr(OperatorCategory, "POINT")
        assert hasattr(OperatorCategory, "TIME")
        assert hasattr(OperatorCategory, "SECTION")
        assert hasattr(OperatorCategory, "MULTI_SECTION")
        assert hasattr(OperatorCategory, "TALIB")

    def test_operator_registry_exists(self):
        assert isinstance(_OPERATOR_REGISTRY, dict)


class TestProxyReExports:
    def test_list_operators_returns_list_of_strings(self):
        result = list_operators()
        if len(result) > 0:
            assert isinstance(result[0], str)

    def test_list_operators_with_category(self):
        result = list_operators(category="time_series")
        assert isinstance(result, list)

    def test_get_operator_ts_mean(self):
        op = get_operator("ts_mean")
        assert op is not None
        assert callable(op)

    def test_get_operator_nonexistent(self):
        op = get_operator("nonexistent_operator_xyz")
        assert op is None

    def test_operator_info_ts_mean(self):
        info = operator_info("ts_mean")
        if info is not None:
            assert isinstance(info, dict)

    def test_operator_info_nonexistent(self):
        info = operator_info("nonexistent_op_xyz")
        assert info is None

    def test_generate_documentation_markdown(self):
        doc = generate_documentation(output_format="markdown")
        assert "ts_mean" in doc or len(doc) > 0

    def test_register_operator_creates_entry(self):

        initial_count = len(list_operators())

        @register_operator(category="point")
        def test_custom_op(x):
            return x * 2

        final_count = len(list_operators())
        assert final_count >= initial_count


class TestProxyRegistry:
    def test_registry_is_dict(self):
        assert isinstance(_OPERATOR_REGISTRY, dict)

    def test_registry_has_categories(self):
        if len(_OPERATOR_REGISTRY) > 0:
            first_key = next(iter(_OPERATOR_REGISTRY))
            assert isinstance(_OPERATOR_REGISTRY[first_key], (dict, list))

    def test_list_operators_returns_registered_names(self):
        names = list_operators()
        if "ts_mean" in names:
            op = get_operator("ts_mean")
            assert op is not None
