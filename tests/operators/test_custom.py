# -*- coding: utf-8 -*-
"""QuantNodes.operators.custom 单元测试"""
import pytest

import polars as pl


class TestCustomOperatorImports:
    def test_custom_operator_importable(self):
        from QuantNodes.operators import CustomOperator

        assert hasattr(CustomOperator, "point")
        assert hasattr(CustomOperator, "time")
        assert hasattr(CustomOperator, "section")
        assert hasattr(CustomOperator, "multi_section")
        assert hasattr(CustomOperator, "talib")

    def test_operator_template_importable(self):
        from QuantNodes.operators import OperatorTemplate

        assert callable(OperatorTemplate)

    def test_decorator_functions_importable(self):
        from QuantNodes.operators import point, time, section

        assert callable(point)
        assert callable(time)
        assert callable(section)


class TestCustomOperatorRegistry:
    def setup_method(self):
        from QuantNodes.operators.registry import _CustomOperatorRegistry

        _CustomOperatorRegistry.unregister_all()

    def test_unregister_all_clears_all(self):
        from QuantNodes.operators import CustomOperator
        from QuantNodes.operators.registry import _CustomOperatorRegistry

        @CustomOperator.point("test_op_1")
        def op1(f):
            return f

        @CustomOperator.time("test_op_2")
        def op2(f):
            return f

        assert _CustomOperatorRegistry.count() == 2
        CustomOperator.unregister_all()
        assert _CustomOperatorRegistry.count() == 0

    def test_unregister_specific(self):
        from QuantNodes.operators import CustomOperator
        from QuantNodes.operators.registry import _CustomOperatorRegistry

        @CustomOperator.point("test_op_x")
        def opx(f):
            return f

        assert _CustomOperatorRegistry.count() == 1
        result = CustomOperator.unregister("test_op_x")
        assert result is True
        assert _CustomOperatorRegistry.count() == 0

    def test_unregister_nonexistent_returns_false(self):
        from QuantNodes.operators import CustomOperator

        result = CustomOperator.unregister("nonexistent_op_xyz")
        assert result is False


class TestCustomOperatorBuilder:
    def setup_method(self):
        from QuantNodes.operators.registry import _CustomOperatorRegistry

        _CustomOperatorRegistry.unregister_all()

    def test_decorator_style_point_op(self):
        from QuantNodes.operators import CustomOperator
        from QuantNodes.factor_node.factor_functions import get_operator

        @CustomOperator.point("my_double")
        def my_double(f, multiplier=2.0):
            return f * multiplier

        func = get_operator("my_double")
        assert func is not None

        expr = my_double(pl.col("close"))
        assert isinstance(expr, pl.Expr)

    def test_decorator_style_time_op(self):
        from QuantNodes.operators import CustomOperator
        from QuantNodes.factor_node.factor_functions import get_operator

        @CustomOperator.time("my_rolling_sum")
        def my_rolling_sum(f, window=5):
            return f.rolling_sum(window)

        func = get_operator("my_rolling_sum")
        assert func is not None

    def test_decorator_style_section_op(self):
        from QuantNodes.operators import CustomOperator
        from QuantNodes.factor_node.factor_functions import get_operator

        @CustomOperator.section("my_zscore")
        def my_zscore(f, eps=1e-8):
            return (f - f.mean()) / (f.std() + eps)

        func = get_operator("my_zscore")
        assert func is not None

    def test_builder_style_point_op(self):
        from QuantNodes.operators import CustomOperator
        from QuantNodes.factor_node.factor_functions import get_operator

        my_times_two = (
            CustomOperator.point("my_times_two")
            .param("multiplier", float, 2.0, "乘数")
            .execute(lambda f, multiplier: f * multiplier)
            .doc("将因子乘以常数")
            .register()
        )

        func = get_operator("my_times_two")
        assert func is not None

        expr = my_times_two(pl.col("close"))
        assert isinstance(expr, pl.Expr)

    def test_builder_with_alias(self):
        from QuantNodes.operators import CustomOperator
        from QuantNodes.factor_node.factor_functions import get_operator

        @CustomOperator.point("my_alias_test").alias("alias_1").alias("alias_2")
        def my_alias_test(f):
            return f

        assert get_operator("my_alias_test") is not None
        assert get_operator("alias_1") is not None
        assert get_operator("alias_2") is not None

    def test_builder_register_without_execute_raises(self):
        from QuantNodes.operators import CustomOperator

        builder = CustomOperator.point("bad_op")

        with pytest.raises(ValueError, match="execute"):
            builder.register()

    def test_custom_operator_list(self):
        from QuantNodes.operators import CustomOperator

        CustomOperator.unregister_all()

        @CustomOperator.point("custom_list_op1")
        def op1(f):
            return f

        @CustomOperator.time("custom_list_op2")
        def op2(f):
            return f

        custom_list = CustomOperator.list()
        assert "custom_list_op1" in custom_list
        assert "custom_list_op2" in custom_list

    def test_custom_operator_count(self):
        from QuantNodes.operators import CustomOperator

        CustomOperator.unregister_all()

        assert CustomOperator.count() == 0

        @CustomOperator.point("count_test_op")
        def op(f):
            return f

        assert CustomOperator.count() == 1


class TestOperatorTemplate:
    def setup_method(self):
        from QuantNodes.operators.registry import _CustomOperatorRegistry

        _CustomOperatorRegistry.unregister_all()

    def test_template_creation(self):
        from QuantNodes.operators import OperatorTemplate

        template = OperatorTemplate(
            name="my_ewm_30",
            category="time",
            template="ewm_mean",
            defaults={"span": 30},
        )

        assert template.name == "my_ewm_30"
        assert template.category == "time"
        assert template.template == "ewm_mean"
        assert template.defaults == {"span": 30}

    def test_template_call(self):
        from QuantNodes.operators import OperatorTemplate

        template = OperatorTemplate(
            name="my_ewm_30",
            category="time",
            template="ewm_mean",
            defaults={"span": 30},
        )

        expr = template(pl.col("close"))
        assert isinstance(expr, pl.Expr)

    def test_template_register(self):
        from QuantNodes.operators import OperatorTemplate
        from QuantNodes.factor_node.factor_functions import get_operator

        template = OperatorTemplate(
            name="my_ewm_30_reg",
            category="time",
            template="ewm_mean",
            defaults={"span": 30},
        )

        template.register()

        func = get_operator("my_ewm_30_reg")
        assert func is not None

    def test_template_with_override(self):
        from QuantNodes.operators import OperatorTemplate

        template = OperatorTemplate(
            name="my_ewm_override",
            category="time",
            template="ewm_mean",
            defaults={"span": 30},
        )

        expr = template(pl.col("close"), span=10)
        assert isinstance(expr, pl.Expr)

    def test_template_repr(self):
        from QuantNodes.operators import OperatorTemplate

        template = OperatorTemplate(
            name="repr_test",
            category="time",
            template="ewm_mean",
            defaults={"span": 30},
        )

        r = repr(template)
        assert "repr_test" in r
        assert "ewm_mean" in r


class TestCascadeLookup:
    def setup_method(self):
        from QuantNodes.operators.registry import _CustomOperatorRegistry

        _CustomOperatorRegistry.unregister_all()

    def test_custom_then_builtin_lookup(self):
        from QuantNodes.operators import CustomOperator
        from QuantNodes.factor_node.factor_functions import get_operator

        @CustomOperator.point("my_custom_ts_mean")
        def my_ts_mean(f):
            return f

        func = get_operator("my_custom_ts_mean")
        assert func is not None

        builtin_func = get_operator("ts_mean")
        assert builtin_func is not None

    def test_builtin_still_accessible(self):
        from QuantNodes.factor_node.factor_functions import get_operator

        func = get_operator("ts_mean")
        assert func is not None

    def test_custom_not_in_builtin_registry(self):
        from QuantNodes.operators import CustomOperator
        from QuantNodes.factor_node.factor_functions import _OPERATOR_REGISTRY

        CustomOperator.unregister_all()

        @CustomOperator.point("not_in_builtin")
        def op(f):
            return f

        assert "not_in_builtin" not in _OPERATOR_REGISTRY.get("point", {})


class TestIntegrationWithExpressionParser:
    def setup_method(self):
        from QuantNodes.operators.registry import _CustomOperatorRegistry

        _CustomOperatorRegistry.unregister_all()

    def test_custom_operator_in_expr_context(self):
        from QuantNodes.operators import CustomOperator
        from QuantNodes.factor_node.factor_functions import get_operator

        @CustomOperator.time("my_momentum")
        def my_momentum(f, window=20):
            return f / f.shift(window) - 1

        func = get_operator("my_momentum")
        assert func is not None

        expr = func(pl.col("close"), window=60)
        assert isinstance(expr, pl.Expr)

    def test_chained_operators(self):
        from QuantNodes.operators import CustomOperator
        from QuantNodes.factor_node.factor_functions import get_operator, zscore

        @CustomOperator.time("my_rolling_median")
        def my_rolling_median(f, window=5):
            return f.rolling_median(window)

        func = get_operator("my_rolling_median")
        assert func is not None

        expr = func(pl.col("close"), window=10)
        assert isinstance(expr, pl.Expr)

        zscore_expr = zscore(expr)
        assert isinstance(zscore_expr, pl.Expr)


class TestBackwardCompatibility:
    def setup_method(self):
        from QuantNodes.operators.registry import _CustomOperatorRegistry

        _CustomOperatorRegistry.unregister_all()

    def test_existing_register_operator_still_works(self):
        from QuantNodes.operators.proxy import register_operator, get_operator

        @register_operator(category="point")
        def old_style_op(f):
            return f.abs()

        func = get_operator("old_style_op")
        assert func is not None

    def test_existing_builtin_operators_still_accessible(self):
        from QuantNodes.factor_node.factor_functions import get_operator

        funcs = [
            get_operator("ts_mean"),
            get_operator("zscore"),
            get_operator("rank"),
            get_operator("rolling_sum"),
        ]

        for func in funcs:
            assert func is not None


class TestCustomOperatorBuilderChain:
    """CustomOperator Builder 链式调用完整测试"""

    def setup_method(self):
        from QuantNodes.operators.registry import _CustomOperatorRegistry
        _CustomOperatorRegistry.unregister_all()

    def test_builder_full_chain_point(self):
        from QuantNodes.operators import CustomOperator
        from QuantNodes.factor_node.factor_functions import get_operator

        result = (
            CustomOperator.point("chain_test_1")
            .param("a", float, 1.0, "参数A")
            .param("b", int, 10, "参数B")
            .execute(lambda f, a, b: f * a + b)
            .doc("测试完整链式调用")
            .alias("chain_alias_1")
            .alias("chain_alias_2")
            .register()
        )

        assert get_operator("chain_test_1") is not None
        assert get_operator("chain_alias_1") is not None
        assert get_operator("chain_alias_2") is not None
        assert callable(result)

    def test_builder_full_chain_time(self):
        from QuantNodes.operators import CustomOperator
        from QuantNodes.factor_node.factor_functions import get_operator

        result = (
            CustomOperator.time("chain_time_1")
            .param("window", int, 5, "窗口大小")
            .execute(lambda f, window: f.rolling_mean(window))
            .doc("时间序列算子")
            .register()
        )

        assert get_operator("chain_time_1") is not None
        assert callable(result)

    def test_builder_full_chain_section(self):
        from QuantNodes.operators import CustomOperator
        from QuantNodes.factor_node.factor_functions import get_operator

        (
            CustomOperator.section("chain_section_1")
            .param("method", str, "zscore", "标准化方法")
            .execute(lambda f, method: (f - f.mean()) / f.std())
            .doc("截面算子")
            .register()
        )

        assert get_operator("chain_section_1") is not None

    def test_builder_multiple_params_override(self):
        from QuantNodes.operators import CustomOperator

        op = (
            CustomOperator.point("multi_param_test")
            .param("x", float, 1.0, "参数X")
            .param("y", float, 2.0, "参数Y")
            .param("z", int, 5, "参数Z")
            .execute(lambda f, x, y, z: f * x * y + z)
            .register()
        )

        df = pl.DataFrame({"value": [1.0, 2.0, 3.0]})
        result = df.select(op(pl.col("value"), x=2.0, y=0.5, z=10))
        assert isinstance(result, pl.DataFrame)

    def test_builder_param_type_validation(self):
        from QuantNodes.operators import CustomOperator

        builder = (
            CustomOperator.point("type_test")
            .param("count", int, 0, "计数")
            .param("ratio", float, 1.0, "比率")
            .param("name", str, "default", "名称")
            .execute(lambda f, count, ratio, name: f * ratio)
        )

        assert builder._params["count"]["type"] is int  # noqa: E721
        assert builder._params["ratio"]["type"] is float  # noqa: E721
        assert builder._params["name"]["type"] is str  # noqa: E721

    def test_builder_duplicate_alias_ignored(self):
        from QuantNodes.operators import CustomOperator
        from QuantNodes.factor_node.factor_functions import get_operator

        @CustomOperator.point("dup_alias_test").alias("dup1").alias("dup1").alias("dup2")
        def dup_alias_test(f):
            return f

        assert get_operator("dup_alias_test") is not None
        assert get_operator("dup1") is not None
        assert get_operator("dup2") is not None

    def test_builder_empty_docstring(self):
        from QuantNodes.operators import CustomOperator

        result = (
            CustomOperator.point("empty_doc")
            .execute(lambda f: f)
            .doc("")
            .register()
        )

        assert result is not None

    def test_builder_chain_called_twice(self):
        from QuantNodes.operators import CustomOperator
        from QuantNodes.factor_node.factor_functions import get_operator

        builder = (
            CustomOperator.point("double_register")
            .execute(lambda f: f * 2)
        )

        builder.register()
        assert get_operator("double_register") is not None

    def test_builder_with_talib_category(self):
        from QuantNodes.operators import CustomOperator

        @CustomOperator.talib("my_talib_wrapper")
        def my_talib_wrapper(expr, period=14):
            from QuantNodes.operators import talib_ops
            return talib_ops.rsi(expr, timeperiod=period)

        from QuantNodes.factor_node.factor_functions import get_operator
        assert get_operator("my_talib_wrapper") is not None

    def test_builder_with_multi_section_category(self):
        from QuantNodes.operators import CustomOperator
        from QuantNodes.factor_node.factor_functions import get_operator

        @CustomOperator.multi_section("my_multi_section")
        def my_multi_section(f1, f2, method="corr"):
            return f1.rolling_corr(f2, 20)

        assert get_operator("my_multi_section") is not None

    def test_builder_param_with_none_default(self):
        from QuantNodes.operators import CustomOperator

        builder = (
            CustomOperator.point("none_default")
            .param("optional", float, None, "可选参数")
            .execute(lambda f, optional: f if optional is None else f * optional)
        )

        assert builder._params["optional"]["default"] is None

    def test_builder_multiple_aliases_lookup(self):
        from QuantNodes.operators import CustomOperator
        from QuantNodes.factor_node.factor_functions import get_operator

        @CustomOperator.point("main_name").alias("alias_a").alias("alias_b").alias("alias_c")
        def main_name(f):
            return f

        assert get_operator("main_name") is not None
        assert get_operator("alias_a") is not None
        assert get_operator("alias_b") is not None
        assert get_operator("alias_c") is not None
