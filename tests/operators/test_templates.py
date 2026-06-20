# -*- coding: utf-8 -*-
"""QuantNodes.operators.templates 单元测试"""
import pytest
import polars as pl

from QuantNodes.operators.templates import OperatorTemplate
from QuantNodes.factor_node.factor_functions import get_operator


class TestOperatorTemplateBasics:
    """OperatorTemplate 基础测试"""

    def test_template_creation(self):
        template = OperatorTemplate(
            name="ma5",
            category="time",
            template="ts_mean",
            defaults={"window": 5}
        )
        assert template.name == "ma5"
        assert template.category == "time"
        assert template.template == "ts_mean"

    def test_template_repr(self):
        template = OperatorTemplate(
            name="test_template",
            category="point",
            template="test_op"
        )
        repr_str = repr(template)
        assert "test_template" in repr_str

    def test_template_properties(self):
        template = OperatorTemplate(
            name="std5",
            category="time",
            template="ts_std",
            defaults={"window": 5}
        )
        assert template.name == "std5"
        assert template.category == "time"
        assert template.template == "ts_std"
        assert template.defaults == {"window": 5}


class TestOperatorTemplateCall:
    """OperatorTemplate 调用测试"""

    def test_template_call_returns_expr(self):
        template = OperatorTemplate(
            name="ma5",
            category="time",
            template="ts_mean",
            defaults={"window": 5}
        )
        result = template(pl.col("close"))
        assert isinstance(result, pl.Expr)

    def test_template_call_with_override(self):
        template = OperatorTemplate(
            name="ma10",
            category="time",
            template="ts_mean",
            defaults={"window": 5}
        )
        result = template(pl.col("price"), window=10)
        assert isinstance(result, pl.Expr)

    def test_template_call_invalid_template(self):
        template = OperatorTemplate(
            name="invalid",
            category="invalid_category",
            template="nonexistent_template"
        )
        with pytest.raises(ValueError, match="not found"):
            template(pl.col("x"))


class TestOperatorTemplateRegistration:
    """OperatorTemplate 注册测试"""

    def setup_method(self):
        from QuantNodes.operators.registry import _CustomOperatorRegistry
        _CustomOperatorRegistry.unregister_all()

    def test_template_register(self):
        template = OperatorTemplate(
            name="test_reg_template",
            category="time",
            template="ts_mean",
            defaults={"window": 3}
        )
        template.register()
        op = get_operator("test_reg_template")
        assert op is not None


class TestOperatorTemplateDefaults:
    """OperatorTemplate 默认参数测试"""

    def test_default_params_applied(self):
        template = OperatorTemplate(
            name="zscore_template",
            category="section",
            template="zscore",
            defaults={}
        )
        result = template(pl.col("factor"))
        assert isinstance(result, pl.Expr)


class TestOperatorTemplateInContext:
    """OperatorTemplate 在表达式上下文中测试"""

    def test_template_in_select(self):
        template = OperatorTemplate(
            name="sma5",
            category="time",
            template="ts_mean",
            defaults={"window": 5}
        )
        df = pl.DataFrame({"price": [10.0, 11.0, 12.0, 13.0, 14.0]})
        result = df.select([
            pl.col("price"),
            template(pl.col("price")).alias("sma")
        ])
        assert "sma" in result.columns


class TestOperatorTemplateEdgeCases:
    """OperatorTemplate 边界情况测试"""

    def test_template_with_empty_defaults(self):
        template = OperatorTemplate(
            name="empty_defaults",
            category="time",
            template="ts_mean",
            defaults={}
        )
        result = template(pl.col("x"))
        assert isinstance(result, pl.Expr)


class TestOperatorTemplateReusability:
    """OperatorTemplate 可复用性测试"""

    def test_template_reuse_multiple_times(self):
        template = OperatorTemplate(
            name="reuse_template",
            category="time",
            template="ts_mean",
            defaults={"window": 3}
        )
        df = pl.DataFrame({
            "price1": [10.0, 11.0, 12.0],
            "price2": [20.0, 21.0, 22.0]
        })
        result = df.select([
            template(pl.col("price1")).alias("ma1"),
            template(pl.col("price2")).alias("ma2")
        ])
        assert "ma1" in result.columns
        assert "ma2" in result.columns
