# coding=utf-8
"""QuantNodes.factor_node.factor_functions 单元测试"""

import numpy as np
import polars as pl
from QuantNodes.factor_node.factor_functions import (
    list_operators,
    get_operator,
    operator_info,
    generate_documentation,
    register_operator,
    OperatorCategory,
    _OPERATOR_REGISTRY,
    _ensure_expr,
)
from QuantNodes.factor_node.factor_functions._helpers import (
    _cum_single_median,
    _cum_single_quantile,
    _cum_dual_corr,
    _CUM_SINGLE_FUNCS,
    _CUM_DUAL_FUNCS,
)


class TestOperatorRegistry:
    def test_list_operators_returns_list(self):
        result = list_operators()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_list_operators_with_category(self):
        result = list_operators(category="point")
        assert isinstance(result, list)

    def test_get_operator_existing(self):
        op = get_operator("abs")
        assert op is not None
        assert callable(op)

    def test_get_operator_nonexistent(self):
        op = get_operator("nonexistent_operator_xyz")
        assert op is None

    def test_operator_info_existing(self):
        info = operator_info("abs")
        assert info is not None
        assert info["name"] == "abs"

    def test_operator_info_nonexistent(self):
        info = operator_info("nonexistent_operator_xyz")
        assert info is None

    def test_generate_documentation_markdown(self):
        doc = generate_documentation(output_format="markdown")
        assert isinstance(doc, str)
        assert len(doc) > 0

    def test_generate_documentation_json(self):
        doc = generate_documentation(output_format="json")
        assert isinstance(doc, str)
        assert "abs" in doc


class TestOperatorCategory:
    def test_category_constants(self):
        assert OperatorCategory.POINT == "point"
        assert OperatorCategory.TIME == "time"
        assert OperatorCategory.SECTION == "section"


class TestEnsureExpr:
    def test_ensure_expr_with_expr(self):
        expr = pl.col("test")
        result = _ensure_expr(expr)
        assert isinstance(result, pl.Expr)

    def test_ensure_expr_with_string(self):
        result = _ensure_expr("test_col")
        assert isinstance(result, pl.Expr)

    def test_ensure_expr_with_literal(self):
        result = _ensure_expr(42.0)
        assert isinstance(result, pl.Expr)


class TestHelpers:
    def test_cum_single_median_odd(self):
        result = _cum_single_median([1.0, 2.0, 3.0])
        assert result == 2.0

    def test_cum_single_median_even(self):
        result = _cum_single_median([1.0, 2.0, 3.0, 4.0])
        assert result == 2.5

    def test_cum_single_quantile(self):
        result = _cum_single_quantile([1.0, 2.0, 3.0, 4.0], quantile=0.5)
        assert result == 2.5

    def test_cum_dual_corr_valid(self):
        arr1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        arr2 = [2.0, 4.0, 6.0, 8.0, 10.0]
        result = _cum_dual_corr(arr1, arr2)
        assert result is not None
        assert abs(result - 1.0) < 0.001

    def test_cum_dual_corr_insufficient(self):
        result = _cum_dual_corr([1.0], [2.0])
        assert result is None

    def test_cum_single_funcs_keys(self):
        assert "median" in _CUM_SINGLE_FUNCS
        assert "kurt" in _CUM_SINGLE_FUNCS
        assert "skew" in _CUM_SINGLE_FUNCS

    def test_cum_dual_funcs_keys(self):
        assert "corr" in _CUM_DUAL_FUNCS
        assert "cov" in _CUM_DUAL_FUNCS


class TestRegisterOperator:
    def test_register_operator_decorator(self):
        @register_operator(OperatorCategory.POINT)
        def test_operator(value):
            return value

        assert "test_operator" in _OPERATOR_REGISTRY[OperatorCategory.POINT]
        del _OPERATOR_REGISTRY[OperatorCategory.POINT]["test_operator"]


class TestMathPointOperators:
    def test_abs_operator(self):
        from QuantNodes.factor_node.factor_functions.math_ops import abs as ff_abs
        df = pl.DataFrame({"a": [-1.0, 2.0, -3.0]})
        result = df.select(ff_abs(pl.col("a")))
        assert result["a"].to_list() == [1.0, 2.0, 3.0]

    def test_sign_operator(self):
        from QuantNodes.factor_node.factor_functions.math_ops import sign
        df = pl.DataFrame({"a": [-5.0, 0.0, 5.0]})
        result = df.select(sign(pl.col("a")))
        assert result["a"].to_list() == [-1.0, 0.0, 1.0]

    def test_sqrt_operator(self):
        from QuantNodes.factor_node.factor_functions.math_ops import sqrt
        df = pl.DataFrame({"a": [1.0, 4.0, 9.0]})
        result = df.select(sqrt(pl.col("a")))
        assert np.allclose(result["a"].to_list(), [1.0, 2.0, 3.0])

    def test_add_operator(self):
        from QuantNodes.factor_node.factor_functions.math_ops import add
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = df.select(add(pl.col("a"), pl.col("b")))
        assert result["a"].to_list() == [4.0, 6.0]

    def test_mul_operator(self):
        from QuantNodes.factor_node.factor_functions.math_ops import mul
        df = pl.DataFrame({"a": [2.0, 3.0], "b": [4.0, 5.0]})
        result = df.select(mul(pl.col("a"), pl.col("b")))
        assert result["a"].to_list() == [8.0, 15.0]

    def test_fill_zero(self):
        from QuantNodes.factor_node.factor_functions.math_ops import fill_zero
        df = pl.DataFrame({"a": [1.0, None, 3.0]})
        result = df.select(fill_zero(pl.col("a")))
        assert result["a"].to_list() == [1.0, 0.0, 3.0]

    def test_isnull_operator(self):
        from QuantNodes.factor_node.factor_functions.math_ops import isnull
        df = pl.DataFrame({"a": [1.0, None, 3.0]})
        result = df.select(isnull(pl.col("a")))
        assert result["a"].to_list() == [False, True, False]

    def test_ceil_operator(self):
        from QuantNodes.factor_node.factor_functions.math_ops import ceil
        df = pl.DataFrame({"a": [1.2, 2.7, 3.0]})
        result = df.select(ceil(pl.col("a")))
        assert result["a"].to_list() == [2.0, 3.0, 3.0]

    def test_floor_operator(self):
        from QuantNodes.factor_node.factor_functions.math_ops import floor
        df = pl.DataFrame({"a": [1.2, 2.7, 3.0]})
        result = df.select(floor(pl.col("a")))
        assert result["a"].to_list() == [1.0, 2.0, 3.0]


class TestSectionOperators:
    def test_zscore_operator(self):
        from QuantNodes.factor_node.factor_functions.section_ops import zscore
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = df.select(zscore(pl.col("a")))
        assert len(result) == 5

    def test_rank_operator(self):
        from QuantNodes.factor_node.factor_functions.section_ops import rank
        df = pl.DataFrame({"a": [3.0, 1.0, 2.0]})
        result = df.select(rank(pl.col("a")))
        assert len(result) == 3

    def test_winsorize_operator(self):
        from QuantNodes.factor_node.factor_functions.section_ops import winsorize
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = df.select(winsorize(pl.col("a"), lower=0.1, upper=0.1))
        assert len(result) == 5

    def test_scale_operator(self):
        from QuantNodes.factor_node.factor_functions.section_ops import scale
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = df.select(scale(pl.col("a")))
        assert len(result) == 3

    def test_neutralize_operator(self):
        from QuantNodes.factor_node.factor_functions.section_ops import neutralize
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "group": ["g1", "g1", "g2"]})
        result = df.select(neutralize(pl.col("a")))
        assert len(result) == 3


class TestCompositeOperators:
    def test_aggregate_mean(self):
        from QuantNodes.factor_node.factor_functions.composite_ops import aggregate
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "group": ["g1", "g1", "g2"]})
        result = df.select(aggregate(pl.col("a"), "group", method="mean"))
        assert len(result) == 3

    def test_aggr_sum(self):
        from QuantNodes.factor_node.factor_functions.composite_ops import aggr_sum
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "group": ["g1", "g1", "g2"]})
        result = df.select(aggr_sum(pl.col("a"), "group"))
        assert len(result) == 3

    def test_merge_operator(self):
        from QuantNodes.factor_node.factor_functions.composite_ops import merge
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = df.select(merge([pl.col("a"), pl.col("b")], method="add"))
        assert len(result) == 2

    def test_blend_operator(self):
        from QuantNodes.factor_node.factor_functions.composite_ops import blend
        df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        result = df.select(blend(pl.col("a"), pl.col("b"), alpha=0.5))
        assert result["a"].to_list() == [2.0, 3.0]


class TestTimeSeriesOperators:
    def test_rolling_mean(self):
        from QuantNodes.factor_node.factor_functions.time_ops import rolling_mean
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = df.select(rolling_mean(pl.col("a"), window=3))
        assert len(result) == 5

    def test_rolling_max(self):
        from QuantNodes.factor_node.factor_functions.time_ops import rolling_max
        df = pl.DataFrame({"a": [1.0, 5.0, 3.0, 2.0, 4.0]})
        result = df.select(rolling_max(pl.col("a"), window=3))
        assert result["a"].to_list()[2] == 5.0

    def test_rolling_std(self):
        from QuantNodes.factor_node.factor_functions.time_ops import rolling_std
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = df.select(rolling_std(pl.col("a"), window=3))
        assert len(result) == 5

    def test_ts_delta(self):
        from QuantNodes.factor_node.factor_functions.time_ops import ts_delta
        df = pl.DataFrame({"a": [1.0, 3.0, 6.0, 10.0]})
        result = df.select(ts_delta(pl.col("a")))
        assert len(result) == 4

    def test_ts_lag(self):
        from QuantNodes.factor_node.factor_functions.time_ops import ts_lag
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = df.select(ts_lag(pl.col("a")))
        assert result["a"].to_list()[0] is None

    def test_expanding_mean(self):
        from QuantNodes.factor_node.factor_functions.time_ops import expanding_mean
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        result = df.select(expanding_mean(pl.col("a")))
        assert len(result) == 4

    def test_ewm_mean(self):
        from QuantNodes.factor_node.factor_functions.time_ops import ewm_mean
        df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        result = df.select(ewm_mean(pl.col("a"), alpha=0.5))
        assert len(result) == 4


class TestAliases:
    def test_abs_alias_available(self):
        from QuantNodes.factor_node.factor_functions import abs
        assert callable(abs)

    def test_log_alias_available(self):
        from QuantNodes.factor_node.factor_functions import log
        assert callable(log)

    def test_pow_alias_available(self):
        from QuantNodes.factor_node.factor_functions import pow
        assert callable(pow)

    def test_diff_alias(self):
        from QuantNodes.factor_node.factor_functions import delta
        assert callable(delta)
