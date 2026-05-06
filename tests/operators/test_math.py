# coding=utf-8
"""QuantNodes.operators.math 单元测试"""
import numpy as np
import polars as pl
import pytest

from QuantNodes.operators.math import MathOperators


@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [5.0, 4.0, 3.0, 2.0, 1.0],
        "c": [0.0, 1.0, 2.0, 3.0, 4.0],
    })


class TestMathOperatorsBasic:
    def test_add_with_scalar(self, sample_df):
        result = sample_df.select(MathOperators.add("a", 10.0))
        assert result["a"].to_list() == [11.0, 12.0, 13.0, 14.0, 15.0]

    def test_add_with_expr(self, sample_df):
        result = sample_df.select(MathOperators.add("a", pl.col("b")))
        assert result["a"].to_list() == [6.0, 6.0, 6.0, 6.0, 6.0]

    def test_add_with_expr_object(self, sample_df):
        result = sample_df.select(MathOperators.add(pl.col("a"), pl.col("b")))
        assert result["a"].to_list() == [6.0, 6.0, 6.0, 6.0, 6.0]

    def test_sub_with_scalar(self, sample_df):
        result = sample_df.select(MathOperators.sub("a", 1.0))
        assert result["a"].to_list() == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_sub_with_expr(self, sample_df):
        result = sample_df.select(MathOperators.sub("a", pl.col("b")))
        assert result["a"].to_list() == [-4.0, -2.0, 0.0, 2.0, 4.0]

    def test_mul_with_scalar(self, sample_df):
        result = sample_df.select(MathOperators.mul("a", 2.0))
        assert result["a"].to_list() == [2.0, 4.0, 6.0, 8.0, 10.0]

    def test_mul_with_expr(self, sample_df):
        result = sample_df.select(MathOperators.mul("a", pl.col("b")))
        assert result["a"].to_list() == [5.0, 8.0, 9.0, 8.0, 5.0]

    def test_div(self, sample_df):
        result = sample_df.select(MathOperators.div("a", 2.0))
        assert result["a"].to_list() == pytest.approx([0.5, 1.0, 1.5, 2.0, 2.5])

    def test_div_with_expr(self, sample_df):
        result = sample_df.select(MathOperators.div("a", pl.col("b")))
        assert result["a"].to_list() == pytest.approx([0.2, 0.5, 1.0, 2.0, 5.0])


class TestMathOperatorsLog:
    def test_log_natural(self, sample_df):
        result = sample_df.select(MathOperators.log("c"))
        assert result["c"].to_list()[1] == pytest.approx(0.0)

    def test_log_base2(self, sample_df):
        result = sample_df.select(MathOperators.log("c", base="2"))
        assert result["c"].to_list()[2] == pytest.approx(1.0)

    def test_log_base10(self, sample_df):
        result = sample_df.select(MathOperators.log("c", base="10"))
        assert result["c"].to_list()[3] == pytest.approx(np.log10(3.0))

    def test_log1p(self, sample_df):
        result = sample_df.select(MathOperators.log1p("c"))
        assert result["c"].to_list()[0] == pytest.approx(0.0)
        assert result["c"].to_list()[1] == pytest.approx(np.log1p(1.0))


class TestMathOperatorsPower:
    def test_abs(self, sample_df):
        df = pl.DataFrame({"a": [-1.0, 2.0, -3.0, 4.0, -5.0]})
        result = df.select(MathOperators.abs("a"))
        assert result["a"].to_list() == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_pow(self, sample_df):
        result = sample_df.select(MathOperators.pow("a", 2.0))
        assert result["a"].to_list() == [1.0, 4.0, 9.0, 16.0, 25.0]

    def test_sqrt(self, sample_df):
        result = sample_df.select(MathOperators.sqrt("a"))
        assert result["a"].to_list() == pytest.approx([1.0, 1.414, 1.732, 2.0, 2.236], rel=1e-3)


class TestMathOperatorsSignClip:
    def test_sign(self, sample_df):
        df = pl.DataFrame({"a": [-3.0, -1.0, 0.0, 1.0, 3.0]})
        result = df.select(MathOperators.sign("a"))
        assert result["a"].to_list() == [-1.0, -1.0, 0.0, 1.0, 1.0]

    def test_clip_no_bounds(self, sample_df):
        result = sample_df.select(MathOperators.clip("a"))
        assert result["a"].to_list() == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_clip_with_lower(self, sample_df):
        result = sample_df.select(MathOperators.clip("a", lower=2.5))
        assert result["a"].to_list() == [2.5, 2.5, 3.0, 4.0, 5.0]

    def test_clip_with_upper(self, sample_df):
        result = sample_df.select(MathOperators.clip("a", upper=3.5))
        assert result["a"].to_list() == [1.0, 2.0, 3.0, 3.5, 3.5]

    def test_clip_with_both_bounds(self, sample_df):
        result = sample_df.select(MathOperators.clip("a", lower=2.0, upper=4.0))
        assert result["a"].to_list() == [2.0, 2.0, 3.0, 4.0, 4.0]


class TestMathOperatorsRounding:
    def test_floor(self, sample_df):
        df = pl.DataFrame({"a": [1.2, 2.5, 3.7, 4.1, 5.9]})
        result = df.select(MathOperators.floor("a"))
        assert result["a"].to_list() == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_ceil(self, sample_df):
        df = pl.DataFrame({"a": [1.2, 2.5, 3.7, 4.1, 5.9]})
        result = df.select(MathOperators.ceil("a"))
        assert result["a"].to_list() == [2.0, 3.0, 4.0, 5.0, 6.0]

    def test_round(self, sample_df):
        df = pl.DataFrame({"a": [1.234, 2.567, 3.891, 4.999, 5.001]})
        result = df.select(MathOperators.round("a", decimals=1))
        assert result["a"].to_list() == [1.2, 2.6, 3.9, 5.0, 5.0]


class TestMathOperatorsNullHandling:
    def test_nan_to_null(self):
        df = pl.DataFrame({"a": [1.0, float("nan"), 3.0, None, 5.0]})
        result = df.select(MathOperators.nan_to_null("a"))
        assert result["a"].to_list() == [1.0, None, 3.0, None, 5.0]

    def test_fill_null_with_value(self, sample_df):
        df = pl.DataFrame({"a": [1.0, None, 3.0, None, 5.0]})
        result = df.select(MathOperators.fill_null("a", 99.0))
        assert result["a"].to_list() == [1.0, 99.0, 3.0, 99.0, 5.0]

    def test_fill_null_forward(self, sample_df):
        df = pl.DataFrame({"a": [1.0, None, 3.0, 4.0, 5.0]})
        result = df.select(MathOperators.fill_null("a", "forward"))
        assert result["a"].to_list() == [1.0, 1.0, 3.0, 4.0, 5.0]

    def test_fill_null_backward(self, sample_df):
        df = pl.DataFrame({"a": [1.0, 2.0, None, 4.0, 5.0]})
        result = df.select(MathOperators.fill_null("a", "backward"))
        assert result["a"].to_list() == [1.0, 2.0, 4.0, 4.0, 5.0]

    def test_fill_zero(self, sample_df):
        df = pl.DataFrame({"a": [1.0, None, 3.0, None, 5.0]})
        result = df.select(MathOperators.fill_zero("a"))
        assert result["a"].to_list() == [1.0, 0.0, 3.0, 0.0, 5.0]


class TestMathOperatorsTrigonometry:
    def test_sin(self):
        df = pl.DataFrame({"a": [0.0, np.pi / 2, np.pi]})
        result = df.select(MathOperators.sin("a"))
        assert result["a"].to_list()[0] == pytest.approx(0.0)
        assert result["a"].to_list()[1] == pytest.approx(1.0)
        assert result["a"].to_list()[2] == pytest.approx(0.0, abs=1e-10)

    def test_cos(self):
        df = pl.DataFrame({"a": [0.0, np.pi / 2, np.pi]})
        result = df.select(MathOperators.cos("a"))
        assert result["a"].to_list()[0] == pytest.approx(1.0)
        assert result["a"].to_list()[1] == pytest.approx(0.0)
        assert result["a"].to_list()[2] == pytest.approx(-1.0)

    def test_tan(self):
        df = pl.DataFrame({"a": [0.0, np.pi / 4, np.pi]})
        result = df.select(MathOperators.tan("a"))
        assert result["a"].to_list()[0] == pytest.approx(0.0)
        assert result["a"].to_list()[1] == pytest.approx(1.0)
        assert result["a"].to_list()[2] == pytest.approx(0.0, abs=1e-10)

    def test_arcsin(self):
        df = pl.DataFrame({"a": [0.0, 1.0, -1.0]})
        result = df.select(MathOperators.arcsin("a"))
        assert result["a"].to_list()[0] == pytest.approx(0.0)
        assert result["a"].to_list()[1] == pytest.approx(np.pi / 2)
        assert result["a"].to_list()[2] == pytest.approx(-np.pi / 2)

    def test_arccos(self):
        df = pl.DataFrame({"a": [1.0, 0.0, -1.0]})
        result = df.select(MathOperators.arccos("a"))
        assert result["a"].to_list()[0] == pytest.approx(0.0)
        assert result["a"].to_list()[1] == pytest.approx(np.pi / 2)
        assert result["a"].to_list()[2] == pytest.approx(np.pi)

    def test_arctan(self):
        df = pl.DataFrame({"a": [0.0, 1.0, -1.0]})
        result = df.select(MathOperators.arctan("a"))
        assert result["a"].to_list()[0] == pytest.approx(0.0)
        assert result["a"].to_list()[1] == pytest.approx(np.pi / 4)
        assert result["a"].to_list()[2] == pytest.approx(-np.pi / 4)
