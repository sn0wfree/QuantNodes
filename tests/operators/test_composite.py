# coding=utf-8
"""QuantNodes.operators.composite 单元测试"""
import polars as pl
import pytest

from QuantNodes.operators.composite import CompositeOperators


@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "f2": [5.0, 4.0, 3.0, 2.0, 1.0],
        "f3": [2.0, 2.0, 2.0, 2.0, 2.0],
    })


class TestCompositeWeightedSum:
    def test_weighted_sum_two_factors(self, sample_df):
        result = sample_df.select(
            CompositeOperators.weighted_sum(["f1", "f2"], [0.6, 0.4])
        )
        expected = [1.0 * 0.6 + 5.0 * 0.4, 2.0 * 0.6 + 4.0 * 0.4, 3.0 * 0.6 + 3.0 * 0.4,
                    4.0 * 0.6 + 2.0 * 0.4, 5.0 * 0.6 + 1.0 * 0.4]
        assert result["f1"].to_list() == pytest.approx(expected)

    def test_weighted_sum_three_factors(self, sample_df):
        result = sample_df.select(
            CompositeOperators.weighted_sum(["f1", "f2", "f3"], [0.5, 0.3, 0.2])
        )
        expected = [1.0 * 0.5 + 5.0 * 0.3 + 2.0 * 0.2, 2.0 * 0.5 + 4.0 * 0.3 + 2.0 * 0.2,
                    3.0 * 0.5 + 3.0 * 0.3 + 2.0 * 0.2, 4.0 * 0.5 + 2.0 * 0.3 + 2.0 * 0.2,
                    5.0 * 0.5 + 1.0 * 0.3 + 2.0 * 0.2]
        assert result["f1"].to_list() == pytest.approx(expected)

    def test_weighted_sum_with_expr(self, sample_df):
        result = sample_df.select(
            CompositeOperators.weighted_sum([pl.col("f1"), pl.col("f2")], [0.5, 0.5])
        )
        expected = [3.0, 3.0, 3.0, 3.0, 3.0]
        assert result["f1"].to_list() == pytest.approx(expected)


class TestCompositeWeightedAvg:
    def test_weighted_avg_equal_weights(self, sample_df):
        result = sample_df.select(
            CompositeOperators.weighted_avg(["f1", "f2"])
        )
        expected = [(1.0 + 5.0) / 2, (2.0 + 4.0) / 2, (3.0 + 3.0) / 2,
                    (4.0 + 2.0) / 2, (5.0 + 1.0) / 2]
        assert result["f1"].to_list() == pytest.approx(expected)

    def test_weighted_avg_custom_weights(self, sample_df):
        result = sample_df.select(
            CompositeOperators.weighted_avg(["f1", "f2"], [0.7, 0.3])
        )
        expected = [1.0 * 0.7 + 5.0 * 0.3, 2.0 * 0.7 + 4.0 * 0.3, 3.0 * 0.7 + 3.0 * 0.3,
                    4.0 * 0.7 + 2.0 * 0.3, 5.0 * 0.7 + 1.0 * 0.3]
        assert result["f1"].to_list() == pytest.approx(expected)

    def test_weighted_avg_auto_normalize(self, sample_df):
        result = sample_df.select(
            CompositeOperators.weighted_avg(["f1", "f2"], [70, 30])
        )
        expected = [1.0 * 0.7 + 5.0 * 0.3, 2.0 * 0.7 + 4.0 * 0.3, 3.0 * 0.7 + 3.0 * 0.3,
                    4.0 * 0.7 + 2.0 * 0.3, 5.0 * 0.7 + 1.0 * 0.3]
        assert result["f1"].to_list() == pytest.approx(expected)


class TestCompositeMaxMin:
    def test_max(self, sample_df):
        result = sample_df.select(
            CompositeOperators.max(["f1", "f2", "f3"])
        )
        assert result["f1"].to_list() == [5.0, 4.0, 3.0, 4.0, 5.0]

    def test_min(self, sample_df):
        result = sample_df.select(
            CompositeOperators.min(["f1", "f2", "f3"])
        )
        assert result["f1"].to_list() == [1.0, 2.0, 2.0, 2.0, 1.0]

    def test_abs_max(self, sample_df):
        df = pl.DataFrame({
            "a": [-5.0, 3.0, -2.0, 4.0, -1.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        result = df.select(
            CompositeOperators.abs_max(["a", "b"])
        )
        assert result["a"].to_list() == [5.0, 3.0, 3.0, 4.0, 5.0]


class TestCompositeBlend:
    def test_blend_default_alpha(self, sample_df):
        result = sample_df.select(
            CompositeOperators.blend("f1", "f2")
        )
        expected = [1.0 * 0.5 + 5.0 * 0.5, 2.0 * 0.5 + 4.0 * 0.5, 3.0 * 0.5 + 3.0 * 0.5,
                    4.0 * 0.5 + 2.0 * 0.5, 5.0 * 0.5 + 1.0 * 0.5]
        assert result["f1"].to_list() == pytest.approx(expected)

    def test_blend_custom_alpha(self, sample_df):
        result = sample_df.select(
            CompositeOperators.blend("f1", "f2", alpha=0.8)
        )
        expected = [1.0 * 0.8 + 5.0 * 0.2, 2.0 * 0.8 + 4.0 * 0.2, 3.0 * 0.8 + 3.0 * 0.2,
                    4.0 * 0.8 + 2.0 * 0.2, 5.0 * 0.8 + 1.0 * 0.2]
        assert result["f1"].to_list() == pytest.approx(expected)

    def test_blend_with_expr(self, sample_df):
        result = sample_df.select(
            CompositeOperators.blend(pl.col("f1"), pl.col("f2"), alpha=0.3)
        )
        expected = [1.0 * 0.3 + 5.0 * 0.7, 2.0 * 0.3 + 4.0 * 0.7, 3.0 * 0.3 + 3.0 * 0.7,
                    4.0 * 0.3 + 2.0 * 0.7, 5.0 * 0.3 + 1.0 * 0.7]
        assert result["f1"].to_list() == pytest.approx(expected)


class TestCompositeCombine:
    def test_combine_sum(self, sample_df):
        result = sample_df.select(
            CompositeOperators.combine(["f1", "f2", "f3"], method="sum")
        )
        expected = [1.0 + 5.0 + 2.0, 2.0 + 4.0 + 2.0, 3.0 + 3.0 + 2.0,
                    4.0 + 2.0 + 2.0, 5.0 + 1.0 + 2.0]
        assert result.to_series(0).to_list() == pytest.approx(expected)

    def test_combine_avg(self, sample_df):
        result = sample_df.select(
            CompositeOperators.combine(["f1", "f2", "f3"], method="avg")
        )
        expected = [(1.0 + 5.0 + 2.0) / 3, (2.0 + 4.0 + 2.0) / 3, (3.0 + 3.0 + 2.0) / 3,
                    (4.0 + 2.0 + 2.0) / 3, (5.0 + 1.0 + 2.0) / 3]
        assert result.to_series(0).to_list() == pytest.approx(expected)

    def test_combine_mul(self, sample_df):
        result = sample_df.select(
            CompositeOperators.combine(["f1", "f2", "f3"], method="mul")
        )
        expected = [1.0 * 5.0 * 2.0, 2.0 * 4.0 * 2.0, 3.0 * 3.0 * 2.0,
                    4.0 * 2.0 * 2.0, 5.0 * 1.0 * 2.0]
        assert result.to_series(0).to_list() == pytest.approx(expected)

    def test_combine_max(self, sample_df):
        result = sample_df.select(
            CompositeOperators.combine(["f1", "f2", "f3"], method="max")
        )
        assert result.to_series(0).to_list() == [5.0, 4.0, 3.0, 4.0, 5.0]


class TestCompositeSelectTop:
    def test_select_top_ascending(self, sample_df):
        result = sample_df.select(
            CompositeOperators.select_top("f1", n=3, ascending=True)
        )
        ranks = result["f1"].to_list()
        assert ranks == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_select_top_descending(self, sample_df):
        result = sample_df.select(
            CompositeOperators.select_top("f1", n=3, ascending=False)
        )
        ranks = result["f1"].to_list()
        assert ranks == [5.0, 4.0, 3.0, 2.0, 1.0]


class TestCompositeFilter:
    def test_filter_positive(self, sample_df):
        df = pl.DataFrame({"a": [-1.0, 2.0, -3.0, 4.0, -5.0]})
        result = df.select(CompositeOperators.filter_positive("a"))
        assert result["a"].to_list() == [-1.0, 0.0, -3.0, 0.0, -5.0]

    def test_filter_negative(self, sample_df):
        df = pl.DataFrame({"a": [-1.0, 2.0, -3.0, 4.0, -5.0]})
        result = df.select(CompositeOperators.filter_negative("a"))
        assert result["a"].to_list() == [0.0, 2.0, 0.0, 4.0, 0.0]

    def test_abs_filter(self, sample_df):
        df = pl.DataFrame({"a": [-1.0, 2.0, -3.0, 4.0, -5.0]})
        result = df.select(CompositeOperators.abs_filter("a", threshold=2.5))
        assert result["a"].to_list() == [0.0, 0.0, -3.0, 4.0, -5.0]


class TestCompositeRankSort:
    def test_rank_sort_equal_weights(self, sample_df):
        result = sample_df.select(
            CompositeOperators.rank_sort(["f1", "f2"])
        )
        assert len(result["f1"]) == 5

    def test_rank_sort_with_weights(self, sample_df):
        result = sample_df.select(
            CompositeOperators.rank_sort(["f1", "f2"], weights=[0.6, 0.4])
        )
        assert len(result["f1"]) == 5
