# coding=utf-8
"""QuantNodes.operators.section 单元测试"""
import numpy as np
import polars as pl
import pytest

from QuantNodes.operators.section import SectionOperators


@pytest.fixture
def sample_df():
    return pl.DataFrame({
        "factor": [1.0, 2.0, 3.0, 4.0, 5.0],
        "group": ["A", "A", "B", "B", "B"],
        "target": [0.1, 0.2, 0.3, 0.4, 0.5],
    })


class TestSectionRank:
    def test_rank_dense(self, sample_df):
        result = sample_df.select(SectionOperators.rank("factor", method="dense"))
        values = result["factor"].to_list()
        assert values == [0.0, 0.25, 0.5, 0.75, 1.0]

    def test_rank_ordinal(self, sample_df):
        result = sample_df.select(SectionOperators.rank("factor", method="ordinal"))
        values = result["factor"].to_list()
        assert values == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_rank_average(self, sample_df):
        result = sample_df.select(SectionOperators.rank("factor", method="average"))
        values = result["factor"].to_list()
        assert values == [0.0, 0.25, 0.5, 0.75, 1.0]


class TestSectionZscore:
    def test_zscore(self, sample_df):
        result = sample_df.select(SectionOperators.zscore("factor"))
        values = result["factor"].to_list()
        mean = 3.0
        std = np.std([1.0, 2.0, 3.0, 4.0, 5.0], ddof=1)
        expected = [(v - mean) / std for v in [1.0, 2.0, 3.0, 4.0, 5.0]]
        assert values == pytest.approx(expected)

    def test_zscore_with_expr(self, sample_df):
        result = sample_df.select(SectionOperators.zscore(pl.col("factor")))
        values = result["factor"].to_list()
        assert len(values) == 5


class TestSectionWinsorize:
    def test_winsorize_default(self, sample_df):
        result = sample_df.select(SectionOperators.winsorize("factor"))
        values = result["factor"].to_list()
        assert values == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_winsorize_with_bounds(self, sample_df):
        result = sample_df.select(SectionOperators.winsorize("factor", lower=0.2, upper=0.2))
        values = result["factor"].to_list()
        assert values[0] >= 2.0
        assert values[-1] <= 4.0


class TestSectionNeutralize:
    def test_neutralize_by_group(self, sample_df):
        result = sample_df.select(SectionOperators.neutralize("factor", "group"))
        values = result["factor"].to_list()
        assert values[0] == -0.5
        assert values[2] == -1.0

    def test_neutralize_market(self, sample_df):
        result = sample_df.select(SectionOperators.neutralize_market("factor"))
        values = result["factor"].to_list()
        mean_val = sum(values) / len(values)
        assert abs(mean_val) < 1e-10


class TestSectionScale:
    def test_scale_zscore(self, sample_df):
        result = sample_df.select(SectionOperators.scale("factor", method="zscore"))
        values = result["factor"].to_list()
        assert abs(sum(values)) < 1e-10

    def test_scale_minmax(self, sample_df):
        result = sample_df.select(SectionOperators.scale("factor", method="minmax"))
        values = result["factor"].to_list()
        assert values[0] == 0.0
        assert values[-1] == pytest.approx(1.0)

    def test_scale_abs(self, sample_df):
        result = sample_df.select(SectionOperators.scale("factor", method="abs"))
        values = result["factor"].to_list()
        assert values[-1] == 1.0


class TestSectionPercentile:
    def test_percentile(self, sample_df):
        result = sample_df.select(SectionOperators.percentile("factor"))
        values = result["factor"].to_list()
        assert values[0] == pytest.approx(0.2)
        assert values[-1] == pytest.approx(1.0)


class TestSectionIC:
    def test_rank_ic(self, sample_df):
        result = sample_df.select(SectionOperators.rank_ic("factor", "target"))
        values = result["factor"].to_list()
        assert len(values) == 1
        assert -1 <= values[0] <= 1

    def test_ic(self, sample_df):
        result = sample_df.select(SectionOperators.ic("factor", "target"))
        values = result["factor"].to_list()
        assert len(values) == 1
        assert -1 <= values[0] <= 1


class TestSectionGroupNorm:
    def test_group_norm_zscore(self, sample_df):
        result = sample_df.select(SectionOperators.group_norm("factor", "group", method="zscore"))
        values = result["factor"].to_list()
        assert len(values) == 5

    def test_group_norm_rank(self, sample_df):
        result = sample_df.select(SectionOperators.group_norm("factor", "group", method="rank"))
        values = result["factor"].to_list()
        assert len(values) == 5


class TestSectionGroupWinsorize:
    def test_group_winsorize(self, sample_df):
        result = sample_df.select(SectionOperators.group_winsorize("factor", "group"))
        values = result["factor"].to_list()
        assert len(values) == 5


class TestSectionStringInput:
    def test_rank_str(self, sample_df):
        result = sample_df.select(SectionOperators.rank("factor"))
        assert len(result) == 5

    def test_zscore_str(self, sample_df):
        result = sample_df.select(SectionOperators.zscore("factor"))
        assert len(result) == 5

    def test_winsorize_str(self, sample_df):
        result = sample_df.select(SectionOperators.winsorize("factor"))
        assert len(result) == 5

    def test_neutralize_str(self, sample_df):
        result = sample_df.select(SectionOperators.neutralize("factor", "group"))
        assert len(result) == 5

    def test_scale_str(self, sample_df):
        result = sample_df.select(SectionOperators.scale("factor"))
        assert len(result) == 5

    def test_percentile_str(self, sample_df):
        result = sample_df.select(SectionOperators.percentile("factor"))
        assert len(result) == 5
