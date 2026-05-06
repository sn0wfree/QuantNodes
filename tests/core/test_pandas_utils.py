# coding=utf-8
"""QuantNodes.core.pandas_utils 单元测试"""
import numpy as np
import pandas as pd
import pytest

from QuantNodes.core.pandas_utils import (
    panel_to_dataframe,
    dataframe_to_panel,
    align_dataframes,
    forward_fill_panel,
    fillna_by_value,
    winsorize_series,
    standardize_zscore,
    standardize_rank,
    cross_section_zscore,
    cross_section_rank,
    shift_df,
    resample_panel,
    melt_panel,
    pivot_long,
    pivot_wide,
)


class TestPanelToDataframe:
    def test_list_input(self):
        df1 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
        result = panel_to_dataframe([df1, df2])
        assert isinstance(result, pd.DataFrame)

    def test_empty_dict(self):
        result = panel_to_dataframe({})
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_empty_list(self):
        result = panel_to_dataframe([])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            panel_to_dataframe("invalid")


class TestDataframeToPanel:
    def test_no_multiindex(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = dataframe_to_panel(df)
        assert "_default" in result

    def test_multiindex(self):
        columns = pd.MultiIndex.from_tuples([("f1", "id1"), ("f1", "id2"), ("f2", "id1")])
        df = pd.DataFrame(np.random.randn(3, 3), columns=columns)
        result = dataframe_to_panel(df, level=0)
        assert "f1" in result
        assert "f2" in result


class TestAlignDataframes:
    def test_outer(self):
        df1 = pd.DataFrame({"a": [1]}, index=[0, 1])
        df2 = pd.DataFrame({"a": [2]}, index=[1, 2])
        result = align_dataframes([df1, df2], how="outer")
        assert len(result[0].index) == 3

    def test_inner(self):
        df1 = pd.DataFrame({"a": [1]}, index=[0, 1])
        df2 = pd.DataFrame({"a": [2]}, index=[1, 2])
        result = align_dataframes([df1, df2], how="inner")
        assert len(result[0].index) == 1

    def test_left(self):
        df1 = pd.DataFrame({"a": [1]}, index=[0, 1])
        df2 = pd.DataFrame({"a": [2]}, index=[1, 2])
        result = align_dataframes([df1, df2], how="left")
        assert list(result[0].index) == [0, 1]

    def test_right(self):
        df1 = pd.DataFrame({"a": [1]}, index=[0, 1])
        df2 = pd.DataFrame({"a": [2]}, index=[1, 2])
        result = align_dataframes([df1, df2], how="right")
        assert list(result[0].index) == [1, 2]

    def test_invalid_how(self):
        df1 = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError):
            align_dataframes([df1, df1], how="invalid")

    def test_empty(self):
        assert align_dataframes([]) == []

    def test_single(self):
        df = pd.DataFrame({"a": [1]})
        assert align_dataframes([df]) == [df]

    def test_axis_1(self):
        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"a": [3], "c": [4]})
        result = align_dataframes([df1, df2], axis=1, how="outer")
        assert "b" in result[0].columns
        assert "c" in result[0].columns


class TestForwardFillPanel:
    def test_basic(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        panel = {"f": df}
        result = forward_fill_panel(panel)
        assert result["f"]["a"].iloc[1] == 1.0

    def test_with_limit(self):
        df = pd.DataFrame({"a": [np.nan, np.nan, 3.0]})
        panel = {"f": df}
        result = forward_fill_panel(panel, limit=1)
        assert pd.isna(result["f"]["a"].iloc[0])


class TestFillnaByValue:
    def test_scalar(self):
        df = pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]})
        result = fillna_by_value(df, 0)
        assert result["a"].iloc[1] == 0
        assert result["b"].iloc[0] == 0

    def test_dict(self):
        df = pd.DataFrame({"a": [np.nan], "b": [np.nan]})
        result = fillna_by_value(df, {"a": -999, "b": 0})
        assert result["a"].iloc[0] == -999
        assert result["b"].iloc[0] == 0

    def test_with_condition(self):
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        cond = pd.DataFrame({"a": [False, True, False]})
        result = fillna_by_value(df, 0, condition=cond)
        assert result["a"].iloc[1] == 0
        assert result["a"].iloc[0] == 1.0


class TestWinsorizeSeries:
    def test_basic(self):
        s = pd.Series([1, 2, 3, 100])
        result = winsorize_series(s, lower=0.1, upper=0.1)
        assert result.max() < 100

    def test_empty(self):
        s = pd.Series([], dtype=float)
        result = winsorize_series(s)
        assert len(result) == 0

    def test_normal_data(self):
        s = pd.Series(np.random.randn(100))
        result = winsorize_series(s, lower=0.05, upper=0.05)
        assert len(result) == 100


class TestStandardizeZscore:
    def test_basic(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = standardize_zscore(df)
        assert abs(result["a"].mean()) < 1e-10

    def test_selective_columns(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        result = standardize_zscore(df, columns=["a"])
        assert abs(result["a"].mean()) < 1e-10
        assert result["b"].iloc[0] == 10.0

    def test_zero_std(self):
        df = pd.DataFrame({"a": [5.0, 5.0, 5.0]})
        result = standardize_zscore(df)
        assert (result["a"] == 0).all()


class TestStandardizeRank:
    def test_basic(self):
        s = pd.Series([10, 20, 30])
        result = standardize_rank(s, pct=True)
        assert result.min() > 0
        assert result.max() <= 1

    def test_empty(self):
        s = pd.Series([], dtype=float)
        result = standardize_rank(s)
        assert len(result) == 0

    def test_descending(self):
        s = pd.Series([1, 2, 3])
        result = standardize_rank(s, ascending=False)
        assert result.iloc[0] > result.iloc[-1]


class TestCrossSectionZscore:
    def test_basic(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = cross_section_zscore(df)
        assert isinstance(result, pd.DataFrame)

    def test_with_groupby(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [10.0, 20.0, 30.0, 40.0],
            "g": [1, 1, 2, 2],
        })
        result = cross_section_zscore(df[["a", "b"]], groupby=df["g"])
        assert isinstance(result, pd.DataFrame)


class TestCrossSectionRank:
    def test_basic(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        result = cross_section_rank(df)
        assert isinstance(result, pd.DataFrame)


class TestShiftDf:
    def test_basic(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = shift_df(df, periods=1)
        assert pd.isna(result["a"].iloc[0])
        assert result["a"].iloc[1] == 1

    def test_negative(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = shift_df(df, periods=-1)
        assert result["a"].iloc[0] == 2

    def test_freq_without_datetime(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            shift_df(df, periods=1, freq="D")


class TestResamplePanel:
    def test_basic(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({"a": range(10)}, index=dates)
        panel = {"f": df}
        result = resample_panel(panel, rule="W")
        assert len(result["f"]) < 10

    def test_non_datetime(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        panel = {"f": df}
        result = resample_panel(panel, rule="M")
        assert len(result["f"]) == 3


class TestMeltPanel:
    def test_basic(self):
        df1 = pd.DataFrame({"x": [1, 2]})
        df2 = pd.DataFrame({"x": [3, 4]})
        panel = {"f1": df1, "f2": df2}
        result = melt_panel(panel)
        assert len(result) == 4
        assert "item" in result.columns

    def test_empty(self):
        result = melt_panel({})
        assert isinstance(result, pd.DataFrame)


class TestPivotLong:
    def test_basic(self):
        df = pd.DataFrame({"id": [1, 2], "a": [10, 20], "b": [30, 40]})
        result = pivot_long(df, id_vars=["id"], value_vars=["a", "b"])
        assert len(result) == 4


class TestPivotWide:
    def test_basic(self):
        df = pd.DataFrame({"id": [1, 1, 2], "var": ["a", "b", "a"], "val": [10, 20, 30]})
        result = pivot_wide(df, index="id", columns="var", values="val")
        assert "a" in result.columns
