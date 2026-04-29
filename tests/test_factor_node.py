# -*- coding: utf-8 -*-
"""FactorNode unit tests"""
import pytest
import pandas as pd
import numpy as np

from QuantNodes.core.node import BaseNode, NodeExecutionError
from QuantNodes.factor_node.factor_nodes import (
    FactorNode, FactorPipeline,
    PointFactorNode, ArithmeticFactorNode,
    TimeFactorNode, ExpandingFactorNode,
    CrossSectionFactorNode, GroupRankFactorNode,
    PanelFactorNode, DelayFactorNode, DeltaFactorNode,
    _zscore_fn, _demean_fn, _mad_fn,
    _add, _sub, _mul, _div,
    _groupby_transform,
)


@pytest.fixture
def sample_df():
    """测试用 DataFrame"""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'date': ['2020-01-01', '2020-01-01', '2020-01-02', '2020-01-02', '2020-01-03'],
        'close': [100.0, 105.0, 102.0, 108.0, 110.0],
        'open': [98.0, 103.0, 101.0, 106.0, 109.0],
        'volume': [1000.0, 1500.0, 1200.0, 1800.0, 2000.0],
        'return': [0.02, 0.019, 0.01, 0.019, 0.018],
    })


@pytest.fixture
def timeseries_df():
    """时间序列测试 DataFrame"""
    dates = pd.date_range('2020-01-01', periods=10)
    return pd.DataFrame({
        'date': dates,
        'value': [float(v) for v in range(10, 20)],
    })


@pytest.fixture
def panel_df():
    """面板数据测试 DataFrame（含 dt 列）"""
    return pd.DataFrame({
        'dt': ['2020-01-01'] * 4 + ['2020-01-02'] * 4,
        'industry': ['A', 'A', 'B', 'B'] * 2,
        'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    })


# ===========================================================================
# 共用算子函数测试
# ===========================================================================

class TestSharedOperators:
    """测试共用算子函数"""

    def test_zscore_fn(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _zscore_fn(s)
        assert result.mean() == pytest.approx(0.0, abs=1e-10)
        assert result.std() == pytest.approx(1.0, abs=1e-10)

    def test_zscore_fn_zero_std(self):
        s = pd.Series([3.0, 3.0, 3.0])
        result = _zscore_fn(s)
        expected = s - s.mean()
        pd.testing.assert_series_equal(result, expected)

    def test_demean_fn(self):
        s = pd.Series([10.0, 20.0, 30.0])
        result = _demean_fn(s)
        assert result.mean() == pytest.approx(0.0)

    def test_mad_fn(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
        result = _mad_fn(s)
        assert result > 0

    def test_arithmetic_ops(self):
        a = pd.Series([10.0, 20.0])
        b = pd.Series([2.0, 4.0])
        assert (_add(a, b) == pd.Series([12.0, 24.0])).all()
        assert (_sub(a, b) == pd.Series([8.0, 16.0])).all()
        assert (_mul(a, b) == pd.Series([20.0, 80.0])).all()
        assert (_div(a, b) == pd.Series([5.0, 5.0])).all()

    def test_groupby_transform(self, panel_df):
        result = _groupby_transform(panel_df, 'value', 'dt', _demean_fn)
        assert len(result) == len(panel_df)

    def test_groupby_transform_with_groupby(self, panel_df):
        result = _groupby_transform(panel_df, 'value', 'dt', _demean_fn, groupby='industry')
        assert len(result) == len(panel_df)


# ===========================================================================
# FactorNode 基类辅助方法测试
# ===========================================================================

class TestFactorNodeHelpers:
    """测试基类辅助方法"""

    def test_validate_input_none(self):
        f = PointFactorNode(expression="a", result_name="a")
        with pytest.raises(ValueError, match="input_data is required"):
            f._validate_input(None)

    def test_validate_input_type(self):
        f = PointFactorNode(expression="a", result_name="a")
        with pytest.raises(TypeError, match="Expected DataFrame"):
            f._validate_input("not a dataframe")

    def test_validate_input_copies(self, sample_df):
        f = PointFactorNode(expression="close", result_name="close")
        result = f._validate_input(sample_df)
        result['new_col'] = 999
        assert 'new_col' not in sample_df.columns

    def test_finalize(self):
        f = PointFactorNode(expression="a", result_name="a")
        s = pd.Series([1.0, 2.0, 3.0])
        result = f._finalize(s)
        assert isinstance(result, pd.DataFrame)
        assert result.columns[0] == f.name

    def test_get_dt_key_with_dt(self):
        df = pd.DataFrame({'dt': [1, 2, 3], 'val': [4, 5, 6]})
        assert FactorNode._get_dt_key(df) == 'dt'

    def test_get_dt_key_without_dt(self):
        df = pd.DataFrame({'val': [4, 5, 6]}, index=[1, 2, 3])
        assert FactorNode._get_dt_key(df) is df.index

    def test_extract_first_col_dataframe(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        result = FactorNode._extract_first_col(df)
        pd.testing.assert_series_equal(result, df['a'])

    def test_extract_first_col_series(self):
        s = pd.Series([1, 2, 3])
        result = FactorNode._extract_first_col(s)
        pd.testing.assert_series_equal(result, s)


# ===========================================================================
# PointFactorNode 测试
# ===========================================================================

class TestPointFactorNode:

    def test_point_factor_expression(self, sample_df):
        factor = PointFactorNode(expression="close / open - 1", result_name="ret")
        result = factor.execute(sample_df)
        assert "ret" in result.columns
        assert len(result) == len(sample_df)

    def test_point_factor_func(self, sample_df):
        factor = PointFactorNode(
            func=lambda row: row['close'] / row['open'] - 1,
            result_name="ret"
        )
        result = factor.execute(sample_df)
        assert "ret" in result.columns

    def test_point_factor_no_input(self):
        factor = PointFactorNode(expression="close / open - 1")
        with pytest.raises(NodeExecutionError, match="input_data is required"):
            factor.execute(None)

    def test_point_factor_no_expression_no_func(self, sample_df):
        factor = PointFactorNode(result_name="x")
        with pytest.raises(NodeExecutionError, match="Either expression or func"):
            factor.execute(sample_df)


# ===========================================================================
# ArithmeticFactorNode 测试
# ===========================================================================

class TestArithmeticFactorNode:

    def test_arithmetic_add(self, sample_df):
        f1 = PointFactorNode(expression="close", result_name="close")
        f2 = PointFactorNode(expression="open", result_name="open")
        factor = ArithmeticFactorNode(factors=[f1, f2], operator="add")
        result = factor.execute(sample_df)
        expected = sample_df["close"] + sample_df["open"]
        pd.testing.assert_series_equal(
            result.iloc[:, 0].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_arithmetic_sub(self, sample_df):
        f1 = PointFactorNode(expression="close", result_name="close")
        f2 = PointFactorNode(expression="open", result_name="open")
        factor = ArithmeticFactorNode(factors=[f1, f2], operator="sub")
        result = factor.execute(sample_df)
        expected = sample_df["close"] - sample_df["open"]
        pd.testing.assert_series_equal(
            result.iloc[:, 0].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_arithmetic_mul(self, sample_df):
        f1 = PointFactorNode(expression="close", result_name="close")
        f2 = PointFactorNode(expression="open", result_name="open")
        factor = ArithmeticFactorNode(factors=[f1, f2], operator="mul")
        result = factor.execute(sample_df)
        expected = sample_df["close"] * sample_df["open"]
        pd.testing.assert_series_equal(
            result.iloc[:, 0].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_arithmetic_div(self, sample_df):
        f1 = PointFactorNode(expression="close", result_name="close")
        f2 = PointFactorNode(expression="open", result_name="open")
        factor = ArithmeticFactorNode(factors=[f1, f2], operator="div")
        result = factor.execute(sample_df)
        expected = sample_df["close"] / sample_df["open"]
        pd.testing.assert_series_equal(
            result.iloc[:, 0].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            rtol=1e-5,
        )

    def test_arithmetic_invalid_op(self, sample_df):
        f1 = PointFactorNode(expression="close", result_name="close")
        f2 = PointFactorNode(expression="open", result_name="open")
        factor = ArithmeticFactorNode(factors=[f1, f2], operator="pow")
        with pytest.raises(NodeExecutionError, match="Unknown operator"):
            factor.execute(sample_df)

    def test_arithmetic_less_than_2_factors(self, sample_df):
        f1 = PointFactorNode(expression="close", result_name="close")
        factor = ArithmeticFactorNode(factors=[f1], operator="add")
        with pytest.raises(NodeExecutionError, match="At least 2 factors"):
            factor.execute(sample_df)


# ===========================================================================
# TimeFactorNode 测试
# ===========================================================================

class TestTimeFactorNode:

    def test_time_mean(self, timeseries_df):
        factor = TimeFactorNode(window=3, operation="mean", column="value")
        result = factor.execute(timeseries_df)
        assert len(result) == len(timeseries_df)
        assert result.iloc[2, 0] == pytest.approx(11.0)

    def test_time_std(self, timeseries_df):
        factor = TimeFactorNode(window=3, operation="std", column="value")
        result = factor.execute(timeseries_df)
        assert len(result) == len(timeseries_df)
        assert result.iloc[2, 0] > 0

    def test_time_sum(self, timeseries_df):
        factor = TimeFactorNode(window=3, operation="sum", column="value")
        result = factor.execute(timeseries_df)
        assert len(result) == len(timeseries_df)
        assert result.iloc[2, 0] == pytest.approx(33.0)

    def test_time_min(self, timeseries_df):
        factor = TimeFactorNode(window=3, operation="min", column="value")
        result = factor.execute(timeseries_df)
        assert len(result) == len(timeseries_df)
        assert result.iloc[2, 0] == pytest.approx(10.0)

    def test_time_max(self, timeseries_df):
        factor = TimeFactorNode(window=3, operation="max", column="value")
        result = factor.execute(timeseries_df)
        assert len(result) == len(timeseries_df)
        assert result.iloc[2, 0] == pytest.approx(12.0)

    def test_time_corr(self, timeseries_df):
        timeseries_df['value2'] = timeseries_df['value'] * 2
        factor = TimeFactorNode(window=3, operation="corr", columns=["value", "value2"])
        result = factor.execute(timeseries_df)
        assert len(result) == len(timeseries_df)

    def test_time_cov(self, timeseries_df):
        timeseries_df['value2'] = timeseries_df['value'] * 2
        factor = TimeFactorNode(window=3, operation="cov", columns=["value", "value2"])
        result = factor.execute(timeseries_df)
        assert len(result) == len(timeseries_df)

    def test_time_no_column(self, timeseries_df):
        factor = TimeFactorNode(window=3, operation="mean")
        with pytest.raises(NodeExecutionError, match="column is required"):
            factor.execute(timeseries_df)

    def test_time_corr_no_columns(self, timeseries_df):
        factor = TimeFactorNode(window=3, operation="corr")
        with pytest.raises(NodeExecutionError, match="columns must be a list of 2"):
            factor.execute(timeseries_df)

    def test_time_unknown_op(self, timeseries_df):
        factor = TimeFactorNode(window=3, operation="median", column="value")
        with pytest.raises(NodeExecutionError, match="Unknown operation"):
            factor.execute(timeseries_df)


# ===========================================================================
# ExpandingFactorNode 测试
# ===========================================================================

class TestExpandingFactorNode:

    def test_expanding_mean(self, timeseries_df):
        factor = ExpandingFactorNode(operation="mean", column="value")
        result = factor.execute(timeseries_df)
        assert len(result) == len(timeseries_df)
        assert result.iloc[0, 0] == pytest.approx(10.0)
        assert result.iloc[4, 0] == pytest.approx(12.0)

    def test_expanding_std(self, timeseries_df):
        factor = ExpandingFactorNode(operation="std", column="value")
        result = factor.execute(timeseries_df)
        assert len(result) == len(timeseries_df)

    def test_expanding_sum(self, timeseries_df):
        factor = ExpandingFactorNode(operation="sum", column="value")
        result = factor.execute(timeseries_df)
        assert len(result) == len(timeseries_df)
        assert result.iloc[2, 0] == pytest.approx(33.0)

    def test_expanding_min(self, timeseries_df):
        factor = ExpandingFactorNode(operation="min", column="value")
        result = factor.execute(timeseries_df)
        assert result.iloc[5, 0] == pytest.approx(10.0)

    def test_expanding_max(self, timeseries_df):
        factor = ExpandingFactorNode(operation="max", column="value")
        result = factor.execute(timeseries_df)
        assert result.iloc[5, 0] == pytest.approx(15.0)

    def test_expanding_no_column(self, timeseries_df):
        factor = ExpandingFactorNode(operation="mean")
        with pytest.raises(NodeExecutionError, match="column is required"):
            factor.execute(timeseries_df)

    def test_expanding_unknown_op(self, timeseries_df):
        factor = ExpandingFactorNode(operation="median", column="value")
        with pytest.raises(NodeExecutionError, match="Unknown operation"):
            factor.execute(timeseries_df)


# ===========================================================================
# CrossSectionFactorNode 测试
# ===========================================================================

class TestCrossSectionFactorNode:

    def test_cross_section_rank(self, sample_df):
        factor = CrossSectionFactorNode(operation="rank", column="return")
        result = factor.execute(sample_df)
        assert len(result) == len(sample_df)

    def test_cross_section_zscore(self, panel_df):
        factor = CrossSectionFactorNode(operation="zscore", column="value")
        result = factor.execute(panel_df)
        assert len(result) == len(panel_df)

    def test_cross_section_demean(self, panel_df):
        factor = CrossSectionFactorNode(operation="demean", column="value")
        result = factor.execute(panel_df)
        assert len(result) == len(panel_df)

    def test_cross_section_mad(self, panel_df):
        factor = CrossSectionFactorNode(operation="mad", column="value")
        result = factor.execute(panel_df)
        assert len(result) == len(panel_df)

    def test_cross_section_percentile(self, panel_df):
        factor = CrossSectionFactorNode(operation="percentile", column="value")
        result = factor.execute(panel_df)
        assert len(result) == len(panel_df)

    def test_cross_section_zscore_groupby(self, panel_df):
        factor = CrossSectionFactorNode(operation="zscore", column="value", groupby="industry")
        result = factor.execute(panel_df)
        assert len(result) == len(panel_df)

    def test_cross_section_demean_groupby(self, panel_df):
        factor = CrossSectionFactorNode(operation="demean", column="value", groupby="industry")
        result = factor.execute(panel_df)
        assert len(result) == len(panel_df)

    def test_cross_section_no_column(self, sample_df):
        factor = CrossSectionFactorNode(operation="rank")
        with pytest.raises(NodeExecutionError, match="column is required"):
            factor.execute(sample_df)

    def test_cross_section_unknown_op(self, sample_df):
        factor = CrossSectionFactorNode(operation="median", column="return")
        with pytest.raises(NodeExecutionError, match="Unknown operation"):
            factor.execute(sample_df)


# ===========================================================================
# GroupRankFactorNode 测试
# ===========================================================================

class TestGroupRankFactorNode:

    def test_group_rank(self, panel_df):
        factor = GroupRankFactorNode(column="value", groupby="industry")
        result = factor.execute(panel_df)
        assert len(result) == len(panel_df)

    def test_group_rank_ascending(self, panel_df):
        factor = GroupRankFactorNode(column="value", groupby="industry", ascending=True)
        result = factor.execute(panel_df)
        assert len(result) == len(panel_df)


# ===========================================================================
# PanelFactorNode 测试
# ===========================================================================

class TestPanelFactorNode:

    def test_panel_zscore(self, sample_df):
        factor = PanelFactorNode(operations=[("zscore", {"column": "return"})])
        result = factor.execute(sample_df)
        assert len(result) == len(sample_df)

    def test_panel_demean(self, panel_df):
        factor = PanelFactorNode(operations=[("demean", {"column": "value"})])
        result = factor.execute(panel_df)
        assert len(result) == len(panel_df)

    def test_panel_rank(self, panel_df):
        factor = PanelFactorNode(operations=[("rank", {"column": "value"})])
        result = factor.execute(panel_df)
        assert len(result) == len(panel_df)

    def test_panel_mul_combine(self, panel_df):
        factor = PanelFactorNode(
            operations=[
                ("zscore", {"column": "value"}),
                ("rank", {"column": "value"}),
            ],
            combine="mul",
        )
        result = factor.execute(panel_df)
        assert len(result) == len(panel_df)

    def test_panel_mean_combine(self, panel_df):
        factor = PanelFactorNode(
            operations=[
                ("zscore", {"column": "value"}),
                ("demean", {"column": "value"}),
            ],
            combine="mean",
        )
        result = factor.execute(panel_df)
        assert len(result) == len(panel_df)

    def test_panel_no_operations(self, sample_df):
        factor = PanelFactorNode(operations=[])
        with pytest.raises(NodeExecutionError, match="No operations specified"):
            factor.execute(sample_df)

    def test_panel_unknown_op(self, sample_df):
        factor = PanelFactorNode(operations=[("median", {"column": "return"})])
        with pytest.raises(NodeExecutionError, match="Unknown operation"):
            factor.execute(sample_df)

    def test_panel_unknown_combine(self, panel_df):
        factor = PanelFactorNode(
            operations=[("zscore", {"column": "value"})],
            combine="median",
        )
        with pytest.raises(NodeExecutionError, match="Unknown combine method"):
            factor.execute(panel_df)


# ===========================================================================
# DelayFactorNode 测试
# ===========================================================================

class TestDelayFactorNode:

    def test_delay(self, timeseries_df):
        base_factor = TimeFactorNode(window=3, operation="mean", column="value")
        factor = DelayFactorNode(base_factor=base_factor, periods=1)
        result = factor.execute(timeseries_df)
        assert len(result) == len(timeseries_df)

    def test_delay_no_input(self, timeseries_df):
        base_factor = TimeFactorNode(window=3, operation="mean", column="value")
        factor = DelayFactorNode(base_factor=base_factor, periods=1)
        with pytest.raises(NodeExecutionError, match="input_data is required"):
            factor.execute(None)


# ===========================================================================
# DeltaFactorNode 测试
# ===========================================================================

class TestDeltaFactorNode:

    def test_delta_diff(self, timeseries_df):
        base_factor = TimeFactorNode(window=3, operation="mean", column="value")
        factor = DeltaFactorNode(base_factor=base_factor, periods=1, mode="diff")
        result = factor.execute(timeseries_df)
        assert len(result) == len(timeseries_df)

    def test_delta_pct_change(self, timeseries_df):
        base_factor = TimeFactorNode(window=3, operation="mean", column="value")
        factor = DeltaFactorNode(base_factor=base_factor, periods=1, mode="pct_change")
        result = factor.execute(timeseries_df)
        assert len(result) == len(timeseries_df)

    def test_delta_unknown_mode(self, timeseries_df):
        base_factor = TimeFactorNode(window=3, operation="mean", column="value")
        factor = DeltaFactorNode(base_factor=base_factor, periods=1, mode="invalid")
        with pytest.raises(NodeExecutionError, match="Unknown mode"):
            factor.execute(timeseries_df)

    def test_delta_no_input(self, timeseries_df):
        base_factor = TimeFactorNode(window=3, operation="mean", column="value")
        factor = DeltaFactorNode(base_factor=base_factor, periods=1, mode="diff")
        with pytest.raises(NodeExecutionError, match="input_data is required"):
            factor.execute(None)


# ===========================================================================
# FactorPipeline 测试
# ===========================================================================

class TestFactorPipeline:

    def test_pipeline_execute(self, sample_df):
        f1 = PointFactorNode(expression="close / open - 1", result_name="ret")
        f2 = PointFactorNode(expression="volume / 1000", result_name="vol_adj")
        pipeline = FactorPipeline([f1, f2])
        results = pipeline.execute(sample_df)
        assert "PointFactorNode" in results
        assert "PointFactorNode_1" in results
        assert isinstance(results["PointFactorNode"], pd.DataFrame)

    def test_pipeline_rshift(self, sample_df):
        f1 = PointFactorNode(expression="close", result_name="close")
        f2 = PointFactorNode(expression="open", result_name="open")
        pipeline = f1 >> f2
        results = pipeline.execute(sample_df)
        assert len(results) == 2

    def test_pipeline_chain(self, sample_df):
        f1 = PointFactorNode(expression="close", result_name="close")
        f2 = PointFactorNode(expression="open", result_name="open")
        f3 = PointFactorNode(expression="volume", result_name="volume")
        pipeline = (f1 >> f2) >> f3
        results = pipeline.execute(sample_df)
        assert len(results) == 3
