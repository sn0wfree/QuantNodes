# -*- coding: utf-8 -*-
"""FactorNode unit tests"""
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, '/home/ll/Public/QuantNodes/QuantNodes')

from core.node import BaseNode
from factor_node.factor_node import FactorNode, FactorPipeline
from factor_node.point_factor import PointFactorNode, ArithmeticFactorNode
from factor_node.time_factor import TimeFactorNode, ExpandingFactorNode
from factor_node.cross_section_factor import CrossSectionFactorNode, GroupRankFactorNode
from factor_node.panel_factor import PanelFactorNode, DelayFactorNode, DeltaFactorNode


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


class TestPointFactorNode:
    """Tests for PointFactorNode"""

    def test_point_factor_expression(self, sample_df):
        """测试表达式计算"""
        factor = PointFactorNode(expression="close / open - 1", result_name="ret")
        result = factor.execute(sample_df)
        
        assert "ret" in result.columns
        assert len(result) == len(sample_df)

    def test_point_factor_func(self, sample_df):
        """测试自定义函数"""
        factor = PointFactorNode(
            func=lambda row: row['close'] / row['open'] - 1,
            result_name="ret"
        )
        result = factor.execute(sample_df)
        
        assert "ret" in result.columns

    def test_point_factor_no_input(self):
        """测试无输入"""
        factor = PointFactorNode(expression="close / open - 1")
        with pytest.raises(Exception, match="input_data is required"):
            factor.execute(None)


class TestArithmeticFactorNode:
    """Tests for ArithmeticFactorNode"""

    def test_arithmetic_add(self, sample_df):
        """测试加法"""
        f1 = PointFactorNode(expression="close", result_name="close")
        f2 = PointFactorNode(expression="open", result_name="open")
        
        factor = ArithmeticFactorNode(factors=[f1, f2], operator="add")
        result = factor.execute(sample_df)
        
        expected = sample_df["close"] + sample_df["open"]
        pd.testing.assert_series_equal(result.iloc[:, 0].reset_index(drop=True), expected.reset_index(drop=True), check_names=False)

    def test_arithmetic_div(self, sample_df):
        """测试除法"""
        f1 = PointFactorNode(expression="close", result_name="close")
        f2 = PointFactorNode(expression="open", result_name="open")
        
        factor = ArithmeticFactorNode(factors=[f1, f2], operator="div")
        result = factor.execute(sample_df)
        
        expected = (sample_df["close"] / sample_df["open"]).dropna()
        pd.testing.assert_series_equal(result.iloc[:, 0].reset_index(drop=True), expected.reset_index(drop=True), check_names=False, rtol=1e-5)


class TestTimeFactorNode:
    """Tests for TimeFactorNode"""

    def test_time_mean(self, timeseries_df):
        """测试移动平均"""
        factor = TimeFactorNode(window=3, operation="mean", column="value")
        result = factor.execute(timeseries_df)
        
        assert len(result) == len(timeseries_df)
        assert result.iloc[2, 0] == pytest.approx(11.0)


class TestCrossSectionFactorNode:
    """Tests for CrossSectionFactorNode"""

    def test_cross_section_rank(self, sample_df):
        """测试横截面排名"""
        factor = CrossSectionFactorNode(operation="rank", column="return")
        result = factor.execute(sample_df)
        
        assert len(result) == len(sample_df)


class TestPanelFactorNode:
    """Tests for PanelFactorNode"""

    def test_panel_zscore(self, sample_df):
        """测试面板标准化"""
        factor = PanelFactorNode(
            operations=[("zscore", {"column": "return"})]
        )
        result = factor.execute(sample_df)
        
        assert len(result) == len(sample_df)


class TestDelayFactorNode:
    """Tests for DelayFactorNode"""

    def test_delay(self, timeseries_df):
        """测试延迟"""
        base_factor = TimeFactorNode(window=3, operation="mean", column="value")
        factor = DelayFactorNode(base_factor=base_factor, periods=1)
        result = factor.execute(timeseries_df)
        
        assert len(result) == len(timeseries_df)


class TestDeltaFactorNode:
    """Tests for DeltaFactorNode"""

    def test_delta_diff(self, timeseries_df):
        """测试差分"""
        base_factor = TimeFactorNode(window=3, operation="mean", column="value")
        factor = DeltaFactorNode(base_factor=base_factor, periods=1, mode="diff")
        result = factor.execute(timeseries_df)
        
        assert len(result) == len(timeseries_df)


class TestFactorPipeline:
    """Tests for FactorPipeline"""

    def test_pipeline_execute(self, sample_df):
        """测试管道执行"""
        f1 = PointFactorNode(expression="close / open - 1", result_name="ret")
        f2 = PointFactorNode(expression="volume / 1000", result_name="vol_adj")
        
        pipeline = FactorPipeline([f1, f2])
        results = pipeline.execute(sample_df)
        
        assert "PointFactorNode" in results
        assert "PointFactorNode_1" in results
        assert isinstance(results["PointFactorNode"], pd.DataFrame)
