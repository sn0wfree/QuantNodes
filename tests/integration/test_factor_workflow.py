# coding=utf-8
"""QuantNodes.integration.test_factor_workflow 集成测试"""
import pytest
import pandas as pd
import polars as pl

from QuantNodes.operator_node.transform import TransformNode
from QuantNodes.operators.section import SectionOperators
from QuantNodes.operators.time_series import TimeSeriesOperators
from QuantNodes.operators.composite import CompositeOperators


class TestFactorDefinitionToCalculation:
    """测试因子定义 -> 计算 -> 回测的完整流程"""

    def test_factor_creation_and_transformation(self):
        raw_data = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "code": ["A", "B", "A", "B"],
            "price": [100.0, 200.0, 101.0, 201.0],
            "volume": [1000, 2000, 1100, 2100],
        })

        select_node = TransformNode().select(["date", "code", "price", "volume"])
        selected = select_node.execute(raw_data)
        assert len(selected.columns) == 4

    def test_factor_with_operators(self):
        df = pl.DataFrame({
            "factor1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "factor2": [5.0, 4.0, 3.0, 2.0, 1.0],
        })

        combined = CompositeOperators.weighted_sum(
            ["factor1", "factor2"],
            [0.6, 0.4]
        )

        result = df.select(combined)
        values = result["factor1"].to_list()
        assert values[0] == pytest.approx(1.0 * 0.6 + 5.0 * 0.4)

    def test_factor_zscore_normalization(self):
        df = pl.DataFrame({
            "factor": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        result = df.select(SectionOperators.zscore("factor"))
        values = result["factor"].to_list()
        assert abs(sum(values)) < 1e-10


class TestCrossSectionOperations:
    """测试截面运算"""

    def test_section_rank(self):
        df = pl.DataFrame({
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "code": ["A", "B", "A", "B"],
            "factor": [1.0, 2.0, 3.0, 4.0],
        })

        ranked = df.select(SectionOperators.rank("factor", method="dense"))
        values = ranked["factor"].to_list()
        assert values[0] == 0.0
        assert values[-1] == 1.0

    def test_section_winsorize(self):
        df = pl.DataFrame({
            "factor": [-10.0, -5.0, 0.0, 5.0, 10.0],
        })

        result = df.select(SectionOperators.winsorize("factor", lower=0.2, upper=0.2))
        values = result["factor"].to_list()
        assert values[0] >= -5.0
        assert values[-1] <= 5.0


class TestTimeSeriesAndSectionCombined:
    """测试时序运算和截面运算的组合"""

    def test_rolling_mean(self):
        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        })

        rolling_expr = TimeSeriesOperators.ts_mean("a", window=3)
        result = df.select(rolling_expr)
        assert len(result) == 10

    def test_factor_pipeline_multi_step_pandas(self):
        pdf = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "code": ["A", "B", "A", "B"],
            "factor": [0.5, 1.5, 2.5, 3.5],
            "return": [0.01, 0.02, 0.03, 0.04],
        })

        select_step = TransformNode().select(["date", "code", "factor", "return"])
        filtered = select_step.execute(pdf)

        assert len(filtered) == 4
        assert "factor" in filtered.columns


class TestDataframeToPolarsConversion:
    """测试 DataFrame 和 Polars 之间的转换"""

    def test_pandas_to_polars(self):
        pdf = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        })

        ldf = pl.DataFrame(pdf)
        assert isinstance(ldf, pl.DataFrame)
        assert ldf.shape == (3, 2)


class TestOperatorComposition:
    """测试算子组合"""

    def test_composite_weighted_avg(self):
        df = pl.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "f2": [4.0, 5.0, 6.0],
            "f3": [7.0, 8.0, 9.0],
        })

        weighted = CompositeOperators.weighted_avg(["f1", "f2", "f3"])
        result = df.select(weighted)
        assert result.shape == (3, 1)
