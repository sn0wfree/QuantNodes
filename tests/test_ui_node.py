# coding=utf-8
"""
DisplayNode 单元测试
"""
import pytest
import pandas as pd
from QuantNodes.ui_node import (
    VisualizationData,
    VisualizationType,
    TableDisplayNode,
    ChartDisplayNode,
    MetricDisplayNode,
    TextDisplayNode,
)


class TestVisualizationData:
    """VisualizationData 数据类测试"""

    def test_default_values(self):
        result = VisualizationData()
        assert result.viz_type == VisualizationType.TABLE
        assert result.title == ""
        assert result.data is None
        assert result.columns == []
        assert result.metadata == {}

    def test_custom_values(self):
        result = VisualizationData(
            viz_type=VisualizationType.CHART,
            title="Test Chart",
            data={"x": [1, 2, 3]},
            columns=["x", "y"],
            metadata={"source": "test"}
        )
        assert result.viz_type == VisualizationType.CHART
        assert result.title == "Test Chart"
        assert result.data == {"x": [1, 2, 3]}
        assert result.columns == ["x", "y"]
        assert result.metadata == {"source": "test"}


class TestVisualizationType:
    """VisualizationType 枚举测试"""

    def test_visualization_types(self):
        assert VisualizationType.TABLE == "table"
        assert VisualizationType.CHART == "chart"
        assert VisualizationType.METRIC == "metric"
        assert VisualizationType.TEXT == "text"
        assert VisualizationType.IMAGE == "image"


class TestTableDisplayNode:
    """TableDisplayNode 测试"""

    def test_init_default(self):
        node = TableDisplayNode()
        assert node.name == "TableDisplay"
        assert node.title == ""

    def test_init_custom(self):
        node = TableDisplayNode(
            name="CustomTable",
            title="回测结果",
        )
        assert node.name == "CustomTable"
        assert node.title == "回测结果"

    def test_execute_dataframe(self):
        node = TableDisplayNode(title="Test")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = node.execute(df)

        assert isinstance(result, VisualizationData)
        assert result.viz_type == VisualizationType.TABLE
        assert result.title == "Test"
        assert result.columns == ["a", "b"]
        assert isinstance(result.data, pd.DataFrame)

    def test_execute_dict(self):
        node = TableDisplayNode(title="Dict Test")
        data = {"x": 1, "y": 2}
        result = node.execute(data)

        assert result.viz_type == VisualizationType.TABLE
        assert result.columns == ["x", "y"]
        assert isinstance(result.data, pd.DataFrame)


class TestChartDisplayNode:
    """ChartDisplayNode 测试"""

    def test_init_default(self):
        node = ChartDisplayNode()
        assert node.name == "ChartDisplay"
        assert node.title == ""
        assert node.chart_type == "line"

    def test_init_custom(self):
        node = ChartDisplayNode(
            name="CustomChart",
            title="收益曲线",
            chart_type="area"
        )
        assert node.name == "CustomChart"
        assert node.title == "收益曲线"
        assert node.chart_type == "area"

    def test_execute_dataframe(self):
        node = ChartDisplayNode(title="Equity Curve", chart_type="line")
        df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "value": [100, 105]})
        result = node.execute(df)

        assert isinstance(result, VisualizationData)
        assert result.viz_type == VisualizationType.CHART
        assert result.title == "Equity Curve"
        assert result.metadata["chart_type"] == "line"

    def test_execute_dict(self):
        node = ChartDisplayNode()
        data = {"x": [1, 2, 3], "y": [10, 20, 30]}
        result = node.execute(data)

        assert result.viz_type == VisualizationType.CHART
        assert isinstance(result.data, pd.DataFrame)

    def test_chart_types(self):
        for chart_type in ["line", "bar", "area", "pie"]:
            node = ChartDisplayNode(chart_type=chart_type)
            result = node.execute(pd.DataFrame({"x": [1, 2]}))
            assert result.metadata["chart_type"] == chart_type


class TestMetricDisplayNode:
    """MetricDisplayNode 测试"""

    def test_init_default(self):
        node = MetricDisplayNode()
        assert node.name == "MetricDisplay"
        assert node.title == ""

    def test_init_custom(self):
        node = MetricDisplayNode(
            name="CustomMetric",
            title="总收益",
        )
        assert node.name == "CustomMetric"
        assert node.title == "总收益"

    def test_execute_numeric(self):
        node = MetricDisplayNode(title="Win Rate")
        result = node.execute(0.75)

        assert isinstance(result, VisualizationData)
        assert result.viz_type == VisualizationType.METRIC
        assert result.title == "Win Rate"
        assert result.data == 0.75

    def test_execute_dict_with_metadata(self):
        node = MetricDisplayNode(title="Return")
        data = {"value": 10000, "delta": 1500, "description": "总收益"}
        result = node.execute(data)

        assert result.data == 10000
        assert result.metadata["delta"] == 1500
        assert result.metadata["description"] == "总收益"


class TestTextDisplayNode:
    """TextDisplayNode 测试"""

    def test_init_default(self):
        node = TextDisplayNode()
        assert node.name == "TextDisplay"
        assert node.title == ""

    def test_init_custom(self):
        node = TextDisplayNode(
            name="CustomText",
            title="策略描述",
        )
        assert node.name == "CustomText"
        assert node.title == "策略描述"

    def test_execute_string(self):
        node = TextDisplayNode(title="Description")
        result = node.execute("这是一个测试策略")

        assert isinstance(result, VisualizationData)
        assert result.viz_type == VisualizationType.TEXT
        assert result.title == "Description"
        assert result.data == "这是一个测试策略"


class TestDisplayNodeStats:
    """DisplayNode 统计功能测试"""

    def test_stats_enabled(self):
        node = TableDisplayNode()
        assert node.stats is not None
        assert node.stats.execute_count == 0

    def test_stats_updated(self):
        node = TableDisplayNode()
        df = pd.DataFrame({"a": [1, 2]})
        node.execute(df)

        assert node.stats.execute_count == 1
        assert node.stats.success_count == 1


class TestDisplayNodeExecution:
    """DisplayNode 执行流程测试"""

    def test_execute_updates_state(self):
        node = MetricDisplayNode()
        result = node.execute(42)

        assert result is not None
        assert node.state.value == "success"

    def test_execute_with_none_input(self):
        node = TableDisplayNode()
        result = node.execute(None)

        assert result.viz_type == VisualizationType.TABLE
        assert result.data is None

    def test_node_id_generated(self):
        node1 = TableDisplayNode()
        node2 = TableDisplayNode()
        assert node1.node_id != node2.node_id


class TestDisplayNodeEdgeCases:
    """DisplayNode 边界情况测试"""

    def test_table_empty_dataframe(self):
        node = TableDisplayNode(title="Empty")
        df = pd.DataFrame()
        result = node.execute(df)

        assert result.viz_type == VisualizationType.TABLE
        assert len(result.columns) == 0

    def test_table_list_input(self):
        node = TableDisplayNode(title="List")
        result = node.execute([1, 2, 3])

        assert result.viz_type == VisualizationType.TABLE
        assert result.data is not None

    def test_chart_empty_dataframe(self):
        node = ChartDisplayNode(title="Empty Chart")
        df = pd.DataFrame()
        result = node.execute(df)

        assert result.viz_type == VisualizationType.CHART

    def test_chart_list_input(self):
        node = ChartDisplayNode(title="List")
        result = node.execute([1, 2, 3])

        assert result.viz_type == VisualizationType.CHART
        assert result.data == [1, 2, 3]

    def test_metric_none_value(self):
        node = MetricDisplayNode(title="None")
        result = node.execute(None)

        assert result.data is None

    def test_text_none_input(self):
        node = TextDisplayNode(title="None")
        result = node.execute(None)

        assert result.data is None

    def test_text_with_special_characters(self):
        node = TextDisplayNode(title="Special")
        result = node.execute("测试\n换行\t制表")

        assert result.data == "测试\n换行\t制表"


class TestDisplayNodeName:
    """DisplayNode 名称测试"""

    def test_default_names(self):
        assert TableDisplayNode().name == "TableDisplay"
        assert ChartDisplayNode().name == "ChartDisplay"
        assert MetricDisplayNode().name == "MetricDisplay"
        assert TextDisplayNode().name == "TextDisplay"

    def test_custom_name_preserved(self):
        node = TableDisplayNode(name="MyTable")
        assert node.name == "MyTable"


class TestVisualizationDataMetadata:
    """VisualizationData 元数据测试"""

    def test_metadata_default_empty(self):
        result = VisualizationData()
        assert result.metadata == {}

    def test_metadata_custom(self):
        result = VisualizationData(
            metadata={"source": "backtest", "timestamp": "2024-01-01"}
        )
        assert result.metadata["source"] == "backtest"

    def test_result_carries_metadata_through_execute(self):
        node = TableDisplayNode()
        df = pd.DataFrame({"a": [1]})
        result = node.execute(df)

        assert result.metadata == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
