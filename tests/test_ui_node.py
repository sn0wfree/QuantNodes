# coding=utf-8
"""
UINode 单元测试
"""
import pytest
import pandas as pd
from QuantNodes.ui_node import (
    UINode,
    UIDisplayResult,
    DisplayType,
    TableDisplayNode,
    ChartDisplayNode,
    MetricDisplayNode,
    TextDisplayNode,
)


class TestUIDisplayResult:
    """UIDisplayResult 数据类测试"""

    def test_default_values(self):
        result = UIDisplayResult()
        assert result.display_type == DisplayType.TABLE
        assert result.title == ""
        assert result.data is None
        assert result.config == {}
        assert result.columns == []
        assert result.metadata == {}

    def test_custom_values(self):
        result = UIDisplayResult(
            display_type=DisplayType.CHART,
            title="Test Chart",
            data={"x": [1, 2, 3]},
            columns=["x", "y"],
            metadata={"source": "test"}
        )
        assert result.display_type == DisplayType.CHART
        assert result.title == "Test Chart"
        assert result.data == {"x": [1, 2, 3]}
        assert result.columns == ["x", "y"]
        assert result.metadata == {"source": "test"}


class TestDisplayType:
    """DisplayType 枚举测试"""

    def test_display_types(self):
        assert DisplayType.TABLE == "table"
        assert DisplayType.CHART == "chart"
        assert DisplayType.METRIC == "metric"
        assert DisplayType.TEXT == "text"
        assert DisplayType.IMAGE == "image"


class TestTableDisplayNode:
    """TableDisplayNode 测试"""

    def test_init_default(self):
        node = TableDisplayNode()
        assert node.name == "TableDisplay"
        assert node.title == ""
        assert node.page_size == 50

    def test_init_custom(self):
        node = TableDisplayNode(
            name="CustomTable",
            title="回测结果",
            page_size=100
        )
        assert node.name == "CustomTable"
        assert node.title == "回测结果"
        assert node.page_size == 100

    def test_execute_dataframe(self):
        node = TableDisplayNode(title="Test")
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = node.execute(df)

        assert isinstance(result, UIDisplayResult)
        assert result.display_type == DisplayType.TABLE
        assert result.title == "Test"
        assert result.columns == ["a", "b"]
        assert isinstance(result.data, pd.DataFrame)

    def test_execute_dict(self):
        node = TableDisplayNode(title="Dict Test")
        data = {"x": 1, "y": 2}
        result = node.execute(data)

        assert result.display_type == DisplayType.TABLE
        assert result.columns == ["x", "y"]
        assert isinstance(result.data, pd.DataFrame)

    def test_execute_with_config(self):
        node = TableDisplayNode(config={"use_container_width": False})
        df = pd.DataFrame({"a": [1]})
        result = node.execute(df)

        assert result.config["use_container_width"] is False


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

        assert isinstance(result, UIDisplayResult)
        assert result.display_type == DisplayType.CHART
        assert result.title == "Equity Curve"
        assert result.config["chart_type"] == "line"

    def test_execute_dict(self):
        node = ChartDisplayNode()
        data = {"x": [1, 2, 3], "y": [10, 20, 30]}
        result = node.execute(data)

        assert result.display_type == DisplayType.CHART
        assert isinstance(result.data, pd.DataFrame)

    def test_chart_types(self):
        for chart_type in ["line", "bar", "area", "pie"]:
            node = ChartDisplayNode(chart_type=chart_type)
            result = node.execute(pd.DataFrame({"x": [1, 2]}))
            assert result.config["chart_type"] == chart_type


class TestMetricDisplayNode:
    """MetricDisplayNode 测试"""

    def test_init_default(self):
        node = MetricDisplayNode()
        assert node.name == "MetricDisplay"
        assert node.title == ""
        assert node.delta is None
        assert node.delta_color == "off"

    def test_init_custom(self):
        node = MetricDisplayNode(
            name="CustomMetric",
            title="总收益",
            delta=15.5,
            delta_color="normal"
        )
        assert node.name == "CustomMetric"
        assert node.title == "总收益"
        assert node.delta == 15.5
        assert node.delta_color == "normal"

    def test_execute_numeric(self):
        node = MetricDisplayNode(title="Win Rate")
        result = node.execute(0.75)

        assert isinstance(result, UIDisplayResult)
        assert result.display_type == DisplayType.METRIC
        assert result.title == "Win Rate"
        assert result.data == 0.75

    def test_execute_dict_with_delta(self):
        node = MetricDisplayNode(title="Return")
        data = {"value": 10000, "delta": 1500, "delta_color": "normal"}
        result = node.execute(data)

        assert result.data == 10000
        assert result.config["delta"] == 1500
        assert result.config["delta_color"] == "normal"


class TestTextDisplayNode:
    """TextDisplayNode 测试"""

    def test_init_default(self):
        node = TextDisplayNode()
        assert node.name == "TextDisplay"
        assert node.title == ""
        assert node.markdown is True

    def test_init_custom(self):
        node = TextDisplayNode(
            name="CustomText",
            title="策略描述",
            markdown=False
        )
        assert node.name == "CustomText"
        assert node.title == "策略描述"
        assert node.markdown is False

    def test_execute_string(self):
        node = TextDisplayNode(title="Description")
        result = node.execute("这是一个测试策略")

        assert isinstance(result, UIDisplayResult)
        assert result.display_type == DisplayType.TEXT
        assert result.title == "Description"
        assert result.data == "这是一个测试策略"

    def test_execute_markdown_config(self):
        node = TextDisplayNode(markdown=True)
        result = node.execute("# Hello")

        assert result.config["markdown"] is True


class TestUINodeStats:
    """UINode 统计功能测试"""

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


class TestUINodeExecution:
    """UINode 执行流程测试"""

    def test_execute_updates_state(self):
        node = MetricDisplayNode()
        result = node.execute(42)

        assert result is not None
        assert node.state.value == "success"

    def test_execute_with_none_input(self):
        node = TableDisplayNode()
        result = node.execute(None)

        assert result.display_type == DisplayType.TABLE
        assert result.data is None

    def test_node_id_generated(self):
        node1 = TableDisplayNode()
        node2 = TableDisplayNode()
        assert node1.node_id != node2.node_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
