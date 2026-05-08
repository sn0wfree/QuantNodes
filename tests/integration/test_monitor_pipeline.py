# coding=utf-8
"""QuantNodes.integration.test_monitor_pipeline 集成测试"""
import pandas as pd

from QuantNodes.operator_node.transform import TransformNode


class TestMonitorPipelineIntegration:
    """测试策略 -> 回测 -> 监控告警的完整流程"""

    def test_transform_node_to_dataframe_flow(self):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "code": ["A", "A", "A"],
            "factor": [0.1, 0.2, 0.3],
            "return": [0.01, 0.02, 0.03],
        })

        node = (
            TransformNode()
            .select(["date", "code", "factor"])
            .filter("factor > 0.15")
        )

        result = node.execute(df)
        assert len(result) == 2
        assert "factor" in result.columns

    def test_transform_node_chaining(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [10, 20, 30, 40, 50],
        })

        node1 = TransformNode().select(["a"])
        node2 = TransformNode().filter("a > 2")

        combined = node1.then(node2)
        result = combined.execute(df)

        assert len(result) == 3
        assert list(result.columns) == ["a"]


class TestFactorWorkflowIntegration:
    """测试因子定义 -> 计算 -> 存储的工作流"""

    def test_factor_transform_workflow(self):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "code": ["A", "B", "A", "B"],
            "price": [100, 200, 101, 201],
            "volume": [1000, 2000, 1100, 2100],
        })

        node = (
            TransformNode()
            .select(["date", "code", "price", "volume"])
            .filter("volume > 1500")
        )

        result = node.execute(df)
        assert len(result) == 2
        assert all(result["volume"] > 1500)


class TestOperatorNodeChainIntegration:
    """测试 OperatorNode 链式操作"""

    def test_operator_chain_with_transform(self):
        df = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5],
            "col2": [5, 4, 3, 2, 1],
            "category": ["A", "A", "B", "B", "B"],
        })

        select_node = TransformNode().select(["col1", "col2", "category"])
        filter_node = TransformNode().filter("col1 > 2")
        sort_node = TransformNode().sort_by("col1", ascending=True)

        pipeline = select_node.then(filter_node).then(sort_node)
        result = pipeline.execute(df)

        assert len(result) == 3
        assert result["col1"].iloc[0] == 3


class TestDataProcessingPipeline:
    """测试数据处理流水线"""

    def test_multi_step_processing(self):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "value": [1.0, 2.0, 3.0, 4.0],
        })

        steps = [
            TransformNode().select(["date", "value"]),
            TransformNode().filter("value > 1.5"),
            TransformNode().sort_by("value", ascending=False),
        ]

        result = df
        for step in steps:
            result = step.execute(result)

        assert len(result) == 3
        assert result["value"].iloc[0] == 4.0
