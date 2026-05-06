# coding=utf-8
"""QuantNodes.operator_node.transform 单元测试"""
import pandas as pd
import pytest

from QuantNodes.operator_node.transform import TransformNode


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "col1": [1, 2, 3, 4, 5],
        "col2": [10, 20, 30, 40, 50],
        "category": ["A", "A", "B", "B", "B"],
        "value": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


class TestTransformNodeSelect:
    def test_select_columns(self, sample_df):
        node = TransformNode().select(["col1", "col2"])
        result = node.execute(sample_df)
        assert list(result.columns) == ["col1", "col2"]
        assert len(result) == 5

    def test_select_single_column(self, sample_df):
        node = TransformNode().select(["col1"])
        result = node.execute(sample_df)
        assert list(result.columns) == ["col1"]


class TestTransformNodeDrop:
    def test_drop_column(self, sample_df):
        node = TransformNode().drop(["col2"])
        result = node.execute(sample_df)
        assert "col2" not in result.columns
        assert "col1" in result.columns


class TestTransformNodeFilter:
    def test_filter_with_string(self, sample_df):
        node = TransformNode().filter("col1 > 2")
        result = node.execute(sample_df)
        assert len(result) == 3
        assert result["col1"].min() > 2

    def test_filter_with_callable(self, sample_df):
        node = TransformNode().filter(lambda df: df["value"] > 2.0)
        result = node.execute(sample_df)
        assert len(result) == 3
        assert all(result["value"] > 2.0)


class TestTransformNodeAggregate:
    def test_aggregate_sum(self, sample_df):
        node = TransformNode().aggregate(
            group_by=["category"],
            agg={"value": "sum"}
        )
        result = node.execute(sample_df)
        assert len(result) == 2
        assert "value" in result.columns

    def test_aggregate_mean(self, sample_df):
        node = TransformNode().aggregate(
            group_by=["category"],
            agg={"value": "mean"}
        )
        result = node.execute(sample_df)
        assert len(result) == 2


class TestTransformNodeSortBy:
    def test_sort_by_ascending(self, sample_df):
        node = TransformNode().sort_by("col1", ascending=True)
        result = node.execute(sample_df)
        assert result["col1"].iloc[0] == 1

    def test_sort_by_descending(self, sample_df):
        node = TransformNode().sort_by("col1", ascending=False)
        result = node.execute(sample_df)
        assert result["col1"].iloc[0] == 5


class TestTransformNodeRename:
    def test_rename_columns(self, sample_df):
        node = TransformNode().rename({"col1": "new_col1"})
        result = node.execute(sample_df)
        assert "new_col1" in result.columns
        assert "col1" not in result.columns


class TestTransformNodeFillna:
    def test_fillna_value(self, sample_df):
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        node = TransformNode().fillna(0.0)
        result = node.execute(df)
        assert result["a"].tolist() == [1.0, 0.0, 3.0]


class TestTransformNodeApply:
    def test_apply_function(self, sample_df):
        node = TransformNode().apply(lambda df: df * 2)
        result = node.execute(sample_df[["col1"]])
        assert result["col1"].iloc[0] == 2


class TestTransformNodeChaining:
    def test_chaining_operations(self, sample_df):
        node = (
            TransformNode()
            .select(["col1", "value"])
            .filter("col1 > 2")
            .sort_by("col1", ascending=True)
        )
        result = node.execute(sample_df)
        assert len(result) == 3
        assert list(result.columns) == ["col1", "value"]

    def test_then_chaining(self, sample_df):
        node1 = TransformNode().select(["col1", "col2"])
        node2 = TransformNode().filter("col1 > 2")
        combined = node1.then(node2)
        result = combined.execute(sample_df)
        assert len(result) == 3


class TestTransformNodeRepr:
    def test_repr(self, sample_df):
        node = TransformNode().select(["col1"])
        assert "TransformNode" in repr(node)
        assert "operations=1" in repr(node)


class TestTransformNodeErrors:
    def test_execute_without_input(self):
        from QuantNodes.core.node import NodeExecutionError
        node = TransformNode().select(["col1"])
        with pytest.raises(NodeExecutionError):
            node.execute()

    def test_execute_with_wrong_type(self):
        from QuantNodes.core.node import NodeExecutionError
        node = TransformNode().select(["col1"])
        with pytest.raises(NodeExecutionError):
            node.execute("not a dataframe")
