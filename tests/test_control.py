# coding=utf-8
"""
控制流节点单元测试
"""

import pytest
import pandas as pd
import numpy as np

from QuantNodes.core import (
    BaseNode,
    IfNode,
    MapNode,
    WhileNode,
)


class MultiplyNode(BaseNode):
    """乘法节点"""
    def __init__(self, factor: int = 2, name=None):
        super().__init__(name=name or "Multiply")
        self.factor = factor

    def _execute(self, input_data, **kwargs):
        return input_data * self.factor


class AddOneNode(BaseNode):
    """加一节点"""
    def _execute(self, input_data, **kwargs):
        return input_data + 1


class SharpeImproverNode(BaseNode):
    """改进夏普比率的节点（用于测试 WhileNode）"""
    def __init__(self, improvement=0.1):
        super().__init__(name="SharpeImprover")
        self.improvement = improvement

    def _execute(self, input_data, **kwargs):
        class Result:
            def __init__(self, sharpe):
                self.metrics = type('Metrics', (), {'sharpe': sharpe})()
        return Result(input_data.metrics.sharpe + self.improvement)


class TestIfNode:
    """IfNode 条件分支测试"""

    def test_if_with_both_branches(self):
        """测试有真假两个分支"""
        node = IfNode(
            condition=lambda x: x > 10,
            true_branch=MultiplyNode(2),
            false_branch=MultiplyNode(3),
        )

        # 条件为 True
        assert node.execute(20) == 40
        assert node._last_branch_taken == True

        # 条件为 False
        assert node.execute(5) == 15
        assert node._last_branch_taken == False

    def test_if_without_false_branch(self):
        """测试只有真分支（False 时返回原输入）"""
        node = IfNode(
            condition=lambda x: x > 10,
            true_branch=MultiplyNode(2),
        )

        assert node.execute(20) == 40
        assert node.execute(5) == 5  # 返回原输入

    def test_if_complex_condition(self):
        """测试复杂条件"""
        def check_dataframe(df):
            return df['value'].mean() > 50

        node = IfNode(
            condition=check_dataframe,
            true_branch=MultiplyNode(2),
            false_branch=MultiplyNode(0.5),
        )

        df1 = pd.DataFrame({'value': [100, 200, 300]})
        result1 = node.execute(df1)
        assert (result1['value'] == [200, 400, 600]).all()

        df2 = pd.DataFrame({'value': [10, 20, 30]})
        result2 = node.execute(df2)
        assert (result2['value'] == [5, 10, 15]).all()

    def test_if_to_dict(self):
        """测试节点导出"""
        node = IfNode(
            condition=lambda x: x > 0,
            true_branch=MultiplyNode(2),
            false_branch=MultiplyNode(3),
        )
        d = node.to_info()
        assert d['true_branch'] is not None
        assert d['false_branch'] is not None


class TestMapNode:
    """MapNode 分组映射测试"""

    def test_map_with_dataframe_column(self):
        """测试按 DataFrame 列名分组"""
        df = pd.DataFrame({
            'date': ['2020-01-01', '2020-01-01', '2020-01-02', '2020-01-02'],
            'value': [1, 2, 3, 4],
        })

        class SumGroupNode(BaseNode):
            def _execute(self, group_df, **kwargs):
                return group_df['value'].sum()

        mapper = MapNode(
            node=SumGroupNode(),
            group_by='date',
            parallel=False,
        )

        results = mapper.execute(df)
        assert len(results) == 2  # 两个日期分组

    def test_map_with_custom_function(self):
        """测试自定义分组函数"""
        data = [1, 2, 3, 4, 5, 6]

        mapper = MapNode(
            node=MultiplyNode(10),
            group_by=lambda x: 'even' if x % 2 == 0 else 'odd',
            parallel=False,
        )

        results = mapper.execute(data)
        # [(key, result), ...]
        assert len(results) == 2

    def test_map_list_without_groupby(self):
        """测试列表默认分组"""
        data = [1, 2, 3]

        mapper = MapNode(
            node=MultiplyNode(10),
            group_by=None,
            parallel=False,
        )

        results = mapper.execute(data)
        assert len(results) == 3

    def test_map_parallel(self):
        """测试并行执行"""
        df = pd.DataFrame({
            'group': ['A'] * 10 + ['B'] * 10 + ['C'] * 10,
            'value': range(30),
        })

        class NoopNode(BaseNode):
            def _execute(self, data, **kwargs):
                return len(data)

        mapper = MapNode(
            node=NoopNode(),
            group_by='group',
            parallel=True,
            max_workers=2,
        )

        results = mapper.execute(df)
        assert len(results) == 3
        # 每个分组应该有 10 条
        group_lengths = sorted(length for key, length in results)
        assert group_lengths == [10, 10, 10]

    def test_map_to_dict(self):
        """测试节点导出"""
        mapper = MapNode(
            node=MultiplyNode(2),
            group_by='date',
            max_workers=4,
        )
        d = mapper.to_info()
        assert d['group_by'] == 'date'
        assert d['max_workers'] == 4


class TestWhileNode:
    """WhileNode 条件循环测试"""

    def test_while_basic(self):
        """测试基本循环"""
        loop = WhileNode(
            condition=lambda x: x < 10,
            body=AddOneNode(),
            max_iterations=20,
        )

        result = loop.execute(0)
        assert result == 10
        assert loop.iteration_count == 10

    def test_while_max_iterations(self):
        """测试达到最大迭代次数"""
        loop = WhileNode(
            condition=lambda x: True,  # 永远为 True
            body=AddOneNode(),
            max_iterations=5,
        )

        result = loop.execute(0)
        assert result == 5
        assert loop.iteration_count == 5

    def test_while_zero_iterations(self):
        """测试一次也不执行"""
        loop = WhileNode(
            condition=lambda x: False,  # 永远为 False
            body=AddOneNode(),
            max_iterations=10,
        )

        result = loop.execute(5)
        assert result == 5
        assert loop.iteration_count == 0

    def test_while_with_object(self):
        """测试对象输入"""
        class MockResult:
            def __init__(self, sharpe):
                self.metrics = type('Metrics', (), {'sharpe': sharpe})()

        loop = WhileNode(
            condition=lambda result: result.metrics.sharpe < 1.5,
            body=SharpeImproverNode(0.2),
            max_iterations=10,
        )

        start = MockResult(1.0)
        result = loop.execute(start)
        assert result.metrics.sharpe >= 1.5
        assert loop.iteration_count == 3  # 1.0 -> 1.2 -> 1.4 -> 1.6

    def test_while_to_dict(self):
        """测试节点导出"""
        loop = WhileNode(
            condition=lambda x: x < 10,
            body=AddOneNode(),
            max_iterations=5,
        )
        d = loop.to_info()
        assert d['max_iterations'] == 5
        assert d['body'] is not None
