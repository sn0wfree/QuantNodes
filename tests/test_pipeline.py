# coding=utf-8
"""
Pipeline 组合原语单元测试
"""

import pytest

from QuantNodes.core import (
    BaseNode,
    Pipeline,
    Parallel,
    Join,
)


class MultiplyNode(BaseNode):
    """乘法节点"""
    def __init__(self, factor: int = 2, name=None):
        super().__init__(name=name or "Multiply")
        self.factor = factor

    def _execute(self, input_data, **kwargs):
        return input_data * self.factor


class AddNode(BaseNode):
    """加法节点"""
    def __init__(self, value: int = 1, name=None):
        super().__init__(name=name or "Add")
        self.value = value

    def _execute(self, input_data, **kwargs):
        return input_data + self.value


class DictKeyNode(BaseNode):
    """返回字典中指定 key 的节点"""
    def __init__(self, key: str, name=None):
        super().__init__(name=name or f"GetKey_{key}")
        self.key = key

    def _execute(self, input_data: dict, **kwargs):
        return input_data[self.key]


class TestPipeline:
    """Pipeline 线性管道测试"""

    def test_empty_pipeline(self):
        """测试空管道"""
        p = Pipeline([])
        assert len(p) == 0

    def test_single_node(self):
        """测试单个节点"""
        p = Pipeline([MultiplyNode(2)])
        result = p.execute(5)
        assert result == 10

    def test_multiple_nodes(self):
        """测试多个节点"""
        p = Pipeline([
            MultiplyNode(2),  # 5 * 2 = 10
            AddNode(3),        # 10 + 3 = 13
            MultiplyNode(10),  # 13 * 10 = 130
        ])
        result = p.execute(5)
        assert result == 130

    def test_rshift_chain(self):
        """测试 >> 链式调用"""
        p = MultiplyNode(2) >> AddNode(3) >> MultiplyNode(10)
        result = p.execute(5)
        assert result == 130

    def test_nested_pipeline(self):
        """测试嵌套管道"""
        inner = Pipeline([AddNode(1), MultiplyNode(2)])
        outer = Pipeline([MultiplyNode(10), inner, AddNode(5)])
        # 5 * 10 = 50 -> +1 = 51 -> *2 = 102 -> +5 = 107
        result = outer.execute(5)
        assert result == 107

    def test_pipeline_rshift(self):
        """测试 Pipeline 与其他节点组合"""
        p = Pipeline([MultiplyNode(2)]) >> AddNode(3)
        assert len(p) == 2
        result = p.execute(5)
        assert result == 13

    def test_pipeline_iteration(self):
        """测试管道迭代"""
        nodes = [MultiplyNode(2), AddNode(3), MultiplyNode(4)]
        p = Pipeline(nodes)
        for i, node in enumerate(p):
            assert node is nodes[i]

    def test_pipeline_index(self):
        """测试管道索引访问"""
        p = Pipeline([MultiplyNode(2), AddNode(3)])
        assert isinstance(p[0], MultiplyNode)
        assert isinstance(p[1], AddNode)

    def test_pipeline_to_info(self):
        """测试管道导出"""
        p = Pipeline([MultiplyNode(2), AddNode(3)])
        d = p.to_info()
        assert d['class'] == 'Pipeline'
        assert len(d['nodes']) == 2


class TestParallel:
    """Parallel 并行分叉测试"""

    def test_parallel_basic(self):
        """测试基本并行"""
        p = Parallel({
            'double': MultiplyNode(2),
            'triple': MultiplyNode(3),
            'add5': AddNode(5),
        }, parallel=False)  # 串行，方便测试

        result = p.execute(10)
        assert result['double'] == 20
        assert result['triple'] == 30
        assert result['add5'] == 15

    def test_parallel_with_threads(self):
        """测试真正的并行执行"""
        p = Parallel({
            'a': MultiplyNode(2),
            'b': MultiplyNode(3),
        }, parallel=True, max_workers=2)

        result = p.execute(10)
        assert result['a'] == 20
        assert result['b'] == 30

    def test_parallel_combine(self):
        """测试 | 合并运算符"""
        p1 = Parallel({'a': MultiplyNode(2)})
        p2 = Parallel({'b': AddNode(3)})
        p = p1 | p2

        result = p.execute(10)
        assert result['a'] == 20
        assert result['b'] == 13

    def test_parallel_to_info(self):
        """测试并行节点导出"""
        p = Parallel({'a': MultiplyNode(2)})
        d = p.to_info()
        assert 'branches' in d
        assert 'a' in d['branches']


class TestJoin:
    """Join 聚合组合测试"""

    def test_join_with_kwargs(self):
        """测试关键字参数形式"""
        # lambda 接受 a, b 两个参数
        join = Join(lambda a, b: a + b)
        result = join.execute({'a': 10, 'b': 20})
        assert result == 30

    def test_join_with_dict_arg(self):
        """测试单参数（字典）形式"""
        # lambda 接受整个字典
        join = lambda d: d['a'] * d['b']
        j = Join(join)
        result = j.execute({'a': 10, 'b': 20})
        assert result == 200

    def test_join_with_complex_function(self):
        """测试复杂聚合函数"""
        def weighted_sum(mom, vol, value):
            return mom * 0.5 + vol * 0.3 + value * 0.2

        j = Join(weighted_sum)
        result = j.execute({'mom': 100, 'vol': 50, 'value': 20})
        expected = 100 * 0.5 + 50 * 0.3 + 20 * 0.2
        assert result == expected

    def test_join_non_dict_input(self):
        """测试非字典输入"""
        from QuantNodes.core.node import NodeExecutionError
        j = Join(lambda x: x)
        with pytest.raises(NodeExecutionError) as excinfo:
            j.execute("not a dict")
        assert "dict input" in str(excinfo.value)

    def test_join_to_info(self):
        """测试 Join 导出"""
        j = Join(lambda a, b: a + b)
        d = j.to_info()
        assert 'join_func' in d


class TestPipelineParallelJoin:
    """测试 Pipeline + Parallel + Join 组合使用"""

    def test_full_pipeline(self):
        """测试完整的因子计算管道"""
        pipeline = (
            Parallel({
                'price': MultiplyNode(2),  # 10 -> 20
                'volume': MultiplyNode(3),  # 10 -> 30
            })
            >> Join(lambda price, volume: price / volume)  # 20 / 30 = 0.666...
        )

        result = pipeline.execute(10)
        assert result == 20 / 30

    def test_nested_parallel(self):
        """测试嵌套并行"""
        inner = Parallel({
            'x': AddNode(1),
            'y': MultiplyNode(2),
        })
        outer = Parallel({
            'inner': inner,
            'z': MultiplyNode(10),
        })
        result = outer.execute(5)
        assert result['inner']['x'] == 6
        assert result['inner']['y'] == 10
        assert result['z'] == 50
