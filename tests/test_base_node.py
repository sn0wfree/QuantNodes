# coding=utf-8
"""
BaseNode 基类单元测试
"""

import pytest
from typing import Any

from QuantNodes.core import (
    BaseNode,
    NodeState,
    NodeExecutionError,
)


class TestNode(BaseNode):
    """测试用节点"""

    def _execute(self, input_data: Any = None, **kwargs) -> Any:
        if input_data is None:
            return "ok"
        return input_data * 2


class FailingTestNode(BaseNode):
    """会失败的测试节点"""

    def _execute(self, input_data: Any = None, **kwargs) -> Any:
        raise ValueError("Expected failure")


class TestBaseNode:
    """BaseNode 基类测试"""

    def test_node_initialization(self):
        """测试节点初始化"""
        node = TestNode(name="test_node")
        assert node.name == "test_node"
        assert node.node_id.startswith("test_node_")
        assert len(node.node_id) == len("test_node_") + 8  # 8 hex chars
        assert node.state == NodeState.IDLE
        assert node._last_error is None
        assert node._last_result is None

    def test_default_name(self):
        """测试默认节点名称"""
        node = TestNode()
        assert node.name == "TestNode"

    def test_execute_success(self):
        """测试成功执行"""
        node = TestNode()
        result = node.execute(5)

        assert result == 10
        assert node.state == NodeState.SUCCESS
        assert node._last_result == 10
        assert node._last_error is None
        assert node.stats is not None
        assert node.stats.execute_count == 1
        assert node.stats.success_count == 1
        assert node.stats.failed_count == 0

    def test_execute_failure(self):
        """测试执行失败"""
        node = FailingTestNode()

        with pytest.raises(NodeExecutionError) as excinfo:
            node.execute(5)

        assert "Expected failure" in str(excinfo.value)
        assert node.state == NodeState.FAILED
        assert node._last_error is not None
        assert node.stats.failed_count == 1

    def test_call_method(self):
        """测试函数调用方式"""
        node = TestNode()
        result = node(10)  # __call__
        assert result == 20

    def test_rshift_operator(self):
        """测试 >> 管道运算符"""
        node1 = TestNode(name="A")
        node2 = TestNode(name="B")

        pipeline = node1 >> node2

        from QuantNodes.core import Pipeline
        assert isinstance(pipeline, Pipeline)
        assert len(pipeline) == 2
        assert pipeline[0].name == "A"
        assert pipeline[1].name == "B"

    def test_reset(self):
        """测试重置节点状态"""
        node = TestNode()
        node.execute(5)
        assert node.state == NodeState.SUCCESS
        assert node._last_result == 10
        assert node.stats.execute_count == 1

        node.reset()
        assert node.state == NodeState.IDLE
        assert node._last_result is None
        assert node._last_error is None
        assert node.stats.execute_count == 0

    def test_to_dict(self):
        """测试导出字典"""
        node = TestNode(name="export_test")
        node.execute(5)

        d = node.to_dict()
        assert d['name'] == "export_test"
        assert d['class'] == "TestNode"
        assert d['state'] == "success"
        assert d['stats']['execute_count'] == 1
        assert d['stats']['success_count'] == 1

    def test_copy(self):
        """测试复制节点"""
        node1 = TestNode(name="original", config={"foo": "bar"})
        node2 = node1.copy()

        assert node2.name == "original"
        assert node2.config == {"foo": "bar"}
        assert node2 is not node1  # 确保是不同对象

    def test_hooks(self):
        """测试生命周期钩子"""
        hook_calls = []

        class HookTestNode(BaseNode):
            def _execute(self, input_data, **kwargs):
                hook_calls.append(('execute', input_data))
                return input_data

            def before_execute(self, input_data, **kwargs):
                hook_calls.append(('before', input_data))

            def after_execute(self, result, **kwargs):
                hook_calls.append(('after', result))

        node = HookTestNode()
        node.execute(42)

        assert hook_calls == [
            ('before', 42),
            ('execute', 42),
            ('after', 42),
        ]

    def test_disable_stats(self):
        """测试禁用统计"""
        class NoStatsNode(BaseNode):
            _enable_stats = False

            def _execute(self, input_data, **kwargs):
                return input_data

        node = NoStatsNode()
        node.execute(5)
        assert node.stats is None

    def test_none_input(self):
        """测试 None 输入"""
        node = TestNode()
        result = node.execute(None)
        assert result == "ok"

    def test_custom_config(self):
        """测试自定义配置"""
        node = TestNode(config={"key1": "value1", "key2": 123})
        assert node.config == {"key1": "value1", "key2": 123}

    def test_kwargs_config_merge(self):
        """测试配置合并"""
        node = TestNode(config={"a": 1}, b=2, c=3)
        assert node.config == {"a": 1, "b": 2, "c": 3}
