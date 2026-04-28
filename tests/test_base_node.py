# -*- coding: utf-8 -*-
"""BaseNode unit tests"""
import pytest
from QuantNodes.core.node import BaseNode, NodeState, NodeStats, NodeExecutionError


class EchoNode(BaseNode):
    """Simple test node that returns input"""
    def _execute(self, input_data=None, **kwargs):
        return input_data


class DoubleNode(BaseNode):
    """Test node that doubles input"""
    def _execute(self, input_data=None, **kwargs):
        if isinstance(input_data, (int, float)):
            return input_data * 2
        return input_data


class AddNode(BaseNode):
    """Test node that adds config['value'] to input"""
    def _execute(self, input_data=None, **kwargs):
        if isinstance(input_data, (int, float)):
            return input_data + self.config.get('value', 0)
        return input_data


class FailingNode(BaseNode):
    """Test node that always raises an exception"""
    def _execute(self, input_data=None, **kwargs):
        raise ValueError("Intentional test failure")


class TestNodeState:
    """Tests for NodeState enum"""
    def test_node_state_values(self):
        assert NodeState.IDLE.value == "idle"
        assert NodeState.RUNNING.value == "running"
        assert NodeState.SUCCESS.value == "success"
        assert NodeState.FAILED.value == "failed"

    def test_node_state_is_string_enum(self):
        assert isinstance(NodeState.IDLE, str)


class TestNodeStats:
    """Tests for NodeStats class"""
    def test_stats_initialization(self):
        stats = NodeStats()
        assert stats.execute_count == 0
        assert stats.success_count == 0
        assert stats.failed_count == 0
        assert stats.total_time_ms == 0.0
        assert stats.avg_time_ms == 0.0
        assert stats.last_execute_at is None

    def test_stats_update_success(self):
        stats = NodeStats()
        stats.update(10.0, success=True)
        assert stats.execute_count == 1
        assert stats.success_count == 1
        assert stats.failed_count == 0
        assert stats.total_time_ms == 10.0
        assert stats.avg_time_ms == 10.0

    def test_stats_update_failure(self):
        stats = NodeStats()
        stats.update(5.0, success=False)
        assert stats.execute_count == 1
        assert stats.success_count == 0
        assert stats.failed_count == 1

    def test_stats_multiple_updates(self):
        stats = NodeStats()
        stats.update(10.0, success=True)
        stats.update(20.0, success=True)
        stats.update(5.0, success=False)
        assert stats.execute_count == 3
        assert stats.success_count == 2
        assert stats.failed_count == 1
        assert stats.total_time_ms == 35.0
        assert stats.avg_time_ms == pytest.approx(11.67, rel=0.01)

    def test_stats_to_dict(self):
        stats = NodeStats()
        stats.update(10.0, success=True)
        d = stats.to_dict()
        assert d['execute_count'] == 1
        assert d['success_count'] == 1
        assert d['failed_count'] == 0
        assert d['total_time_ms'] == 10.0


class TestBaseNode:
    """Tests for BaseNode class"""
    def test_node_initialization_default(self):
        node = EchoNode()
        assert node.name == "EchoNode"
        assert node.state == NodeState.IDLE
        assert node.node_id.startswith("EchoNode_")
        assert len(node.node_id.split("_")[-1]) == 8

    def test_node_initialization_with_name(self):
        node = EchoNode(name="CustomName")
        assert node.name == "CustomName"

    def test_node_initialization_with_config(self):
        node = AddNode(config={'value': 10})
        assert node.config['value'] == 10

    def test_node_initialization_with_kwargs(self):
        node = AddNode(value=5)
        assert node.config['value'] == 5

    def test_node_initialization_config_merge(self):
        node = AddNode(config={'value': 10}, value=5)
        assert node.config['value'] == 5

    def test_node_execute_simple(self):
        node = EchoNode()
        result = node.execute("test_data")
        assert result == "test_data"
        assert node.state == NodeState.SUCCESS

    def test_node_execute_with_input(self):
        node = EchoNode()
        result = node.execute(input_data=[1, 2, 3])
        assert result == [1, 2, 3]

    def test_node_callable(self):
        node = EchoNode()
        result = node([1, 2, 3])
        assert result == [1, 2, 3]

    def test_node_state_transitions(self):
        node = DoubleNode()
        assert node.state == NodeState.IDLE
        node.execute(5)
        assert node.state == NodeState.SUCCESS
        assert node._last_result == 10

    def test_node_failed_state(self):
        node = FailingNode()
        with pytest.raises(NodeExecutionError):
            node.execute("test")
        assert node.state == NodeState.FAILED
        assert node._last_error is not None

    def test_node_stats_tracking(self):
        node = DoubleNode()
        node.execute(5)
        assert node.stats is not None
        assert node.stats.execute_count == 1
        assert node.stats.success_count == 1

    def test_node_stats_tracking_failure(self):
        node = FailingNode()
        try:
            node.execute("test")
        except NodeExecutionError:
            pass
        assert node.stats.failed_count == 1

    def test_node_reset(self):
        node = DoubleNode()
        node.execute(5)
        node.reset()
        assert node.state == NodeState.IDLE
        assert node._last_result is None
        assert node._last_error is None
        assert node.stats.execute_count == 0

    def test_node_copy(self):
        node = AddNode(config={'value': 10})
        copy = node.copy()
        assert copy.name == node.name
        assert copy.config == node.config
        assert copy.node_id != node.node_id

    def test_node_to_info(self):
        node = EchoNode(name="TestNode")
        info = node.to_info()
        assert info['node_id'] == node.node_id
        assert info['name'] == "TestNode"
        assert info['class'] == "EchoNode"
        assert info['state'] == "idle"

    def test_node_pipeline_operator(self):
        """Test >> operator creates Pipeline"""
        node1 = EchoNode(name="N1")
        node2 = DoubleNode(name="N2")
        pipeline = node1 >> node2
        assert pipeline.__class__.__name__ == "Pipeline"
        assert len(pipeline.nodes) == 2

    def test_node_equality(self):
        node1 = EchoNode(name="Test")
        node2 = EchoNode(name="Test")
        assert node1.node_id != node2.node_id

    def test_node_without_stats(self):
        class NoStatsNode(BaseNode):
            _enable_stats = False
            def _execute(self, input_data=None, **kwargs):
                return input_data
        
        node = NoStatsNode()
        assert node.stats is None


class TestBaseNodeHooks:
    """Tests for BaseNode hooks"""
    def test_before_execute_hook(self):
        class HookNode(EchoNode):
            def before_execute(self, input_data, **kwargs):
                self._hook_called = True
        
        node = HookNode()
        node.execute("test")
        assert node._hook_called is True

    def test_after_execute_hook(self):
        class HookNode(EchoNode):
            def after_execute(self, result, **kwargs):
                self._hook_result = result
        
        node = HookNode()
        result = node.execute("test")
        assert node._hook_result == "test"

    def test_hooks_disabled(self):
        class HookNode(EchoNode):
            _enable_hooks = False
            def before_execute(self, input_data, **kwargs):
                raise AssertionError("Hook should not be called")
        
        node = HookNode()
        node.execute("test")


class TestBaseNodeValidation:
    """Tests for BaseNode validation"""
    def test_validation_disabled(self):
        class NoValidationNode(BaseNode):
            _enable_validation = False
            def _execute(self, input_data=None, **kwargs):
                return input_data
        
        node = NoValidationNode()
        result = node.execute("test")
        assert result == "test"


class TestBaseNodeSerialization:
    """Tests for BaseNode serialization
    
    Note: Direct BaseNode serialization requires @serializable decorator
    and proper registration. See test_pipeline.py for serialization tests
    using properly registered types.
    """


class TestNodeErrors:
    """Tests for Node error handling"""
    def test_error_propagation(self):
        node = FailingNode()
        with pytest.raises(NodeExecutionError) as exc_info:
            node.execute("test")
        assert "Intentional test failure" in str(exc_info.value)

    def test_last_error_stored(self):
        node = FailingNode()
        try:
            node.execute("test")
        except NodeExecutionError:
            pass
        assert isinstance(node._last_error, ValueError)

    def test_last_input_stored(self):
        node = DoubleNode()
        node.execute(5)
        assert node._last_input == 5
