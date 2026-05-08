# coding=utf-8
"""ConfigNode 基类测试"""
import pytest

from QuantNodes.conf_node.base import ConfigNode


class MockConfigNode(ConfigNode):
    """用于测试的 ConfigNode 实现"""

    def __init__(self, config_data=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self._config_data = config_data or {}

    def _load_config(self):
        return self._config_data

    def _get_config_path(self):
        return None


class TestConfigNode:
    """ConfigNode 基类测试"""

    def test_cannot_instantiate_abstract(self):
        """ConfigNode 是抽象类，不能直接实例化"""
        with pytest.raises(TypeError):
            ConfigNode()

    def test_initialization(self):
        """初始化基本属性"""
        node = MockConfigNode(config_data={"key": "value"}, name="TestNode")
        assert node.name == "TestNode"
        assert node.config == {}

    def test_execute_returns_config(self):
        """execute 返回配置字典"""
        node = MockConfigNode(config_data={"key": "value"})
        result = node.execute()
        assert result == {"key": "value"}

    def test_execute_uses_cache(self):
        """execute 默认使用缓存"""
        node = MockConfigNode(config_data={"key": "value"})
        result1 = node.execute()
        result2 = node.execute()
        assert result1 is result2

    def test_execute_no_cache(self):
        """execute 可以禁用缓存"""
        node = MockConfigNode(config_data={"key": "value"})
        result1 = node.execute(use_cache=True)
        result2 = node.execute(use_cache=False)
        assert result1 == result2

    def test_reload(self):
        """reload 强制重新加载"""
        node = MockConfigNode(config_data={"key": "value"})
        result1 = node.execute()
        result2 = node.reload()
        assert result1 == result2

    def test_get_item(self):
        """字典风格访问"""
        node = MockConfigNode(config_data={"key": "value"})
        assert node["key"] == "value"

    def test_get_item_missing(self):
        """访问不存在的 key"""
        node = MockConfigNode(config_data={})
        with pytest.raises(KeyError):
            _ = node["nonexistent"]

    def test_get_method(self):
        """get 方法"""
        node = MockConfigNode(config_data={"key": "value"})
        assert node.get("key") == "value"
        assert node.get("nonexistent", "default") == "default"

    def test_contains(self):
        """in 操作符"""
        node = MockConfigNode(config_data={"key": "value"})
        assert "key" in node
        assert "nonexistent" not in node

    def test_execute_before_get_item(self):
        """在没有调用 execute 的情况下访问 item"""
        node = MockConfigNode(config_data={"key": "value"})
        assert node["key"] == "value"

    def test_name_default(self):
        """默认名称为类名"""
        node = MockConfigNode(config_data={})
        assert node.name == "MockConfigNode"

    def test_name_custom(self):
        """自定义名称"""
        node = MockConfigNode(config_data={}, name="CustomName")
        assert node.name == "CustomName"

    def test_config_kwarg(self):
        """config 参数"""
        node = MockConfigNode(config_data={"a": 1}, config={"b": 2})
        assert node.config == {"b": 2}

    def test_cached_config_after_execute(self):
        """execute 后缓存配置"""
        node = MockConfigNode(config_data={"key": "value"})
        node.execute()
        assert node._cached_config is not None
        assert node._cached_config == {"key": "value"}

    def test_cached_config_none_before_execute(self):
        """execute 前缓存为 None"""
        node = MockConfigNode(config_data={"key": "value"})
        assert node._cached_config is None
