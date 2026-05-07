# coding=utf-8
"""QuantNodes.factor_node.quant_nodes_object 单元测试"""

from QuantNodes.factor_node.quant_nodes_object import QuantNodesObject


class TestQuantNodesObject:
    def test_creation_with_default_values(self):
        obj = QuantNodesObject()
        assert obj.name == "QuantNodesObject"
        assert obj.config == {}
        assert obj._logger is not None

    def test_creation_with_name(self):
        obj = QuantNodesObject(name="TestObject")
        assert obj.name == "TestObject"
        assert obj.config == {}

    def test_creation_with_config(self):
        obj = QuantNodesObject(name="Test", config={"key": "value"})
        assert obj.name == "Test"
        assert obj.config == {"key": "value"}

    def test_get_config_returns_value(self):
        obj = QuantNodesObject(config={"key": "value"})
        assert obj.get_config("key") == "value"

    def test_get_config_with_default(self):
        obj = QuantNodesObject()
        assert obj.get_config("nonexistent", "default") == "default"

    def test_get_config_returns_none_for_missing_key(self):
        obj = QuantNodesObject()
        assert obj.get_config("nonexistent") is None

    def test_set_config(self):
        obj = QuantNodesObject()
        obj.set_config("new_key", "new_value")
        assert obj.get_config("new_key") == "new_value"

    def test_set_config_overwrites_existing(self):
        obj = QuantNodesObject(config={"key": "old"})
        obj.set_config("key", "new")
        assert obj.get_config("key") == "new"

    def test_repr(self):
        obj = QuantNodesObject(name="TestObj")
        assert repr(obj) == "<QuantNodesObject: TestObj>"

    def test_str(self):
        obj = QuantNodesObject(name="TestObj")
        assert str(obj) == "<QuantNodesObject: TestObj>"

    def test_logger_is_named_after_class(self):
        obj = QuantNodesObject()
        assert "QuantNodesObject" in obj._logger.name
