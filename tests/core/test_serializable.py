# coding=utf-8
"""QuantNodes.core.serializable 单元测试"""
import pytest

from QuantNodes.core.serializable import serializable, Serializable, _REGISTRY


class ConcreteSerializable(Serializable):
    def __init__(self, name: str = "", value: int = 0):
        self.name = name
        self.value = value

    def _get_serializable_fields(self):
        return {"name": self.name, "value": self.value}

    @classmethod
    def _from_dict_impl(cls, data):
        return cls(name=data["name"], value=data["value"])


@serializable
class DecoratedSerializable(Serializable):
    def __init__(self, x: int = 0):
        self.x = x

    def _get_serializable_fields(self):
        return {"x": self.x}

    @classmethod
    def _from_dict_impl(cls, data):
        return cls(x=data["x"])


class TestSerializableDecorator:
    def test_registers_class(self):
        assert "DecoratedSerializable" in _REGISTRY

    def test_class_preserved(self):
        assert DecoratedSerializable._schema_version == "1.0"


class TestSerializableSerialize:
    def test_output_format(self):
        obj = ConcreteSerializable(name="test", value=42)
        data = obj.serialize()
        assert data["type"] == "ConcreteSerializable"
        assert data["_schema_version"] == "1.0"
        assert data["name"] == "test"
        assert data["value"] == 42

    def test_schema_version(self):
        obj = ConcreteSerializable()
        data = obj.serialize()
        assert data["_schema_version"] == "1.0"


class TestSerializableDeserialize:
    def test_missing_type(self):
        with pytest.raises(ValueError, match="Missing 'type'"):
            Serializable.deserialize({"data": 123})

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown serializable type"):
            Serializable.deserialize({"type": "NonExistent"})

    def test_roundtrip(self):
        original = DecoratedSerializable(x=99)
        data = original.serialize()
        restored = Serializable.deserialize(data)
        assert isinstance(restored, DecoratedSerializable)
        assert restored.x == 99


class TestConcreteSerializable:
    def test_roundtrip(self):
        original = ConcreteSerializable(name="hello", value=7)
        data = original.serialize()
        restored = ConcreteSerializable._from_dict_impl(data)
        assert restored.name == "hello"
        assert restored.value == 7


class TestSchemaVersion:
    def test_custom_version(self):
        class V2Serializable(Serializable):
            _schema_version = "2.0"

            def _get_serializable_fields(self):
                return {}

            @classmethod
            def _from_dict_impl(cls, data):
                return cls()

        obj = V2Serializable()
        data = obj.serialize()
        assert data["_schema_version"] == "2.0"
