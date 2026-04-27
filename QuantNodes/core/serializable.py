# coding=utf-8
"""
Serializable Mixin 模块

提供统一的序列化/反序列化接口：
- serialize() / deserialize() 统一 API
- @serializable 注册装饰器
- 注册表机制用于反序列化
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Type, TypeVar


T = TypeVar('T')

_REGISTRY: Dict[str, Type] = {}


def serializable(cls: Type[T]) -> Type[T]:
    """
    注册装饰器：标记类为可序列化

    用法：
        @serializable
        class MyClass(Serializable):
            ...

    所有使用此装饰器的类都会注册到全局注册表中，
    支持通过 deserialize() 反序列化。
    """
    _REGISTRY[cls.__name__] = cls
    return cls


class Serializable(ABC):
    """
    可序列化对象的 Mixin 基类

    子类必须实现：
    - _get_serializable_fields(): 返回需要序列化的字段
    - _from_dict_impl(): 从字典反序列化

    提供统一的 API：
    - serialize(): 序列化为字典
    - deserialize(): 从字典反序列化
    """

    _schema_version: str = "1.0"

    @abstractmethod
    def _get_serializable_fields(self) -> Dict[str, Any]:
        """
        子类实现：返回需要序列化的字段

        Returns:
            包含序列化字段的字典，不包含 type 和 _schema_version
        """
        pass

    @classmethod
    @abstractmethod
    def _from_dict_impl(cls, data: Dict[str, Any]) -> Serializable:
        """
        子类实现：从字典反序列化

        Args:
            data: 序列化字典（已包含 type 和 _schema_version）

        Returns:
            反序列化重建的对象
        """
        pass

    def serialize(self) -> Dict[str, Any]:
        """
        统一序列化方法

        Returns:
            包含 type, _schema_version 和子类字段的字典
        """
        return {
            "type": self.__class__.__name__,
            "_schema_version": self._schema_version,
            **self._get_serializable_fields(),
        }

    @classmethod
    def deserialize(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        统一反序列化方法

        Args:
            data: serialize() 返回的字典

        Returns:
            反序列化重建的对象

        Raises:
            ValueError: 缺少 type 或类型未知
        """
        type_name = data.get("type")
        if not type_name:
            raise ValueError("Missing 'type' in serialized data")

        target = _REGISTRY.get(type_name)
        if not target:
            available = list(_REGISTRY.keys())
            raise ValueError(
                f"Unknown serializable type: {type_name}. "
                f"Available types: {available}"
            )

        return target._from_dict_impl(data)
