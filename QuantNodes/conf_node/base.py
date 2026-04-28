# coding=utf-8
"""
ConfigNode 配置节点基类模块

提供配置节点的基础架构，继承自 BaseNode。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pathlib import Path

from QuantNodes.core.node import BaseNode


class ConfigNode(BaseNode, ABC):
    """
    配置节点基类

    所有配置节点都继承自此类，提供统一的数据读取接口。

    子类必须实现：
        _load_config(): 从配置源加载配置数据
        _get_config_path(): 返回配置路径

    Examples:
        >>> ini = IniConfigNode(file_path="config.ini", section="database")
        >>> settings = ini.execute()
    """

    def __init__(self, name: str = None, config: Dict[str, Any] = None, **kwargs):
        super().__init__(name=name or self.__class__.__name__, config=config, **kwargs)
        self._cached_config: Optional[Dict[str, Any]] = None

    @abstractmethod
    def _load_config(self) -> Dict[str, Any]:
        """
        从配置源加载配置数据

        Returns:
            配置字典
        """
        pass

    @abstractmethod
    def _get_config_path(self) -> Optional[Path]:
        """
        返回配置路径

        Returns:
            配置文件的 Path 对象，如果没有文件概念则返回 None
        """
        pass

    def _execute(self, input_data: Any = None, **kwargs) -> Dict[str, Any]:
        """
        执行配置加载

        Args:
            input_data: 可选的输入数据（会被忽略）

        Returns:
            配置字典
        """
        self._cached_config = self._load_config()
        return self._cached_config

    def execute(self, input_data: Any = None, *, use_cache: bool = True, **kwargs) -> Dict[str, Any]:
        """
        执行配置加载

        Args:
            input_data: 可选的输入数据（会被忽略）
            use_cache: 是否使用缓存，默认 True

        Returns:
            配置字典
        """
        if use_cache and self._cached_config is not None:
            return self._cached_config
        return super().execute(input_data, **kwargs)

    def reload(self) -> Dict[str, Any]:
        """
        强制重新加载配置

        Returns:
            重新加载的配置字典
        """
        self._cached_config = None
        return self.execute(use_cache=False)

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值，如果不存在则返回默认值
        """
        if self._cached_config is None:
            self.execute()
        return self._cached_config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """支持 dict 风格的配置访问"""
        if self._cached_config is None:
            self.execute()
        return self._cached_config[key]

    def __contains__(self, key: str) -> bool:
        """支持 'in' 操作符"""
        if self._cached_config is None:
            self.execute()
        return key in self._cached_config
