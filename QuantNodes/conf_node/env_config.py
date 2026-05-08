# coding=utf-8
"""
EnvConfigNode - 环境变量配置节点
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from QuantNodes.conf_node.base import ConfigNode


class EnvConfigNode(ConfigNode):
    """
    环境变量配置节点

    从环境变量读取配置，支持前缀过滤和类型转换。

    Examples:
        >>> # 读取所有环境变量
        >>> node = EnvConfigNode()
        >>> config = node.execute()
        >>>
        >>> # 只读取以 DB_ 开头的环境变量
        >>> node = EnvConfigNode(prefix="DB_")
        >>> db_config = node.execute()  # {'HOST': 'localhost', 'PORT': '5432', ...}
        >>>
        >>> # 带前缀和类型转换
        >>> node = EnvConfigNode(prefix="DB_", types={'PORT': int, 'DEBUG': bool})
        >>> config = node.execute()  # {'PORT': 5432, 'DEBUG': True, ...}
    """

    def __init__(
        self,
        prefix: Optional[str] = None,
        separator: str = '_',
        types: Optional[Dict[str, type]] = None,
        lowercase_keys: bool = True,
        name: str = None,
        config: Dict[str, Any] = None,
        **kwargs
    ):
        """
        Args:
            prefix: 环境变量前缀，只读取以此前缀开头的变量
            separator: 前缀与变量名之间的分隔符，默认 '_'
            types: 类型转换字典，如 {'PORT': int, 'DEBUG': bool}
            lowercase_keys: 是否将 key 转为小写，默认 True
            name: 节点名称
            config: 额外配置
            **kwargs: 额外参数
        """
        super().__init__(name=name, config=config, **kwargs)
        self.prefix = prefix
        self.separator = separator
        self.types = types or {}
        self.lowercase_keys = lowercase_keys

    def _get_config_path(self) -> Optional[Path]:
        return None

    def _load_config(self) -> Dict[str, Any]:
        """加载环境变量"""
        result = {}

        for key, value in os.environ.items():
            if self.prefix:
                if not key.startswith(self.prefix):
                    continue
                config_key = key[len(self.prefix):]
            else:
                config_key = key

            if not config_key:
                continue

            if self.lowercase_keys:
                config_key = config_key.lower()

            if config_key in self.types:
                value = self._convert_type(config_key, value)

            result[config_key] = value

        return result

    def _convert_type(self, key: str, value: str) -> Any:
        """类型转换"""
        target_type = self.types.get(key)

        if target_type is bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif target_type is int:
            try:
                return int(value)
            except ValueError:
                return value
        elif target_type is float:
            try:
                return float(value)
            except ValueError:
                return value
        else:
            return target_type(value) if target_type else value

    def get(self, key: str, default: Any = None) -> Any:
        """获取环境变量配置"""
        return super().get(key, default)

    @classmethod
    def from_env(cls, key: str, default: Any = None, target_type: type = None) -> Any:
        """
        便捷方法：从环境变量读取单个值

        Args:
            key: 环境变量名
            default: 默认值
            target_type: 目标类型

        Returns:
            环境变量值
        """
        value = os.environ.get(key, default)
        if value is None:
            return default

        if target_type is bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif target_type in (int, float, str):
            return target_type(value)
        return value
