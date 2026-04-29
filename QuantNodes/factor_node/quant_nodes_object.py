# coding=utf-8
"""
QuantNodes 对象基类

使用 dataclass 替代 traits.HasTraits
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class QuantNodesObject:
    """
    QuantNodes 基础对象类
    
    使用 dataclass 提供配置属性系统
    
    Attributes:
        name: 对象名称
        config: 配置字典
    """

    name: str = "QuantNodesObject"
    config: Dict[str, Any] = field(default_factory=dict)
    _logger: logging.Logger = field(default=None, repr=False)

    def __post_init__(self):
        if self._logger is None:
            self._logger = logging.getLogger(f"QuantNodes.{self.__class__.__name__}")

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键
            default: 默认值
        
        Returns:
            配置值
        """
        return self.config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
        self.config[key] = value

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"

    def __str__(self) -> str:
        return self.__repr__()
