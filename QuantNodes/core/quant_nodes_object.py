# coding=utf-8
"""
QuantNodes 对象基类

替代 QuantStudio.__QS_Object__，继承自 traits.HasTraits
"""

import logging
from typing import Any, Dict, Optional

from traits.api import HasTraits, Str


class QuantNodesObject(HasTraits):
    """
    QuantNodes 基础对象类

    继承自 traits.HasTraits，提供配置属性系统

    Attributes:
        Name: 对象名称
        _QS_Logger: 日志记录器
    """

    Name = Str("QuantNodes对象")

    _QS_Logger = logging.getLogger("QuantNodes")

    def __init__(
        self,
        sys_args: Optional[Dict[str, Any]] = None,
        config_file: Optional[str] = None,
        **kwargs,
    ):
        """
        初始化 QuantNodesObject

        Args:
            sys_args: 系统参数字典，用于配置对象属性
            config_file: 配置文件路径（暂未实现）
            **kwargs: 其他关键字参数
        """
        super().__init__(**kwargs)
        self._init_logger()
        self.__QS_initArgs__(sys_args=sys_args or {})

    def _init_logger(self) -> None:
        """初始化日志记录器"""
        self._QS_Logger = logging.getLogger(f"QuantNodes.{self.__class__.__name__}")

    def __QS_initArgs__(self, sys_args: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化配置参数

        子类应重写此方法以初始化自己的配置

        Args:
            sys_args: 系统参数字典
        """
        pass

    def get_trait(self, name: str) -> Any:
        """
        获取 trait 属性值

        Args:
            name: 属性名

        Returns:
            属性值

        Raises:
            AttributeError: 属性不存在
        """
        if self.trait(name) is None:
            raise AttributeError(f"'{self.__class__.__name__}' 对象没有属性 '{name}'")
        return getattr(self, name)

    def set_trait(self, name: str, value: Any) -> None:
        """
        设置 trait 属性值

        Args:
            name: 属性名
            value: 属性值
        """
        if self.trait(name) is None:
            raise AttributeError(f"'{self.__class__.__name__}' 对象没有属性 '{name}'")
        setattr(self, name, value)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.Name}>"

    def __str__(self) -> str:
        return self.__repr__()
