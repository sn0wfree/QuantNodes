# coding=utf-8
"""
operators.templates - 算子模板工厂

基于现有算子创建固定参数变体。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from polars import Expr


class OperatorTemplate:
    """基于现有算子创建变体的模板"""

    def __init__(
        self,
        name: str,
        category: str,
        template: str,
        defaults: Optional[Dict[str, Any]] = None,
    ):
        """初始化模板

        Args:
            name: 新算子名称
            category: 算子分类 (point, time, section, multi_section, talib)
            template: 模板算子名称
            defaults: 默认参数字典
        """
        self._name = name
        self._category = category
        self._template = template
        self._defaults = defaults or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def category(self) -> str:
        return self._category

    @property
    def template(self) -> str:
        return self._template

    @property
    def defaults(self) -> Dict[str, Any]:
        return self._defaults.copy()

    def __call__(self, f: Union[Expr, str], **kwargs) -> Expr:
        """调用时执行模板算子，自动注入默认值"""
        from QuantNodes.factor_node.factor_functions import get_operator

        template_func = get_operator(self._template, self._category)
        if template_func is None:
            raise ValueError(
                f"Template operator '{self._template}' not found in category '{self._category}'"
            )

        merged_kwargs = {**self._defaults, **kwargs}
        return template_func(f, **merged_kwargs)

    def register(self) -> "OperatorTemplate":
        """将模板注册为新的命名算子"""
        from QuantNodes.operators.registry import _CustomOperatorRegistry

        def _template_wrapper(f: Union[Expr, str], **kwargs) -> Expr:
            return self(f, **kwargs)

        _template_wrapper.__name__ = self._name
        _template_wrapper.__qualname__ = f"OperatorTemplate.{self._name}"
        _template_wrapper.__doc__ = (
            f"基于 {self._template} 的模板算子，默认参数: {self._defaults}"
        )

        _CustomOperatorRegistry.register(
            self._category,
            self._name,
            _template_wrapper,
            doc=_template_wrapper.__doc__,
            params={"template": self._template, "defaults": self._defaults},
        )

        return self

    def __repr__(self) -> str:
        return (
            f"OperatorTemplate(name={self._name!r}, "
            f"template={self._template!r}, "
            f"defaults={self._defaults})"
        )
