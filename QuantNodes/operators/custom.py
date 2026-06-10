# coding=utf-8
"""
operators.custom - 用户友好自定义算子 API

提供装饰器风格和 Builder 风格的算子定义接口。
"""

from __future__ import annotations

import inspect
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from polars import Expr

from QuantNodes.operators.registry import _CustomOperatorRegistry
from QuantNodes.operators.templates import OperatorTemplate


class CustomOperatorBuilder:
    """链式构建器"""

    def __init__(self, name: str, category: str):
        self._name = name
        self._category = category
        self._params: Dict[str, dict] = {}
        self._func: Optional[Callable] = None
        self._doc: str = ""
        self._aliases: List[str] = []
        self._decorator_mode: bool = False

    def param(
        self, name: str, type_: Any = None, default: Any = None, desc: str = ""
    ) -> "CustomOperatorBuilder":
        """声明参数"""
        self._params[name] = {"type": type_, "default": default, "desc": desc}
        return self

    def execute(self, func: Callable) -> "CustomOperatorBuilder":
        """设置执行函数"""
        self._func = func
        return self

    def doc(self, docstring: str) -> "CustomOperatorBuilder":
        """设置文档"""
        self._doc = docstring
        return self

    def alias(self, name: str) -> "CustomOperatorBuilder":
        """添加别名"""
        self._aliases.append(name)
        return self

    def register(self) -> Callable:
        """注册并返回被装饰的函数"""
        if self._func is None:
            raise ValueError("execute() must be called before register()")

        return self._do_register(self._func)

    def _do_register(self, func: Callable) -> Callable:
        """执行注册逻辑"""
        if self._func is None:
            self._func = func

        sig = inspect.signature(self._func)

        if not self._doc:
            params_parts = []
            for n, v in self._params.items():
                if v["default"] is not None:
                    params_parts.append(f"{n}={v['default']!r}")
                else:
                    params_parts.append(n)
            params_str = ", ".join(params_parts)
            self._doc = f"{self._name}({params_str})"

        original_func = self._func

        def wrapper(f: Union[Expr, str], **kwargs) -> Expr:
            args = [f]
            for param_name, param_info in self._params.items():
                if param_name not in kwargs and param_info["default"] is not None:
                    kwargs[param_name] = param_info["default"]
            bound = sig.bind(*args, **kwargs)
            return original_func(*bound.args, **bound.kwargs)

        wrapper.__name__ = self._name
        wrapper.__qualname__ = f"CustomOperator.{self._name}"
        wrapper.__doc__ = self._doc

        _CustomOperatorRegistry.register(
            self._category,
            self._name,
            wrapper,
            self._doc,
            self._params,
            self._aliases,
        )

        for alias in self._aliases:
            _CustomOperatorRegistry.register_alias(alias, self._name, self._category)

        return wrapper

    def __call__(self, func: Callable) -> Callable:
        """当作为装饰器使用时直接注册"""
        return self._do_register(func)


class CustomOperator:
    """自定义算子工厂（类方法集合）"""

    @classmethod
    def point(cls, name: str) -> CustomOperatorBuilder:
        """创建 point 算子构建器"""
        return CustomOperatorBuilder(name, "point")

    @classmethod
    def time(cls, name: str) -> CustomOperatorBuilder:
        """创建 time 算子构建器"""
        return CustomOperatorBuilder(name, "time")

    @classmethod
    def section(cls, name: str) -> CustomOperatorBuilder:
        """创建 section 算子构建器"""
        return CustomOperatorBuilder(name, "section")

    @classmethod
    def multi_section(cls, name: str) -> CustomOperatorBuilder:
        """创建 multi_section 算子构建器"""
        return CustomOperatorBuilder(name, "multi_section")

    @classmethod
    def talib(cls, name: str) -> CustomOperatorBuilder:
        """创建 talib 算子构建器"""
        return CustomOperatorBuilder(name, "talib")

    @classmethod
    def from_template(
        cls, name: str, template: str, category: str = None, **defaults
    ) -> OperatorTemplate:
        """基于模板创建算子"""
        return OperatorTemplate(name, category or "time", template, defaults)

    @classmethod
    def time_from(cls, name: str, template: str, **defaults) -> OperatorTemplate:
        """基于 time 模板创建算子"""
        return OperatorTemplate(name, "time", template, defaults)

    @classmethod
    def section_from(
        cls, name: str, template: str, **defaults
    ) -> OperatorTemplate:
        """基于 section 模板创建算子"""
        return OperatorTemplate(name, "section", template, defaults)

    @classmethod
    def point_from(
        cls, name: str, template: str, **defaults
    ) -> OperatorTemplate:
        """基于 point 模板创建算子"""
        return OperatorTemplate(name, "point", template, defaults)

    @classmethod
    def list(
        cls, category: Optional[str] = None, include_builtin: bool = False
    ) -> List[str]:
        """列出算子"""
        custom = _CustomOperatorRegistry.list(category)
        if not include_builtin:
            return custom
        from QuantNodes.factor_node.factor_functions import list_operators

        builtin = list_operators(category)
        seen = set()
        result = []
        for name in custom + builtin:
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result

    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        """获取算子函数（先查自定义，再查内置）"""
        func = _CustomOperatorRegistry.get(name)
        if func:
            return func
        from QuantNodes.factor_node.factor_functions import get_operator

        return get_operator(name)

    @classmethod
    def info(cls, name: str) -> Optional[Dict[str, Any]]:
        """获取算子详细信息"""
        info = _CustomOperatorRegistry.info(name)
        if info:
            return info
        from QuantNodes.factor_node.factor_functions import operator_info

        return operator_info(name)

    @classmethod
    def unregister(cls, name: str) -> bool:
        """注销自定义算子"""
        return _CustomOperatorRegistry.unregister(name)

    @classmethod
    def unregister_all(cls) -> int:
        """注销所有自定义算子"""
        return _CustomOperatorRegistry.unregister_all()

    @classmethod
    def register(
        cls, category: str, name: str, func: Callable, **kwargs
    ) -> Callable:
        """
        直接注册自定义算子的便捷方法
        
        Args:
            category: 算子类别 ("point", "time", "section", "multi_section", "talib")
            name: 算子名称
            func: 算子函数
            **kwargs: 额外参数
        
        Returns:
            注册后的函数
        
        Example:
            >>> def my_double(f, multiplier=2.0):
            ...     return f * multiplier
            >>> CustomOperator.register("point", "my_double", my_double)
        """
        _CustomOperatorRegistry.register(category, name, func, **kwargs)
        return func

    @classmethod
    def count(cls) -> int:
        """返回自定义算子数量"""
        return _CustomOperatorRegistry.count()

    @classmethod
    def export(cls, path: str, format: str = "yaml") -> None:
        """导出到 YAML/JSON"""
        data = {
            "version": 1,
            "exported_at": datetime.now().isoformat(),
            "operators": [],
        }

        for cat in _CustomOperatorRegistry._registry:
            for name, info in _CustomOperatorRegistry._registry[cat].items():
                op_data = {
                    "name": name,
                    "category": cat,
                    "doc": info.get("doc", ""),
                    "source": info.get("source", ""),
                }
                params = info.get("params", {})
                if params:
                    op_data["params"] = params
                data["operators"].append(op_data)

        if format == "yaml":
            import yaml

            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, sort_keys=False)
        else:
            import json

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def import_(cls, path: str) -> int:
        """从 YAML/JSON 导入，返回导入数量"""
        import yaml

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        count = 0
        for op in data.get("operators", []):
            source = op.get("source", "")
            if source:
                namespace = {}
                exec(source, namespace)
                func = namespace.get(op["name"])
                if func:
                    _CustomOperatorRegistry.register(
                        op["category"],
                        op["name"],
                        func,
                        op.get("doc", ""),
                        op.get("params", {}),
                    )
                    count += 1
        return count


def _make_decorator(category: str):
    """创建分类装饰器"""

    def decorator(name: str):
        def wrapper(func: Callable) -> Callable:
            _CustomOperatorRegistry.register(
                category,
                name,
                func,
                inspect.getdoc(func) or "",
            )

            def _wrapper(f: Union[Expr, str], **kwargs) -> Expr:
                return func(f, **kwargs)

            _wrapper.__name__ = func.__name__
            _wrapper.__doc__ = func.__doc__
            return _wrapper

        return decorator(name) if callable(name) else lambda f: decorator(name)(f)

    return decorator


def _register_point(name: str) -> Callable:
    """point 算子装饰器"""

    def decorator(func: Callable) -> Callable:
        _CustomOperatorRegistry.register(
            "point",
            name,
            func,
            inspect.getdoc(func) or "",
        )

        def _wrapper(f: Union[Expr, str], **kwargs) -> Expr:
            return func(f, **kwargs)

        _wrapper.__name__ = func.__name__
        _wrapper.__doc__ = func.__doc__
        return _wrapper

    return decorator


def point(name: str) -> Callable:
    """point 算子装饰器"""

    def decorator(func: Callable) -> Callable:
        _CustomOperatorRegistry.register(
            "point",
            name,
            func,
            inspect.getdoc(func) or "",
        )

        def _wrapper(f: Union[Expr, str], **kwargs) -> Expr:
            return func(f, **kwargs)

        _wrapper.__name__ = func.__name__
        _wrapper.__doc__ = func.__doc__
        return _wrapper

    return decorator


def time(name: str) -> Callable:
    """time 算子装饰器"""

    def decorator(func: Callable) -> Callable:
        _CustomOperatorRegistry.register(
            "time",
            name,
            func,
            inspect.getdoc(func) or "",
        )

        def _wrapper(f: Union[Expr, str], **kwargs) -> Expr:
            return func(f, **kwargs)

        _wrapper.__name__ = func.__name__
        _wrapper.__doc__ = func.__doc__
        return _wrapper

    return decorator


def section(name: str) -> Callable:
    """section 算子装饰器"""

    def decorator(func: Callable) -> Callable:
        _CustomOperatorRegistry.register(
            "section",
            name,
            func,
            inspect.getdoc(func) or "",
        )

        def _wrapper(f: Union[Expr, str], **kwargs) -> Expr:
            return func(f, **kwargs)

        _wrapper.__name__ = func.__name__
        _wrapper.__doc__ = func.__doc__
        return _wrapper

    return decorator
