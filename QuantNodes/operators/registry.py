# coding=utf-8
"""
operators.registry - 用户自定义算子隔离注册表

用户通过 CustomOperator 注册的算子存储在此注册表，
与内置 _OPERATOR_REGISTRY 完全隔离。
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple


class _CustomOperatorRegistry:
    """用户自定义算子隔离注册表"""

    _registry: Dict[str, Dict[str, Dict[str, Any]]] = {
        "point": {},
        "time": {},
        "section": {},
        "multi_section": {},
        "talib": {},
    }
    _aliases: Dict[str, Tuple[str, str]] = {}

    @classmethod
    def register(
        cls,
        category: str,
        name: str,
        func: Callable,
        doc: str = "",
        params: Optional[Dict[str, Any]] = None,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """注册算子到隔离注册表"""
        try:
            source = inspect.getsource(func)
        except (OSError, TypeError):
            source = ""

        cls._registry.setdefault(category, {})
        cls._registry[category][name] = {
            "name": name,
            "category": category,
            "func": func,
            "doc": doc or inspect.getdoc(func) or "",
            "params": params or {},
            "source": source,
        }

        for alias in (aliases or []):
            cls._aliases[alias] = (name, category)

    @classmethod
    def register_alias(cls, alias: str, name: str, category: str) -> None:
        """注册别名"""
        cls._aliases[alias] = (name, category)

    @classmethod
    def get(cls, name: str, category: Optional[str] = None) -> Optional[Callable]:
        """获取算子函数（支持别名）"""
        if name in cls._aliases:
            name, category = cls._aliases[name]

        if category:
            return cls._registry.get(category, {}).get(name, {}).get("func")

        for cat in cls._registry:
            if name in cls._registry[cat]:
                return cls._registry[cat][name]["func"]
        return None

    @classmethod
    def list(cls, category: Optional[str] = None) -> List[str]:
        """列出算子名称"""
        if category:
            return list(cls._registry.get(category, {}).keys())
        return [name for cat in cls._registry for name in cls._registry[cat]]

    @classmethod
    def info(cls, name: str) -> Optional[Dict[str, Any]]:
        """获取算子详细信息"""
        if name in cls._aliases:
            name, category = cls._aliases[name]
        for cat in cls._registry:
            if name in cls._registry[cat]:
                return cls._registry[cat][name]
        return None

    @classmethod
    def unregister(cls, name: str, category: Optional[str] = None) -> bool:
        """注销算子"""
        if name in cls._aliases:
            _, cat = cls._aliases.pop(name)
            cls._registry.get(cat, {}).pop(name, None)
            return True

        if category:
            removed = cls._registry.get(category, {}).pop(name, None)
            return removed is not None

        for cat in cls._registry:
            if cls._registry[cat].pop(name, None):
                return True
        return False

    @classmethod
    def unregister_all(cls) -> int:
        """注销所有自定义算子，返回数量"""
        count = 0
        for cat in cls._registry:
            count += len(cls._registry[cat])
            cls._registry[cat].clear()
        cls._aliases.clear()
        return count

    @classmethod
    def export_dict(cls) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """导出为字典"""
        return cls._registry

    @classmethod
    def import_dict(cls, data: Dict[str, Dict[str, Dict[str, Any]]]) -> int:
        """从字典导入，返回导入数量"""
        count = 0
        for category, operators in data.items():
            for name, info in operators.items():
                if "func" in info:
                    cls.register(
                        category,
                        name,
                        info["func"],
                        info.get("doc", ""),
                        info.get("params", {}),
                    )
                    count += 1
        return count

    @classmethod
    def count(cls) -> int:
        """返回注册的自定义算子数量"""
        return len(cls.list())
