# coding=utf-8
"""
core/plugin.py — 插件发现机制 (Tier 0: Foundation)

通过 Python 标准 entry_points 机制发现第三方 quantnodes 插件。

设计目标:
- 第三方包可在自己的 pyproject.toml 中声明:
    [project.entry-points."quantnodes.tools"]
    my_tool = "my_pkg.tools:MyTool"
- 运行时通过 discover_tools() / discover_operators() 自动加载
- 与 nanobot upstream 风格一致 (其已用 entry_points)
- 向后兼容: 未声明 entry_points 时回退到硬编码列表

回退策略:
- discover_*() 返回 dict (name -> factory)
- 如 entry_points 完全为空, 返回空 dict (调用方需 fallback 到硬编码)
- 如 entry_points 部分加载失败, 记录 warning 但继续

使用示例:
    from QuantNodes.core.plugin import discover_tools, discover_operators

    # 发现所有工具 (含第三方插件)
    tools = discover_tools()
    for name, factory in tools.items():
        tool = factory()
        registry.register(tool)

    # 发现所有算子 (自定义算子可通过 entry_points 注册)
    ops = discover_operators()
    for op_name in ops:
        # op_name 是已注册的算子名
        ...
"""

from __future__ import annotations

import importlib.metadata as md
import logging
from typing import Any, Callable, Dict, List, Type

logger = logging.getLogger(__name__)


# entry_points 组名常量
TOOLS_GROUP = "quantnodes.tools"
OPERATORS_GROUP = "quantnodes.operators"


def discover_tools() -> Dict[str, Type]:
    """通过 entry_points 发现所有 quantnodes.tools 插件

    Returns:
        dict: {entry_name: Tool class}，可调用 class() 实例化。
              如 entry_points 为空，返回空 dict。
              如部分插件加载失败，跳过并记录 warning。

    Example:
        >>> tools = discover_tools()
        >>> SandboxTool = tools["sandbox"]
        >>> tool = SandboxTool()
    """
    result: Dict[str, Type] = {}
    eps = md.entry_points(group=TOOLS_GROUP)
    for ep in eps:
        try:
            cls = ep.load()
            result[ep.name] = cls
        except Exception as e:
            logger.warning(
                "Failed to load tool plugin %r from %r: %s",
                ep.name,
                ep.value,
                e,
            )
    return result


def discover_operators() -> List[str]:
    """通过 entry_points 发现所有 quantnodes.operators 插件名

    Returns:
        list[str]: 已注册的算子名列表。
                   如 entry_points 为空，返回空 list。

    Note:
        算子 plugin 的 entry value 应该是形如 "my_pkg.ops:get_op_names" 的可调用对象，
        调用后返回该包注册的算子名列表。
    """
    result: List[str] = []
    eps = md.entry_points(group=OPERATORS_GROUP)
    for ep in eps:
        try:
            loader = ep.load()
            op_names = loader()
            if isinstance(op_names, (list, tuple)):
                result.extend(str(n) for n in op_names)
            else:
                logger.warning(
                    "Operator plugin %r did not return a list: got %r",
                    ep.name,
                    type(op_names),
                )
        except Exception as e:
            logger.warning(
                "Failed to load operator plugin %r from %r: %s",
                ep.name,
                ep.value,
                e,
            )
    return result


def discover_all() -> Dict[str, Dict[str, Any]]:
    """一次性发现所有 quantnodes.* 插件

    Returns:
        dict: {
            "tools": {name: cls, ...},
            "operators": [name, ...],
        }
    """
    return {
        "tools": discover_tools(),
        "operators": discover_operators(),
    }


__all__ = [
    "TOOLS_GROUP",
    "OPERATORS_GROUP",
    "discover_tools",
    "discover_operators",
    "discover_all",
]