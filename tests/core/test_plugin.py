# coding=utf-8
"""
test_plugin.py - core/plugin.py 单元测试

覆盖:
- 工具常量定义
- discover_tools() 返回 dict, 加载失败不阻塞
- discover_operators() 返回 list, 加载失败不阻塞
- discover_all() 组合接口
- discover_tools() 返回空 (无 entry_points 注册时)
- 错误处理: 插件加载失败 → 记录 warning + 跳过

Note:
    通过 mock importlib.metadata.entry_points 模拟 entry_points 注册,
    避免依赖实际安装状态。
"""

from __future__ import annotations

import importlib.metadata as md
from unittest.mock import MagicMock, patch

import pytest

from QuantNodes.core.plugin import (
    OPERATORS_GROUP,
    TOOLS_GROUP,
    discover_all,
    discover_operators,
    discover_tools,
)


# ==============================================================================
# 常量
# ==============================================================================


class TestConstants:
    def test_tools_group_name(self):
        assert TOOLS_GROUP == "quantnodes.tools"

    def test_operators_group_name(self):
        assert OPERATORS_GROUP == "quantnodes.operators"


# ==============================================================================
# discover_tools
# ==============================================================================


class FakeEntryPoint:
    """模拟 importlib.metadata.EntryPoint

    模仿真实行为: ep.load() 返回 entry 解析的对象 (类/函数) 本身,
    不是调用结果。调用方决定是否再次调用。

    Args:
        name: entry point name
        value: load() 返回的对象 (类、函数等)
        raw: 原始字符串表示 (用于错误日志)
    """

    def __init__(self, name: str, value=None, raw: str = ""):
        self.name = name
        self.value = raw
        self._resolved = value

    def load(self):
        """返回 entry 解析的对象 (不调用)。"""
        if self._resolved is None:
            # 模拟找不到模块: 让 importlib 抛错
            raise ImportError(f"cannot load {self.value}")
        return self._resolved


class TestDiscoverToolsEmpty:
    def test_no_entry_points_returns_empty(self):
        """无 entry_points 注册时返回空 dict。"""
        with patch.object(md, "entry_points", return_value=[]):
            result = discover_tools()
        assert result == {}

    def test_no_tools_group_returns_empty(self):
        """entry_points 不含 quantnodes.tools 组时返回空。"""
        fake_eps = [FakeEntryPoint("other", raw="some.module:Class")]
        with patch.object(md, "entry_points", return_value=fake_eps):
            result = discover_tools()
        assert result == {}


class TestDiscoverToolsLoaded:
    def test_loads_classes_from_entry_points(self):
        """成功加载 entry_points 中声明的类。"""

        class FakeTool:
            pass

        eps = [FakeEntryPoint("my_tool", value=FakeTool, raw="x:Tool")]
        with patch.object(md, "entry_points", return_value=eps):
            result = discover_tools()
        assert "my_tool" in result
        assert result["my_tool"] is FakeTool

    def test_multiple_tools_loaded(self):
        """多个 entry_points 同时加载。"""

        class ToolA:
            pass

        class ToolB:
            pass

        eps = [
            FakeEntryPoint("a", value=ToolA),
            FakeEntryPoint("b", value=ToolB),
        ]
        with patch.object(md, "entry_points", return_value=eps):
            result = discover_tools()
        assert set(result.keys()) == {"a", "b"}

    def test_failed_load_skipped_with_warning(self, caplog):
        """加载失败时跳过并记录 warning。"""
        eps = [FakeEntryPoint("bad", raw="non_existent.module:MissingClass")]
        with patch.object(md, "entry_points", return_value=eps):
            with caplog.at_level("WARNING"):
                result = discover_tools()
        assert "bad" not in result
        assert "Failed to load tool plugin" in caplog.text

    def test_partial_failure_does_not_block_others(self, caplog):
        """部分插件失败不影响其他插件加载。"""

        class GoodTool:
            pass

        eps = [
            FakeEntryPoint("bad", raw="non_existent:Missing"),
            FakeEntryPoint("good", value=GoodTool),
        ]
        with patch.object(md, "entry_points", return_value=eps):
            with caplog.at_level("WARNING"):
                result = discover_tools()
        assert "good" in result
        assert "bad" not in result


# ==============================================================================
# discover_operators
# ==============================================================================


class TestDiscoverOperatorsEmpty:
    def test_no_entry_points_returns_empty_list(self):
        """无 entry_points 注册时返回空 list。"""
        with patch.object(md, "entry_points", return_value=[]):
            result = discover_operators()
        assert result == []


class TestDiscoverOperatorsLoaded:
    def test_loads_op_names_from_loader(self):
        """loader 返回的算子名列表被正确收集。"""
        loader = lambda: ["op_a", "op_b", "op_c"]
        eps = [FakeEntryPoint("builtin", value=loader, raw="some.module:get_ops")]
        with patch.object(md, "entry_points", return_value=eps):
            result = discover_operators()
        assert result == ["op_a", "op_b", "op_c"]

    def test_loader_returns_tuple(self):
        """loader 返回 tuple 也被接受。"""
        loader = lambda: ("op_a", "op_b")
        eps = [FakeEntryPoint("builtin", value=loader, raw="some.module:get_ops")]
        with patch.object(md, "entry_points", return_value=eps):
            result = discover_operators()
        assert result == ["op_a", "op_b"]

    def test_multiple_plugins_merged(self):
        """多个插件的算子合并到一个 list。"""
        eps = [
            FakeEntryPoint("a", value=lambda: ["op1", "op2"], raw="x:get"),
            FakeEntryPoint("b", value=lambda: ["op3"], raw="y:get"),
        ]
        with patch.object(md, "entry_points", return_value=eps):
            result = discover_operators()
        assert set(result) == {"op1", "op2", "op3"}

    def test_non_list_return_warns(self, caplog):
        """loader 返回非 list/tuple 时记录 warning 并跳过。"""
        loader = lambda: "not_a_list"
        eps = [FakeEntryPoint("bad", value=loader, raw="x:get")]
        with patch.object(md, "entry_points", return_value=eps):
            with caplog.at_level("WARNING"):
                result = discover_operators()
        assert result == []
        assert "did not return a list" in caplog.text

    def test_loader_raises_warns_and_skips(self, caplog):
        """loader 抛异常时记录 warning 并跳过。"""
        # 模拟 ep.load() 抛异常 (而非 loader() 抛异常)
        class RaisingEP(FakeEntryPoint):
            def load(self):
                raise RuntimeError("boom")

        eps = [RaisingEP("bad", raw="x:get")]
        with patch.object(md, "entry_points", return_value=eps):
            with caplog.at_level("WARNING"):
                result = discover_operators()
        assert result == []
        assert "Failed to load operator plugin" in caplog.text


# ==============================================================================
# discover_all
# ==============================================================================


class TestDiscoverAll:
    def test_combines_tools_and_operators(self):
        """discover_all 返回 {tools, operators} 组合字典。"""

        class T:
            pass

        eps = [
            FakeEntryPoint("my_tool", value=T),
            FakeEntryPoint("builtin_ops", value=lambda: ["op1"], raw="x:get"),
        ]
        with patch.object(md, "entry_points", return_value=eps):
            result = discover_all()
        assert "tools" in result
        assert "operators" in result
        assert "my_tool" in result["tools"]
        assert result["operators"] == ["op1"]

    def test_empty_returns_empty_dicts(self):
        """无 entry_points 时返回空字典结构。"""
        with patch.object(md, "entry_points", return_value=[]):
            result = discover_all()
        assert result == {"tools": {}, "operators": []}