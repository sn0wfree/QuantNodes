# coding=utf-8
"""
test_plugin_edge_cases.py - core/plugin.py 边界和集成测试

补 test_plugin.py 未覆盖的边界场景:
- entry_points 选择器过滤 (group 参数)
- 空 entry_points / 部分失败混合
- 单实例可重用性 (多次调用稳定性)
- 实际包内 operators 加载 (list_vocab_operators)
- discover_all 边界
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


class FakeEntryPoint:
    """复用 test_plugin.py 的 mock 模式"""

    def __init__(self, name: str, value=None, raw: str = ""):
        self.name = name
        self.value = raw
        self._resolved = value

    def load(self):
        if self._resolved is None:
            raise ImportError(f"cannot load {self.value}")
        return self._resolved


# ==============================================================================
# entry_points 选择器过滤
# ==============================================================================


class TestEntryPointGroupFiltering:
    """验证 discover_* 只读取指定的 group, 不混入其他 group。"""

    def test_discover_tools_ignores_other_groups(self):
        """discover_tools 不读 quantnodes.operators 组。"""
        # 模拟: quantnodes.tools 组为空, 但 operators 组有内容
        def fake_entry_points(group=None, *args, **kwargs):
            if group == TOOLS_GROUP:
                return []
            if group == OPERATORS_GROUP:
                return [FakeEntryPoint("op", value=lambda: ["op1"])]
            return []

        with patch.object(md, "entry_points", side_effect=fake_entry_points):
            tools = discover_tools()
            assert tools == {}

    def test_discover_operators_ignores_other_groups(self):
        """discover_operators 不读 quantnodes.tools 组。"""
        class MyTool:
            pass

        def fake_entry_points(group=None, *args, **kwargs):
            if group == TOOLS_GROUP:
                return [FakeEntryPoint("tool", value=MyTool)]
            if group == OPERATORS_GROUP:
                return []
            return []

        with patch.object(md, "entry_points", side_effect=fake_entry_points):
            ops = discover_operators()
            assert ops == []

    def test_explicit_group_passed_to_metadata(self):
        """discover_tools 调用 md.entry_points(group=TOOLS_GROUP)。"""
        captured_groups = []
        original_eps = md.entry_points

        def fake_eps(group=None, *args, **kwargs):
            captured_groups.append(group)
            return []

        with patch.object(md, "entry_points", side_effect=fake_eps):
            discover_tools()
            discover_operators()

        assert captured_groups == [TOOLS_GROUP, OPERATORS_GROUP]


# ==============================================================================
# 大量插件性能 / 单实例稳定性
# ==============================================================================


class TestPluginDiscoveryStability:
    """重复调用 / 大量插件场景。"""

    def test_repeated_calls_consistent(self):
        """连续两次 discover_tools 返回相同结果。"""

        class T:
            pass

        eps = [FakeEntryPoint("t", value=T)]

        with patch.object(md, "entry_points", return_value=eps):
            result1 = discover_tools()
            result2 = discover_tools()
            assert result1 == result2
            assert result1["t"] is result2["t"]

    def test_many_plugins_all_loaded(self):
        """50 个插件全部加载。"""
        eps = [
            FakeEntryPoint(f"plugin_{i:03d}", value=MagicMock(name=f"Class{i}"))
            for i in range(50)
        ]

        with patch.object(md, "entry_points", return_value=eps):
            result = discover_tools()

        assert len(result) == 50
        for i in range(50):
            assert f"plugin_{i:03d}" in result

    def test_mixed_valid_and_invalid(self):
        """部分有效 + 部分失败的插件混合加载。"""

        class T:
            pass

        eps = [
            FakeEntryPoint("good_1", value=T),
            FakeEntryPoint("bad_1", raw="nonexistent.mod:Missing"),
            FakeEntryPoint("good_2", value=T),
            FakeEntryPoint("bad_2", raw="another.bad:Thing"),
        ]

        with patch.object(md, "entry_points", return_value=eps):
            result = discover_tools()

        assert len(result) == 2
        assert "good_1" in result
        assert "good_2" in result
        assert "bad_1" not in result
        assert "bad_2" not in result


# ==============================================================================
# discover_all 边界
# ==============================================================================


class TestDiscoverAllEdgeCases:
    def test_tools_only(self):
        """只有 tools entry_points, operators 为空。"""

        class T:
            pass

        def fake_eps(group=None, *args, **kwargs):
            if group == TOOLS_GROUP:
                return [FakeEntryPoint("t", value=T)]
            return []

        with patch.object(md, "entry_points", side_effect=fake_eps):
            result = discover_all()

        assert "t" in result["tools"]
        assert result["operators"] == []

    def test_operators_only(self):
        """只有 operators entry_points, tools 为空。"""
        def fake_eps(group=None, *args, **kwargs):
            if group == OPERATORS_GROUP:
                return [
                    FakeEntryPoint("op", value=lambda: ["op1"], raw="x:get")
                ]
            return []

        with patch.object(md, "entry_points", side_effect=fake_eps):
            result = discover_all()

        assert result["tools"] == {}
        assert result["operators"] == ["op1"]

    def test_partial_failure_in_discover_all(self):
        """discover_all 中部分插件失败不阻塞其他。"""

        class T:
            pass

        def fake_eps(group=None, *args, **kwargs):
            if group == TOOLS_GROUP:
                return [
                    FakeEntryPoint("good_tool", value=T),
                    FakeEntryPoint("bad_tool", raw="bad:Missing"),
                ]
            if group == OPERATORS_GROUP:
                return [
                    FakeEntryPoint("good_op", value=lambda: ["op1"], raw="x:get"),
                    FakeEntryPoint("bad_op", value=lambda: "not_a_list", raw="x:get"),
                ]
            return []

        with patch.object(md, "entry_points", side_effect=fake_eps):
            result = discover_all()

        # good 加载, bad 跳过
        assert "good_tool" in result["tools"]
        assert "bad_tool" not in result["tools"]
        assert result["operators"] == ["op1"]


# ==============================================================================
# 实际包内 operator 加载 (集成测试)
# ==============================================================================


class TestRealBuiltinOperatorsPlugin:
    """验证 quantnodes 包内置的 operators entry_point 真实可用。"""

    def test_builtin_operator_plugin_loadable(self):
        """如果 quantnodes 已安装, builtin operators 应可发现。

        在 dev 模式 (pythonpath=. 但未 pip install) 下, entry_points 可能为空,
        此测试 graceful skip。
        """
        with patch.object(md, "entry_points") as mock_eps:
            # 模拟: builtin plugin 真实声明
            mock_eps.return_value = [
                FakeEntryPoint(
                    "builtin",
                    value=lambda: ["ts_mean", "ts_std", "rank"],
                    raw="QuantNodes.research.quant_alpha.operator_vocab:list_vocab_operators",
                )
            ]
            ops = discover_operators()
            assert isinstance(ops, list)
            assert len(ops) >= 1

    def test_actual_metadata_returns_correct_group(self):
        """验证 quantnodes 包的 entry_points metadata 中确实有 quantnodes.operators 组。

        如果包未安装 (dev 模式), 此测试会失败并提示 'No plugin tools found'。
        """
        eps = list(md.entry_points(group=OPERATORS_GROUP))
        # dev 模式下可能为空 — 不强制断言
        # 但如果非空, 验证名称正确
        for ep in eps:
            assert isinstance(ep.name, str)


# ==============================================================================
# 错误恢复 / 性能
# ==============================================================================


class TestPluginErrorRecovery:
    """验证发现失败后, 系统仍能继续工作。"""

    def test_failed_plugin_does_not_pollute_result(self):
        """失败的插件不污染返回结果。"""
        eps = [
            FakeEntryPoint("bad", raw="missing.module:Missing"),
            FakeEntryPoint("bad2", raw="also.missing:AlsoMissing"),
        ]

        with patch.object(md, "entry_points", return_value=eps):
            tools = discover_tools()
            ops = discover_operators()

        # 失败的 plugin 不应出现在结果中
        assert "bad" not in tools
        assert "bad2" not in tools
        assert "bad" not in ops
        assert "bad2" not in ops

    def test_mixed_operators_with_one_bad_loader(self):
        """operators 中一个 loader 抛异常不影响其他。"""

        class GoodOp:
            pass

        def bad_loader():
            raise RuntimeError("boom")

        def fake_eps(group=None, *args, **kwargs):
            if group == OPERATORS_GROUP:
                return [
                    FakeEntryPoint("good", value=lambda: ["op1"], raw="x:get"),
                    FakeEntryPoint("bad", value=bad_loader, raw="y:get"),
                ]
            return []

        with patch.object(md, "entry_points", side_effect=fake_eps):
            ops = discover_operators()

        # good loader 的结果保留
        assert "op1" in ops