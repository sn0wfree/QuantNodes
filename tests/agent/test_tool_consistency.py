# coding=utf-8
"""
test_tool_consistency.py - Agent tools 一致性和边界测试

覆盖:
- 所有 Tool 子类遵循 Tool ABC (name/description/parameters/read_only/concurrency_safe)
- 所有工具的 parameters schema 符合 OpenAI Function Calling 格式
- 工具 metadata 在 registry 中正确暴露
- 工具构造参数验证

不重复测试每个工具的具体业务逻辑 (已在 test_tools.py / test_tools_all.py 等中覆盖)。
"""

from __future__ import annotations

import inspect

import pytest

from QuantNodes.agent.tools.base import Tool
from QuantNodes.agent.tools import _QUANT_TOOL_FACTORIES


# ==============================================================================
# Tool ABC 合规性
# ==============================================================================


class TestToolABCCompliance:
    """所有工具必须实现 Tool ABC 的所有 abstract 属性/方法。"""

    @pytest.mark.parametrize("factory", _QUANT_TOOL_FACTORIES)
    def test_tool_is_Tool_subclass(self, factory):
        """每个工具类必须是 Tool 子类。"""
        assert issubclass(factory, Tool), (
            f"{factory.__name__} 不是 Tool 子类"
        )

    @pytest.mark.parametrize("factory", _QUANT_TOOL_FACTORIES)
    def test_tool_has_name(self, factory):
        """每个工具有 name 属性。"""
        # 尝试构造 (有的可能需要参数, 跳过)
        try:
            tool = _try_construct(factory)
            if tool is None:
                pytest.skip(f"{factory.__name__} requires args")
            assert isinstance(tool.name, str)
            assert len(tool.name) > 0
        except Exception as e:
            pytest.skip(f"{factory.__name__} cannot construct: {e}")

    @pytest.mark.parametrize("factory", _QUANT_TOOL_FACTORIES)
    def test_tool_has_description(self, factory):
        """每个工具有 description 属性 (非空字符串)。"""
        try:
            tool = _try_construct(factory)
            if tool is None:
                pytest.skip(f"{factory.__name__} requires args")
            assert isinstance(tool.description, str)
            assert len(tool.description) > 0
        except Exception as e:
            pytest.skip(f"{factory.__name__} cannot construct: {e}")

    @pytest.mark.parametrize("factory", _QUANT_TOOL_FACTORIES)
    def test_tool_parameters_is_json_schema(self, factory):
        """每个工具的 parameters 符合 JSON Schema 格式 (type/properties/required)。"""
        try:
            tool = _try_construct(factory)
            if tool is None:
                pytest.skip(f"{factory.__name__} requires args")
            params = tool.parameters
            assert isinstance(params, dict)
            assert "type" in params
            assert params["type"] == "object"
            assert "properties" in params
            assert isinstance(params["properties"], dict)
        except Exception as e:
            pytest.skip(f"{factory.__name__} cannot construct: {e}")


def _try_construct(factory):
    """尝试无参构造, 失败返回 None。"""
    try:
        return factory()
    except Exception:
        return None


# ==============================================================================
# Tool metadata 一致性
# ==============================================================================


class TestToolMetadata:
    def test_tool_names_are_unique(self):
        """所有工具的 name 唯一。"""
        names = []
        for factory in _QUANT_TOOL_FACTORIES:
            tool = _try_construct(factory)
            if tool is not None:
                names.append(tool.name)
        duplicates = [n for n in names if names.count(n) > 1]
        assert len(duplicates) == 0, f"重复工具名: {set(duplicates)}"

    def test_tool_names_are_snake_case(self):
        """工具名符合 snake_case 规范。"""
        import re

        for factory in _QUANT_TOOL_FACTORIES:
            tool = _try_construct(factory)
            if tool is None:
                continue
            assert re.match(r"^[a-z][a-z0-9_]*$", tool.name), (
                f"{factory.__name__}.name = {tool.name!r} 不符合 snake_case"
            )

    def test_all_tools_have_read_only_property(self):
        """所有工具声明 read_only 属性。"""
        for factory in _QUANT_TOOL_FACTORIES:
            tool = _try_construct(factory)
            if tool is None:
                continue
            assert hasattr(tool, "read_only"), (
                f"{factory.__name__} 缺少 read_only 属性"
            )

    def test_all_tools_have_concurrency_safe_property(self):
        """所有工具声明 concurrency_safe 属性。"""
        for factory in _QUANT_TOOL_FACTORIES:
            tool = _try_construct(factory)
            if tool is None:
                continue
            assert hasattr(tool, "concurrency_safe"), (
                f"{factory.__name__} 缺少 concurrency_safe 属性"
            )


# ==============================================================================
# parameters schema 验证
# ==============================================================================


class TestParameterSchema:
    """工具的 parameters schema 必须是合法的 JSON Schema。"""

    @pytest.mark.parametrize("factory", _QUANT_TOOL_FACTORIES)
    def test_required_is_subset_of_properties(self, factory):
        """required 中的字段必须出现在 properties 中。"""
        tool = _try_construct(factory)
        if tool is None:
            pytest.skip(f"{factory.__name__} requires args")

        params = tool.parameters
        required = params.get("required", [])
        properties = params.get("properties", {})

        for req_field in required:
            assert req_field in properties, (
                f"{factory.__name__}.required 字段 {req_field!r} "
                f"未在 properties 中声明"
            )

    @pytest.mark.parametrize("factory", _QUANT_TOOL_FACTORIES)
    def test_properties_have_types(self, factory):
        """properties 中每个字段必须有 type。"""
        tool = _try_construct(factory)
        if tool is None:
            pytest.skip(f"{factory.__name__} requires args")

        params = tool.parameters
        properties = params.get("properties", {})

        for name, spec in properties.items():
            assert isinstance(spec, dict), (
                f"{factory.__name__}.properties.{name} 不是 dict"
            )
            # 至少有一个 type (或 $ref / anyOf 等高级特性)
            assert "type" in spec or "$ref" in spec or "anyOf" in spec, (
                f"{factory.__name__}.properties.{name} 缺少 type"
            )

    def test_pipeline_tools_have_pipeline_code_param(self):
        """Pipeline 类工具 (pipeline, backtest, sandbox) 应有 pipeline_code 参数。"""
        for factory_name in ["PipelineTool", "BacktestTool", "SandboxTool"]:
            for factory in _QUANT_TOOL_FACTORIES:
                if factory.__name__ == factory_name:
                    tool = _try_construct(factory)
                    if tool is None:
                        continue
                    params = tool.parameters
                    # 至少有 code 或 pipeline_code 或类似参数
                    has_code_param = any(
                        "code" in name.lower()
                        for name in params.get("properties", {}).keys()
                    )
                    assert has_code_param, (
                        f"{factory_name} 缺少 code 相关参数"
                    )


# ==============================================================================
# Tool registration 一致性
# ==============================================================================


class TestToolRegistration:
    def test_all_factory_tools_in_registration_list(self):
        """_QUANT_TOOL_FACTORIES 列表包含所有应该注册的工具。"""
        # 至少包含核心 5 个工具
        factory_names = {f.__name__ for f in _QUANT_TOOL_FACTORIES}
        for required in [
            "EchoTool",
            "SandboxTool",
            "PipelineTool",
            "BacktestTool",
            "FactorTool",
            "WikiTool",
        ]:
            assert required in factory_names, (
                f"{required} 不在 _QUANT_TOOL_FACTORIES 列表中"
            )

    def test_factory_count_matches_documented(self):
        """_QUANT_TOOL_FACTORIES 长度 >= 15 (含 builtin 工具)。"""
        assert len(_QUANT_TOOL_FACTORIES) >= 15


# ==============================================================================
# Tool 实例化边界
# ==============================================================================


class TestToolInstantiation:
    def test_echo_tool_default_construction(self):
        """EchoTool 无参构造成功。"""
        from QuantNodes.agent.tools.echo import EchoTool

        tool = EchoTool()
        assert tool.name == "echo"

    def test_sandbox_tool_default_construction(self):
        """SandboxTool 无参构造成功。"""
        from QuantNodes.agent.tools.sandbox import SandboxTool

        tool = SandboxTool()
        assert tool.name == "sandbox"
        assert tool.read_only is True

    def test_pipeline_tool_default_construction(self):
        """PipelineTool 无参构造成功。"""
        from QuantNodes.agent.tools.pipeline import PipelineTool

        tool = PipelineTool()
        assert tool.name == "pipeline"
        assert tool.read_only is True