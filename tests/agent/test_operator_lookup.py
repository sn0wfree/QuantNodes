# coding=utf-8
"""OperatorLookupTool 测试

测试 3 个 action: list_operators, get_operator_info, validate_formula
"""

from __future__ import annotations

import pytest

from QuantNodes.agent.tools.operator_lookup import OperatorLookupTool


# ---------------------------------------------------------------------------
# 1. 基本属性
# ---------------------------------------------------------------------------

class TestOperatorLookupToolBasic:
    def test_name(self):
        tool = OperatorLookupTool()
        assert tool.name == "operator_lookup"

    def test_read_only(self):
        tool = OperatorLookupTool()
        assert tool.read_only is True

    def test_parameters_has_action(self):
        tool = OperatorLookupTool()
        params = tool.parameters
        assert "action" in params["properties"]
        assert params["properties"]["action"]["enum"] == [
            "list_operators", "get_operator_info", "validate_formula",
        ]


# ---------------------------------------------------------------------------
# 2. list_operators
# ---------------------------------------------------------------------------

class TestListOperators:
    @pytest.mark.asyncio
    async def test_list_all(self):
        tool = OperatorLookupTool()
        result = await tool.execute(action="list_operators")
        assert "operators" in result
        assert "total" in result
        assert result["total"] > 100  # 162 个算子
        assert result["category_filter"] is None

    @pytest.mark.asyncio
    async def test_list_by_category_time(self):
        tool = OperatorLookupTool()
        result = await tool.execute(action="list_operators", category="time")
        assert result["total"] > 50
        assert all(op["category"] == "time" for op in result["operators"])

    @pytest.mark.asyncio
    async def test_list_by_category_point(self):
        tool = OperatorLookupTool()
        result = await tool.execute(action="list_operators", category="point")
        assert result["total"] > 30
        assert all(op["category"] == "point" for op in result["operators"])

    @pytest.mark.asyncio
    async def test_list_by_category_section(self):
        tool = OperatorLookupTool()
        result = await tool.execute(action="list_operators", category="section")
        assert result["total"] > 10
        assert all(op["category"] == "section" for op in result["operators"])

    @pytest.mark.asyncio
    async def test_list_has_signature(self):
        tool = OperatorLookupTool()
        result = await tool.execute(action="list_operators")
        ts_mean = next(
            op for op in result["operators"] if op["name"] == "ts_mean"
        )
        assert "signature" in ts_mean
        assert "f" in ts_mean["signature"]


# ---------------------------------------------------------------------------
# 3. get_operator_info
# ---------------------------------------------------------------------------

class TestGetOperatorInfo:
    @pytest.mark.asyncio
    async def test_get_ts_mean(self):
        tool = OperatorLookupTool()
        result = await tool.execute(action="get_operator_info", name="ts_mean")
        assert result["name"] == "ts_mean"
        assert result["category"] == "time"
        assert "signature" in result
        assert "parameters" in result
        assert "doc" in result
        assert "default_window" in result

    @pytest.mark.asyncio
    async def test_get_rank(self):
        tool = OperatorLookupTool()
        result = await tool.execute(action="get_operator_info", name="rank")
        assert result["name"] == "rank"
        assert result["category"] == "section"

    @pytest.mark.asyncio
    async def test_get_delta(self):
        tool = OperatorLookupTool()
        result = await tool.execute(action="get_operator_info", name="delta")
        assert result["name"] == "delta"
        assert "signature" in result

    @pytest.mark.asyncio
    async def test_get_nonexistent(self):
        tool = OperatorLookupTool()
        result = await tool.execute(action="get_operator_info", name="nonexistent_op")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_missing_name(self):
        tool = OperatorLookupTool()
        result = await tool.execute(action="get_operator_info")
        assert "error" in result


# ---------------------------------------------------------------------------
# 4. validate_formula
# ---------------------------------------------------------------------------

class TestValidateFormula:
    @pytest.mark.asyncio
    async def test_valid_simple(self):
        tool = OperatorLookupTool()
        result = await tool.execute(
            action="validate_formula", formula="close"
        )
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_valid_ts_mean(self):
        tool = OperatorLookupTool()
        result = await tool.execute(
            action="validate_formula", formula="ts_mean(close, 20)"
        )
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_valid_rank(self):
        tool = OperatorLookupTool()
        result = await tool.execute(
            action="validate_formula", formula="rank(close)"
        )
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_valid_complex(self):
        tool = OperatorLookupTool()
        result = await tool.execute(
            action="validate_formula",
            formula="rank(ts_mean(close, 20))",
        )
        assert result["valid"] is True

    @pytest.mark.asyncio
    async def test_invalid_syntax(self):
        tool = OperatorLookupTool()
        result = await tool.execute(
            action="validate_formula",
            formula="invalid_operator(close, 20)",
        )
        assert result["valid"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_invalid_missing_formula(self):
        tool = OperatorLookupTool()
        result = await tool.execute(action="validate_formula")
        assert "error" in result


# ---------------------------------------------------------------------------
# 5. 错误处理
# ---------------------------------------------------------------------------

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_unknown_action(self):
        tool = OperatorLookupTool()
        result = await tool.execute(action="unknown_action")
        assert "error" in result
