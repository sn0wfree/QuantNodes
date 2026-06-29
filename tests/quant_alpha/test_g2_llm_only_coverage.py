# coding=utf-8
"""
test_g2_llm_only_coverage.py - Phase D.4: g2_llm_only.py coverage (70→80%)

目标行:
  106, 108-149: _generate_with_llm LLM 路径
  154-178: _validate_formula
  201-202: _parse_llm_formulas markdown invalid JSON
  207: _parse_llm_formulas regex fallback

策略:
  - Mock nanobot Agent import → ImportError → exception handler (lines 107-109)
  - Mock _validate_formula → True → LLM path with valid formulas (lines 111-149)
  - Test _validate_formula directly (lines 154-178)
  - Test _parse_llm_formulas markdown invalid JSON (lines 201-202)
  - Test _parse_llm_formulas regex fallback (lines 205-207)
"""

from __future__ import annotations

import builtins
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from QuantNodes.research.quant_alpha.evaluation.baselines.g2_llm_only import (
    G2LlmOnly,
)


# ---------------------------------------------------------------------------
# 1. _generate_with_llm exception path (lines 107-109)
# ---------------------------------------------------------------------------
class TestG2GenerateWithLlmException:
    """Mock nanobot Agent import → ImportError → exception handler → mock fallback."""

    def test_agent_import_error_falls_back_to_mock(self):
        """Agent import 失败时回退到 mock 生成。"""
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "QuantNodes.agent.nanobot_bridge":
                raise ImportError("mock nanobot not available")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            g2 = G2LlmOnly(n=5, seed=42, llm_client=MagicMock())
            factors = g2.generate_factors(n=5)

        assert len(factors) == 5
        assert all(f.source == "g2_llm_only" for f in factors)
        assert all(f.formula for f in factors)

    def test_agent_runtime_error_falls_back_to_mock(self):
        """Agent.run() 运行时异常 → exception handler → mock fallback。"""
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("agent crashed")

        def fake_import(name, *args, **kwargs):
            if name == "QuantNodes.agent.nanobot_bridge":
                mod = MagicMock()
                mod.Agent = MagicMock(return_value=mock_agent)
                return mod
            return original_import(name, *args, **kwargs)

        original_import = builtins.__import__
        with patch("builtins.__import__", side_effect=fake_import):
            g2 = G2LlmOnly(n=3, seed=42, llm_client=MagicMock())
            factors = g2.generate_factors(n=3)

        assert len(factors) == 3
        assert all(f.source == "g2_llm_only" for f in factors)


# ---------------------------------------------------------------------------
# 2. _generate_with_llm LLM success path with mocked validation (lines 111-149)
# ---------------------------------------------------------------------------
class TestG2GenerateWithLlmSuccess:
    """Mock Agent to return valid JSON + mock _validate_formula → cover LLM success path."""

    def test_llm_success_all_valid(self):
        """LLM 返回 3 个有效公式，全部通过验证。"""
        mock_agent = MagicMock()

        async def mock_run(*args, **kwargs):
            return '["delta(close, 5)", "ts_mean(vol, 10)", "sign(ts_std(close, 3))"]'

        mock_agent.run = mock_run

        def fake_import(name, *args, **kwargs):
            if name == "QuantNodes.agent.nanobot_bridge":
                mod = MagicMock()
                mod.Agent = MagicMock(return_value=mock_agent)
                return mod
            return original_import(name, *args, **kwargs)

        original_import = builtins.__import__
        with (
            patch("builtins.__import__", side_effect=fake_import),
            patch.object(G2LlmOnly, "_validate_formula", return_value=True),
        ):
            g2 = G2LlmOnly(n=3, seed=42, llm_client=MagicMock())
            factors = g2.generate_factors(n=3)

        assert len(factors) == 3
        assert all(f.source == "g2_llm_only" for f in factors)

    def test_llm_success_some_invalid_supplement_with_mock(self):
        """LLM 返回 5 个公式，只有 1 个有效 → mock 补充 4 个 (lines 120-128)。"""
        mock_agent = MagicMock()

        async def mock_run(*args, **kwargs):
            return '["delta(close, 5)", "rank(close)", "ts_std(vol, 10)", "INVALID_A", "INVALID_B"]'

        mock_agent.run = mock_run

        def fake_import(name, *args, **kwargs):
            if name == "QuantNodes.agent.nanobot_bridge":
                mod = MagicMock()
                mod.Agent = MagicMock(return_value=mock_agent)
                return mod
            return original_import(name, *args, **kwargs)

        original_import = builtins.__import__
        # _validate_formula: 只有 delta(close, 5) 有效
        def selective_validate(formula: str) -> bool:
            return formula == "delta(close, 5)"

        with (
            patch("builtins.__import__", side_effect=fake_import),
            patch.object(G2LlmOnly, "_validate_formula", side_effect=selective_validate),
        ):
            g2 = G2LlmOnly(n=5, seed=42, llm_client=MagicMock())
            factors = g2.generate_factors(n=5)

        # 1 LLM valid + mock supplement (mock formulas also fail validate, so only 1 factor)
        assert len(factors) >= 1
        # First factor is from LLM (str path, line 132)
        assert factors[0].formula == "delta(close, 5)"


# ---------------------------------------------------------------------------
# 3. _validate_formula direct tests (lines 154-178)
# ---------------------------------------------------------------------------
class TestG2ValidateFormula:
    """直接测试 _validate_formula static method。"""

    def test_valid_formula_returns_true(self):
        """有效公式通过验证。"""
        assert G2LlmOnly._validate_formula("delta(close, 5)") is True

    def test_valid_formula_ts_mean(self):
        """ts_mean 公式通过验证。"""
        assert G2LlmOnly._validate_formula("ts_mean(close, 20)") is True

    def test_invalid_formula_returns_false(self):
        """无效公式 (不支持的算子) 返回 False。"""
        assert G2LlmOnly._validate_formula("IndNeutralize(close, industry)") is False

    def test_invalid_formula_quantile(self):
        """quantile 不支持，返回 False。"""
        assert G2LlmOnly._validate_formula("quantile(close, 0.5)") is False

    def test_garbage_formula_returns_false(self):
        """完全无效的公式返回 False。"""
        assert G2LlmOnly._validate_formula("not_a_formula(???") is False

    def test_simple_close_formula(self):
        """简单 close 公式通过验证。"""
        assert G2LlmOnly._validate_formula("close") is True


# ---------------------------------------------------------------------------
# 4. _parse_llm_formulas edge cases (lines 181-210)
# ---------------------------------------------------------------------------
class TestG2ParseLlmFormulas:
    """测试 _parse_llm_formulas 的各种路径。"""

    def test_json_array(self):
        """直接 JSON 数组解析。"""
        response = '["rank(close)", "ts_mean(vol, 20)"]'
        formulas = G2LlmOnly._parse_llm_formulas(response, 2)
        assert formulas == ["rank(close)", "ts_mean(vol, 20)"]

    def test_markdown_json_block(self):
        """从 markdown 代码块提取。"""
        response = '```json\n["rank(close)", "ts_mean(vol, 20)"]\n```'
        formulas = G2LlmOnly._parse_llm_formulas(response, 2)
        assert formulas == ["rank(close)", "ts_mean(vol, 20)"]

    def test_markdown_invalid_json_falls_through(self):
        """markdown 代码块内 JSON 无效 → fall through to regex (lines 201-202)。"""
        response = '```\n[invalid json here]\n```'
        formulas = G2LlmOnly._parse_llm_formulas(response, 1)
        # fall through to regex, which also fails → empty
        assert formulas == []

    def test_regex_fallback(self):
        """无 JSON/markdown → 正则提取 (lines 205-207)。"""
        response = 'Here are formulas: "rank(close)" and "ts_std(vol, 10)"'
        formulas = G2LlmOnly._parse_llm_formulas(response, 2)
        assert formulas == ["rank(close)", "ts_std(vol, 10)"]

    def test_empty_response_returns_empty(self):
        """空响应返回空列表 (line 207)。"""
        formulas = G2LlmOnly._parse_llm_formulas("", 5)
        assert formulas == []

    def test_no_formulas_returns_empty(self):
        """响应中无公式内容返回空列表 (line 207)。"""
        formulas = G2LlmOnly._parse_llm_formulas("just some text, no formulas", 5)
        assert formulas == []


# ---------------------------------------------------------------------------
# 5. _infer_category direct tests
# ---------------------------------------------------------------------------
class TestG2InferCategory:
    """测试 _infer_category 分类逻辑。"""

    def test_momentum_ts_mean(self):
        assert G2LlmOnly._infer_category("ts_mean(close, 5)") == "momentum"

    def test_momentum_delta(self):
        assert G2LlmOnly._infer_category("delta(close, 3)") == "momentum"

    def test_volatility(self):
        assert G2LlmOnly._infer_category("ts_std(close, 5)") == "volatility"

    def test_volume(self):
        assert G2LlmOnly._infer_category("Mul(vol, close)") == "volume"

    def test_reversal(self):
        assert G2LlmOnly._infer_category("abs(close)") == "reversal"

    def test_value_fallback(self):
        assert G2LlmOnly._infer_category("close") == "value"
