# coding=utf-8
"""
test_parser.py - Alpha-GPT M5 JSON 解析器 + 算子白名单测试

覆盖：
- parse_json_3layer: 3 层降级
- 5 阶段 parse 函数
- validate_formula_operators
- extract_operators
- ALLOWED_OPERATORS 白名单完整性
"""

from __future__ import annotations

import json

import pytest

from QuantNodes.research.quant_alpha.llm import (
    ParseResult,
    parse_json_3layer,
    parse_idea_generator_output,
    parse_formula_translator_output,
    parse_evaluator_output,
    parse_reflector_output,
    parse_critic_output,
    validate_formula_operators,
    extract_operators,
    ALLOWED_OPERATORS,
)


# ==============================================================================
# parse_json_3layer
# ==============================================================================


class TestParseJson3Layer:
    """通用 3 层降级"""

    def test_schema_layer_valid_json(self):
        raw = json.dumps({"ideas": [{"id": "X1", "name": "test"}]})
        r = parse_json_3layer(raw)
        assert r.ok
        assert r.layer == "schema"
        assert r.data == {"ideas": [{"id": "X1", "name": "test"}]}

    def test_regex_layer_with_prefix_text(self):
        raw = (
            "Here's my output:\n\n"
            '{"ideas": [{"id": "X1", "name": "test"}]}\n\nThanks!'
        )
        r = parse_json_3layer(raw)
        assert r.ok
        assert r.layer == "regex"

    def test_regex_layer_with_markdown_wrap(self):
        raw = "```json\n" + json.dumps({"ideas": [{"id": "X1"}]}) + "\n```"
        r = parse_json_3layer(raw)
        assert r.ok
        assert r.layer == "regex"

    def test_layer_3_failure(self):
        raw = "no JSON here at all"
        r = parse_json_3layer(raw)
        assert not r.ok
        assert r.error

    def test_empty_input(self):
        r = parse_json_3layer("")
        assert not r.ok
        r = parse_json_3layer(None)
        assert not r.ok

    def test_with_schema_validator_passes(self):
        def validator(obj):
            return None if "ideas" in obj else "missing ideas"
        raw = json.dumps({"ideas": []})
        r = parse_json_3layer(raw, validator)
        assert r.ok

    def test_with_schema_validator_fails_then_regex(self):
        def validator(obj):
            return None if "ideas" in obj else "missing ideas"
        raw = "prefix " + json.dumps({"ideas": []})
        r = parse_json_3layer(raw, validator)
        assert r.ok
        assert r.layer == "regex"


# ==============================================================================
# 5 阶段 parse 函数
# ==============================================================================


class TestIdeaGeneratorParser:
    def test_valid(self):
        raw = json.dumps({"ideas": [{"id": "I1", "name": "x", "category": "reversal"}]})
        r = parse_idea_generator_output(raw)
        assert r.ok
        assert len(r.data["ideas"]) == 1

    def test_missing_ideas(self):
        raw = json.dumps({"foo": "bar"})
        r = parse_idea_generator_output(raw)
        assert not r.ok

    def test_empty_ideas(self):
        raw = json.dumps({"ideas": []})
        r = parse_idea_generator_output(raw)
        assert not r.ok

    def test_idea_missing_name(self):
        raw = json.dumps({"ideas": [{"id": "I1", "category": "reversal"}]})
        r = parse_idea_generator_output(raw)
        assert not r.ok


class TestFormulaTranslatorParser:
    def test_valid(self):
        raw = json.dumps({"formulas": [{"formula": "rank(x)", "idea_id": "I1"}]})
        r = parse_formula_translator_output(raw)
        assert r.ok

    def test_missing_formula(self):
        raw = json.dumps({"formulas": [{"idea_id": "I1"}]})
        r = parse_formula_translator_output(raw)
        assert not r.ok


class TestEvaluatorParser:
    def test_valid(self):
        raw = json.dumps({"evaluations": [{"formula_id": "F1", "status": "success"}]})
        r = parse_evaluator_output(raw)
        assert r.ok

    def test_missing_status(self):
        raw = json.dumps({"evaluations": [{"formula_id": "F1"}]})
        r = parse_evaluator_output(raw)
        assert not r.ok


class TestReflectorParser:
    def test_valid_keep(self):
        raw = json.dumps({"formula_feedback": [{"formula_id": "F1", "verdict": "keep"}]})
        r = parse_reflector_output(raw)
        assert r.ok

    def test_valid_mutate(self):
        raw = json.dumps({"formula_feedback": [{"verdict": "mutate"}]})
        r = parse_reflector_output(raw)
        assert r.ok

    def test_bad_verdict(self):
        raw = json.dumps({"formula_feedback": [{"verdict": "unknown_xxx"}]})
        r = parse_reflector_output(raw)
        assert not r.ok


class TestCriticParser:
    def test_valid(self):
        raw = json.dumps({"final_pool": [{"formula": "x"}]})
        r = parse_critic_output(raw)
        assert r.ok

    def test_missing_formula(self):
        raw = json.dumps({"final_pool": [{"rank": 1}]})
        r = parse_critic_output(raw)
        assert not r.ok


# ==============================================================================
# validate_formula_operators
# ==============================================================================


class TestFormulaOperatorValidation:
    def test_valid_simple(self):
        assert validate_formula_operators("rank(ts_mean(close, 5))") is None

    def test_valid_nested(self):
        assert (
            validate_formula_operators("sub(close, ts_mean(close, 10))") is None
        )

    def test_invalid_unknown_op(self):
        err = validate_formula_operators("rank(ts_macd(close, 12))")
        assert err is not None
        assert "ts_macd" in err

    def test_extract_operators(self):
        ops = extract_operators("rank(ts_mean(close, 5))")
        assert ops == ["rank", "ts_mean"]

    def test_allowed_operators_minimum(self):
        assert "ts_mean" in ALLOWED_OPERATORS
        assert "rank" in ALLOWED_OPERATORS
        assert "sub" in ALLOWED_OPERATORS
        assert "ts_std" in ALLOWED_OPERATORS


# ==============================================================================
# 截断恢复 (P2)
# ==============================================================================


class TestTruncationRecovery:
    """测试 LLM 输出被 max_tokens 截断时的恢复能力"""

    def test_formula_translator_truncated_mid_list(self):
        """LLM 输出在 formula 列表中间被截断"""
        raw = '{"round": 1, "formulas": [' \
            '{"id": "F-1-1", "idea_id": "I-1-1", "formula": "rank(close)"}, ' \
            '{"id": "F-1-2", "idea_id": "I-1-2", "formula": "ts_mean(close, 5)"}, ' \
            '{"id": "F-1-3", "idea_id": "I-1-3", "formula": "sub(close, ts_'
        r = parse_formula_translator_output(raw)
        assert r.ok
        assert r.layer == "truncated"
        assert "formulas" in r.data
        assert len(r.data["formulas"]) == 2
        assert r.data["formulas"][0]["formula"] == "rank(close)"
        assert r.data["formulas"][1]["formula"] == "ts_mean(close, 5)"
        assert r.data["_recovered_count"] == 2

    def test_idea_generator_truncated(self):
        raw = '{"round": 1, "ideas": [' \
            '{"id": "I-1-1", "name": "反转", "category": "reversal"}, ' \
            '{"id": "I-1-2", "name": "动量", "category": "momen'
        r = parse_idea_generator_output(raw)
        assert r.ok
        assert r.layer == "truncated"
        assert "ideas" in r.data
        assert len(r.data["ideas"]) == 1
        assert r.data["ideas"][0]["id"] == "I-1-1"

    def test_evaluator_truncated(self):
        raw = '{"round": 1, "evaluations": [' \
            '{"formula_id": "F-1-1", "status": "success"}, ' \
            '{"formula_id": "F-1-2", "stat'
        r = parse_evaluator_output(raw)
        assert r.ok
        assert r.layer == "truncated"
        assert "evaluations" in r.data
        assert len(r.data["evaluations"]) == 1

    def test_no_bracket_returns_failure(self):
        """没有任何 {} 或 [] 时应返回失败"""
        raw = "Just some text without any JSON"
        r = parse_formula_translator_output(raw)
        assert not r.ok
        assert "3 layers" in r.error

    def test_empty_input(self):
        r = parse_formula_translator_output("")
        assert not r.ok

    def test_recovered_count_recorded(self):
        """截断恢复后 _recovered_count 应记录恢复数量"""
        raw = '{"formulas": [{"id": "F1", "idea_id": "I1", "formula": "rank(x)"}]}'
        # 正常 JSON 不应触发截断恢复
        r = parse_formula_translator_output(raw)
        assert r.ok
        assert r.layer == "schema"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
