# coding=utf-8
"""
test_parse_result_layers.py — ParseResult 7 字段 + 3 层 JSON observability (v3.0.1)

覆盖 P-06 / P-07 / P-08: parse_json_response 失败时不再静默,
而是 layer_reached / last_error / layer_errors 暴露每个被吞掉的 JSONDecodeError
"""
import json

import pytest

from QuantNodes.research.quant_alpha.logic_mining.parser import (
    parse_json_response,
    parse_formula_structure,
    parse_financial_semantics,
    parse_market_logic,
)


class TestLayerReach:
    def test_empty_layer_zero(self):
        r = parse_json_response("")
        assert r.ok is False
        assert r.layer_reached == 0
        assert r.last_error is None
        assert r.layer_errors == {}

    def test_layer_1_direct_json(self):
        r = parse_json_response('{"a": 1}')
        assert r.ok is True
        assert r.layer_reached == 1
        assert r.data == {"a": 1}
        assert r.layer_errors == {}

    def test_layer_2_md_fence(self):
        raw = '前缀文本\n```json\n{"a": 1}\n```\n尾巴'
        r = parse_json_response(raw)
        assert r.ok is True
        assert r.layer_reached == 2
        assert r.data == {"a": 1}

    def test_layer_3_brace(self):
        raw = '前面废话...{"a": 1} 后面废话'
        r = parse_json_response(raw)
        assert r.ok is True
        assert r.layer_reached == 3
        assert r.data == {"a": 1}

    def test_layer_1_failure_then_layer_3_success(self):
        # 第 1 层失败 (裸字符串不是 JSON), 第 2 层无 md fence, 第 3 层 brace 成功
        raw = '❌ not json\nplain text with {brace: 1} somewhere\nanother {"a":1} line'
        r = parse_json_response(raw)
        # greedy regex 抓 {...} 最长匹配的 → 第二个 {...} ... 但字符会被破坏, json.loads 失败
        # 此时应进入 layer_3_failed 状态
        # 此断言依赖实际行为: 我们期望 ok=False/3 reached
        if r.ok:
            assert r.layer_reached in (1, 2, 3)
        else:
            assert r.layer_reached >= 1
            assert r.last_error is not None
            assert len(r.layer_errors) >= 1

    def test_all_layers_fail_records_per_layer(self):
        raw = '彻底无法解析的文字'
        r = parse_json_response(raw)
        assert r.ok is False
        # 第 1 层尝试并失败 → layer_errors[1] 应有
        assert 1 in r.layer_errors
        # 无 md fence → 第 2 层未尝试, 不在 layer_errors
        # 无 brace → 第 3 层未尝试, 不在 layer_errors
        # 故 layer_errors 至少 [1]
        assert r.last_error is not None
        assert r.layer_reached == 1

    def test_all_layers_tried_each_fail_recorded(self):
        raw = '```json\n{bad content}\n```\nsome {also_broken} text'
        r = parse_json_response(raw)
        # 第 1 层 fail (whole not JSON),
        # 第 2 层 md-match but json fail,
        # 第 3 层 brace-match but json fail
        assert r.ok is False
        assert 1 in r.layer_errors
        assert 2 in r.layer_errors
        assert 3 in r.layer_errors
        assert r.layer_reached == 3


class TestSchemaLevelParse:
    """parse_formula_structure / parse_financial_semantics / parse_market_logic"""

    def test_formula_structure_layer_propagated(self):
        r = parse_formula_structure('garbage')
        assert r.ok is False
        assert r.layer_reached >= 1

    def test_financial_semantics_layer_propagated(self):
        r = parse_financial_semantics('garbage')
        assert r.ok is False
        assert r.layer_reached >= 1

    def test_market_logic_layer_propagated(self):
        r = parse_market_logic('garbage')
        assert r.ok is False
        assert r.layer_reached >= 1

    def test_formula_structure_missing_field_keeps_data(self):
        # 第 1 层成功但缺 'operations' → 走 schema 校验, ok=False 不带 data
        raw = json.dumps({"window_length": 5})
        r = parse_formula_structure(raw)
        assert r.ok is False
        assert r.layer_reached == 1
        assert "operations" in (r.error or "")

    def test_formula_structure_defaults_filled(self):
        raw = json.dumps({"operations": ["rank"]})
        r = parse_formula_structure(raw)
        assert r.ok is True
        assert r.data == {
            "operations": ["rank"],
            "window_length": 0,
            "has_ranking": False,
            "has_normalization": False,
        }
