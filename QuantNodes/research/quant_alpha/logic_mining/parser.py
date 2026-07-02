# coding=utf-8
"""
parser.py - Logic Mining 输出解析

解析三个 Agent 的输出 JSON，包含 Schema 校验和默认值。

Usage::

    from QuantNodes.research.quant_alpha.logic_mining.parser import (
        parse_formula_structure, parse_financial_semantics,
        parse_market_logic, _mock_structure_response, _mock_semantics_response,
        _mock_abstraction_response,
    )

    structure = parse_formula_structure(llm_output)
    semantics = parse_financial_semantics(llm_output)
    logic = parse_market_logic(llm_output)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "ParseResult",
    "parse_json_response",
    "parse_formula_structure",
    "parse_financial_semantics",
    "parse_market_logic",
    "_mock_structure_response",
    "_mock_semantics_response",
    "_mock_abstraction_response",
]


@dataclass
class ParseResult:
    """JSON 解析结果 (v3.0.1: 新增 layer_reached / last_error / layer_errors)

    Attributes:
        ok:              是否成功解析
        data:            成功时的 dict (其余情况为 None)
        error:           失败原因 (聚合)
        raw:             原始输入文本
        layer_reached:   触达的最高层 (0=空/未尝试, 1=直接, 2=md fence, 3=brace)
        last_error:      最近一次的 JSONDecodeError 文本
        layer_errors:    各层的错误文本 (用于定位失败层)
    """
    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    raw: str = ""
    layer_reached: int = 0
    last_error: Optional[str] = None
    layer_errors: Dict[int, str] = field(default_factory=dict)


def parse_json_response(raw: str) -> ParseResult:
    """三层 fallback JSON parser

    1. 直接 json.loads
    2. 从 markdown ```json ... ``` 中提取
    3. 从文本中找第一个 {...} 块

    Args:
        raw: LLM 输出文本

    Returns:
        ParseResult — 失败时 layer_reached=0/1/2/3 标识最远触达层,
        last_error 与 layer_errors 暴露每个被吞掉的 JSONDecodeError
    """
    if not raw:
        return ParseResult(
            ok=False, error="empty response", raw=raw,
            layer_reached=0, last_error=None, layer_errors={},
        )

    # 第 1 层: 直接 JSON
    try:
        return ParseResult(
            ok=True, data=json.loads(raw), raw=raw,
            layer_reached=1, last_error=None, layer_errors={},
        )
    except json.JSONDecodeError as e:
        first_err = str(e)

    # 第 2 层: 从 ```json ... ``` 中提取
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    layer_errors: Dict[int, str] = {1: first_err}
    if md_match:
        try:
            return ParseResult(
                ok=True, data=json.loads(md_match.group(1)), raw=raw,
                layer_reached=2, last_error=None, layer_errors=layer_errors,
            )
        except json.JSONDecodeError as e:
            layer_errors[2] = str(e)
            last_err = str(e)
    else:
        last_err = first_err

    # 第 3 层: 找第一个 {...} 块
    brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
    if brace_match:
        try:
            return ParseResult(
                ok=True, data=json.loads(brace_match.group(0)), raw=raw,
                layer_reached=3, last_error=None, layer_errors=layer_errors,
            )
        except json.JSONDecodeError as e:
            layer_errors[3] = str(e)
            last_err = str(e)

    # 全部失败
    max_layer = max(layer_errors.keys()) if layer_errors else 0
    return ParseResult(
        ok=False,
        error=f"Cannot parse JSON after 3 layers (last layer={max_layer})",
        raw=raw,
        layer_reached=max_layer,
        last_error=last_err,
        layer_errors=layer_errors,
    )


def parse_formula_structure(raw: str) -> ParseResult:
    """解析 FormulaStructureAgent 输出

    期望格式:
    {
        "operations": ["rank", "ts_corr"],
        "window_length": 10,
        "has_ranking": true,
        "has_normalization": false
    }
    """
    result = parse_json_response(raw)
    if not result.ok:
        return result

    data = result.data
    if not isinstance(data, dict):
        return ParseResult(
            ok=False, error="data is not a dict", raw=raw,
            layer_reached=result.layer_reached,
            last_error=result.last_error,
            layer_errors=result.layer_errors,
        )

    # 必填字段验证
    if "operations" not in data:
        return ParseResult(
            ok=False, error="missing 'operations' field", raw=raw,
            layer_reached=result.layer_reached,
            last_error=result.last_error,
            layer_errors=result.layer_errors,
        )

    # 填充默认值
    validated = {
        "operations": data.get("operations", []),
        "window_length": data.get("window_length", 0),
        "has_ranking": data.get("has_ranking", False),
        "has_normalization": data.get("has_normalization", False),
    }

    return ParseResult(ok=True, data=validated, raw=raw)


def parse_financial_semantics(raw: str) -> ParseResult:
    """解析 FinancialSemanticsMappingAgent 输出

    期望格式:
    {
        "price_role": "initial reaction",
        "volume_role": "participation",
        "time_pattern": "persistent co-movement",
        "behavior_interpretation": "lack of confirmation indicates reversal"
    }
    """
    result = parse_json_response(raw)
    if not result.ok:
        return result

    data = result.data
    if not isinstance(data, dict):
        return ParseResult(
            ok=False, error="data is not a dict", raw=raw,
            layer_reached=result.layer_reached,
            last_error=result.last_error,
            layer_errors=result.layer_errors,
        )

    validated = {
        "price_role": data.get("price_role", "unknown"),
        "volume_role": data.get("volume_role", "unknown"),
        "time_pattern": data.get("time_pattern", "unknown"),
        "behavior_interpretation": data.get("behavior_interpretation", "unknown"),
    }

    return ParseResult(ok=True, data=validated, raw=raw)


def parse_market_logic(raw: str) -> ParseResult:
    """解析 MarketLogicAbstractionAgent 输出

    期望格式:
    {
        "predicates": [{"variable": "open", "op": "rank", "threshold": 0}],
        "behavior": {"target": "forward_return_5", "direction": -1, "horizon": 5},
        "operator_whitelist": ["rank", "ts_corr", "sign"],
        "parameter_ranges": {"ts_corr": [5, 30]},
        "sign_constraint": -1
    }
    """
    result = parse_json_response(raw)
    if not result.ok:
        return result

    data = result.data
    if not isinstance(data, dict):
        return ParseResult(
            ok=False, error="data is not a dict", raw=raw,
            layer_reached=result.layer_reached,
            last_error=result.last_error,
            layer_errors=result.layer_errors,
        )

    # 必填字段验证
    if "predicates" not in data or "behavior" not in data:
        return ParseResult(
            ok=False, error="missing 'predicates' or 'behavior'", raw=raw,
            layer_reached=result.layer_reached,
            last_error=result.last_error,
            layer_errors=result.layer_errors,
        )

    return ParseResult(ok=True, data=data, raw=raw)


# ==============================================================================
# Mock 响应（用于离线测试）
# ==============================================================================


def _mock_structure_response(formula: str) -> str:
    """生成 FormulaStructureAgent 的 mock 响应"""
    import re as _re

    # 提取算子
    ops = set(_re.findall(r"\b([a-zA-Z_]\w*)\s*\(", formula))
    ops.discard("if")
    ops.discard("else")

    # 提取最大窗口
    nums = [int(n) for n in _re.findall(r",\s*(\d+)\s*\)", formula)]
    max_window = max(nums) if nums else 0

    return json.dumps({
        "operations": sorted(ops),
        "window_length": max_window,
        "has_ranking": "rank(" in formula,
        "has_normalization": any(op in formula for op in ["zscore", "normalize"]),
    })


def _mock_semantics_response(formula: str) -> str:
    """生成 FinancialSemanticsMappingAgent 的 mock 响应"""
    # 基于算子推断语义
    has_corr = "corr(" in formula or "ts_corr" in formula
    has_volume = "vol" in formula.lower() or "amount" in formula
    has_ts_mean = "ts_mean(" in formula
    has_rank = "rank(" in formula

    price_role = "mean reversion" if "delta(" in formula else "trend indicator"
    volume_role = "participation" if has_volume else "not used"
    time_pattern = (
        "windowed co-movement" if has_corr else
        "moving average" if has_ts_mean else "single point"
    )
    behavior = (
        "divergence signal" if has_corr and has_volume else
        "momentum/reversal indicator" if has_rank else
        "neutral"
    )

    return json.dumps({
        "price_role": price_role,
        "volume_role": volume_role,
        "time_pattern": time_pattern,
        "behavior_interpretation": behavior,
    })


def _mock_abstraction_response(
    formula: str,
    structure: Dict[str, Any],
    semantics: Dict[str, Any],
) -> str:
    """生成 MarketLogicAbstractionAgent 的 mock 响应"""
    # 基于公式特征生成结构化逻辑
    operations = structure.get("operations", [])
    has_volume = semantics.get("volume_role", "not used") != "not used"
    has_corr = "ts_corr" in operations or "corr" in operations
    has_ts_mean = "ts_mean" in operations

    # 行为推断
    if has_corr and has_volume:
        direction = -1  # 量价背离 → 反转
        target = "forward_return_5"
        horizon = 5
        predicates = [
            {"variable": "open", "op": "rank", "threshold": 0},
            {"variable": "volume", "op": "rank", "threshold": 0},
            {"variable": "open", "op": "ts_corr", "threshold": -0.5,
             "window": 10, "second_variable": "volume"},
        ]
        whitelist = ["rank", "ts_corr", "sign", "sub", "mul", "div"]
        param_ranges = {"ts_corr": [5, 30]}
        sign = -1
    elif has_ts_mean:
        direction = -1
        target = "forward_return_5"
        horizon = 5
        predicates = [
            {"variable": "close", "op": "ts_mean", "threshold": 0, "window": 20},
        ]
        whitelist = ["ts_mean", "rank", "sub", "div", "sign"]
        param_ranges = {"ts_mean": [5, 60]}
        sign = -1
    else:
        direction = 1
        target = "forward_return_1"
        horizon = 1
        predicates = [{"variable": "close", "op": "rank", "threshold": 0}]
        whitelist = operations if operations else ["rank"]
        param_ranges = {}
        sign = 1

    return json.dumps({
        "predicates": predicates,
        "behavior": {"target": target, "direction": direction, "horizon": horizon},
        "operator_whitelist": whitelist,
        "parameter_ranges": param_ranges,
        "sign_constraint": sign,
    })