# coding=utf-8
"""
parser.py - LLM 输出 JSON 三层降级解析器（Alpha-GPT M5）

Alpha-GPT 工作流的【所有 5 阶段】输出都依赖 LLM JSON 解析。
LLM 输出不稳定（多余文本 / markdown 包裹 / 截断），需要
3 层降级：JSON Schema → 正则提取 → 重试 LLM。

零新依赖（不引 instructor / outlines）。

Usage::

    from QuantNodes.research.quant_alpha.llm.parser import (
        FormulaParser, parse_idea_generator_output,
        parse_formula_translator_output, parse_evaluator_output,
        parse_reflector_output, parse_critic_output,
    )

    result = parse_idea_generator_output(llm_output)
    if result.ok:
        ideas = result.data["ideas"]
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ==============================================================================
# 通用 Result
# ==============================================================================


@dataclass
class ParseResult:
    """JSON 解析结果"""

    ok: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    layer: str = ""  # "schema" | "regex" | "retry" | "default"
    raw: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "layer": self.layer,
            "error": self.error,
            "data": self.data,
        }


# ==============================================================================
# 通用 3 层降级
# ==============================================================================


def parse_json_3layer(
    raw: str,
    schema_validator: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
) -> ParseResult:
    """3 层降级 JSON 解析

    Layer 1: 直接 json.loads + schema 校验
    Layer 2: 正则提取首个 { ... } 块 + 重新解析
    Layer 3: 失败 → 返回 error（上层可重试 LLM）

    Args:
        raw: LLM 输出文本
        schema_validator: 可选 schema 校验函数，返回 None 表示通过，
            返回 str 表示失败原因

    Returns:
        ParseResult
    """
    if raw is None or not raw.strip():
        return ParseResult(ok=False, error="empty input", raw=raw)

    def _try(s: str) -> Optional[Dict[str, Any]]:
        try:
            obj = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(obj, dict):
            return None
        if schema_validator is not None:
            err = schema_validator(obj)
            if err is not None:
                return None
        return obj

    obj = _try(raw)
    if obj is not None:
        return ParseResult(ok=True, data=obj, layer="schema", raw=raw)

    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        obj = _try(m.group(0))
        if obj is not None:
            return ParseResult(ok=True, data=obj, layer="regex", raw=raw)

    return ParseResult(
        ok=False,
        error="Cannot parse JSON after 2 layers (full raw in ParseResult.raw)",
        raw=raw,
    )


# ==============================================================================
# 5 阶段 schema 校验
# ==============================================================================


def _validate_idea_generator(obj: Dict[str, Any]) -> Optional[str]:
    """IdeaGenerator 输出 schema"""
    if "ideas" not in obj:
        return "missing 'ideas'"
    ideas = obj["ideas"]
    if not isinstance(ideas, list):
        return "'ideas' must be list"
    if len(ideas) == 0:
        return "'ideas' empty"
    for i, idea in enumerate(ideas):
        if not isinstance(idea, dict):
            return f"ideas[{i}] not dict"
        if "id" not in idea or "name" not in idea:
            return f"ideas[{i}] missing id/name"
        if "category" not in idea:
            return f"ideas[{i}] missing category"
    return None


def _validate_formula_translator(obj: Dict[str, Any]) -> Optional[str]:
    """FormulaTranslator 输出 schema"""
    if "formulas" not in obj:
        return "missing 'formulas'"
    formulas = obj["formulas"]
    if not isinstance(formulas, list):
        return "'formulas' must be list"
    if len(formulas) == 0:
        return "'formulas' empty"
    for i, f in enumerate(formulas):
        if not isinstance(f, dict):
            return f"formulas[{i}] not dict"
        if "formula" not in f or "idea_id" not in f:
            return f"formulas[{i}] missing formula/idea_id"
    return None


def _validate_evaluator(obj: Dict[str, Any]) -> Optional[str]:
    """Evaluator 输出 schema"""
    if "evaluations" not in obj:
        return "missing 'evaluations'"
    evals = obj["evaluations"]
    if not isinstance(evals, list):
        return "'evaluations' must be list"
    for i, e in enumerate(evals):
        if not isinstance(e, dict):
            return f"evaluations[{i}] not dict"
        if "formula_id" not in e or "status" not in e:
            return f"evaluations[{i}] missing formula_id/status"
    return None


def _validate_reflector(obj: Dict[str, Any]) -> Optional[str]:
    """Reflector 输出 schema

    兼容两种格式：
    1. 标准格式：包含 formula_feedback 数组
    2. 分析格式：包含 analysis 字段（formula_feedback 可选）
    """
    # 标准格式：必须有 formula_feedback
    if "formula_feedback" in obj:
        feedback = obj["formula_feedback"]
        if not isinstance(feedback, list):
            return "'formula_feedback' must be list"
        for i, fb in enumerate(feedback):
            if "verdict" not in fb:
                return f"formula_feedback[{i}] missing verdict"
            if fb["verdict"] not in {"keep", "mutate", "drop", "merge"}:
                return f"formula_feedback[{i}] bad verdict"
        return None

    # 分析格式：有 analysis 即可（formula_feedback 可选）
    if "analysis" in obj:
        return None

    return "missing 'formula_feedback' or 'analysis'"


def _validate_critic(obj: Dict[str, Any]) -> Optional[str]:
    """Critic 输出 schema"""
    if "final_pool" not in obj:
        return "missing 'final_pool'"
    pool = obj["final_pool"]
    if not isinstance(pool, list):
        return "'final_pool' must be list"
    for i, item in enumerate(pool):
        if "formula" not in item:
            return f"final_pool[{i}] missing formula"
    return None


# ==============================================================================
# 5 阶段 parse 函数
# ==============================================================================


def parse_idea_generator_output(raw: str) -> ParseResult:
    """IdeaGenerator 输出解析"""
    return parse_json_3layer(raw, _validate_idea_generator)


def parse_formula_translator_output(raw: str) -> ParseResult:
    """FormulaTranslator 输出解析"""
    return parse_json_3layer(raw, _validate_formula_translator)


def parse_evaluator_output(raw: str) -> ParseResult:
    """Evaluator 输出解析"""
    return parse_json_3layer(raw, _validate_evaluator)


def parse_reflector_output(raw: str) -> ParseResult:
    """Reflector 输出解析"""
    return parse_json_3layer(raw, _validate_reflector)


def parse_critic_output(raw: str) -> ParseResult:
    """Critic 输出解析"""
    return parse_json_3layer(raw, _validate_critic)


# ==============================================================================
# 公式白名单校验（FormulaTranslator 专用）
# ==============================================================================


ALLOWED_OPERATORS: set[str] = {
    # 时序
    "ts_mean", "ts_std", "ts_sum", "ts_max", "ts_min", "ts_median",
    "ts_rank", "ts_zscore", "ts_skew", "ts_kurt",
    "ts_decay_linear", "ts_corr", "ts_cov", "ts_delay",
    # 截面
    "rank", "zscore", "winsorize", "IndNeutralize",
    # 一元
    "abs", "sign", "log", "sqrt", "signedpower",
    # 二元
    "add", "sub", "mul", "div", "greater", "less",
    # 时序位移
    "delta", "delay",
    # 复合（解析器展开）
    "returns",
    # polars 原生语法兼容
    "shift", "Ref",
}


def extract_operators(formula: str) -> List[str]:
    """从公式字符串中提取所有算子名（词法分析）"""
    return re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", formula)


def validate_formula_operators(formula: str) -> Optional[str]:
    """校验公式中的算子是否在白名单

    Returns:
        None if OK, error message string if invalid
    """
    ops = extract_operators(formula)
    for op in ops:
        if op not in ALLOWED_OPERATORS:
            return f"Unknown operator: {op!r}"
    return None


__all__ = [
    "ParseResult",
    "parse_json_3layer",
    "parse_idea_generator_output",
    "parse_formula_translator_output",
    "parse_evaluator_output",
    "parse_reflector_output",
    "parse_critic_output",
    "validate_formula_operators",
    "extract_operators",
    "ALLOWED_OPERATORS",
]
