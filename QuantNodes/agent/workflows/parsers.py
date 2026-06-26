# coding=utf-8
"""parsers.py — 薄封装层，复用 research/quant_alpha/llm/parser.py。

不复制代码，只做 import re-export。
验证器函数以下划线开头，这里包装为 public 函数。
"""

from __future__ import annotations

from QuantNodes.research.quant_alpha.llm.parser import (
    ALLOWED_OPERATORS,
    ParseResult,
    extract_operators,
    parse_json_3layer,
    validate_formula_operators,
    _validate_critic as validate_critic,
    _validate_evaluator as validate_evaluator,
    _validate_formula_translator as validate_formula_translator,
    _validate_idea_generator as validate_idea_generator,
    _validate_reflector as validate_reflector,
    parse_critic_output,
    parse_evaluator_output,
    parse_formula_translator_output,
    parse_idea_generator_output,
    parse_reflector_output,
)

__all__ = [
    "ParseResult",
    "parse_json_3layer",
    "validate_idea_generator",
    "validate_formula_translator",
    "validate_evaluator",
    "validate_reflector",
    "validate_critic",
    "parse_idea_generator_output",
    "parse_formula_translator_output",
    "parse_evaluator_output",
    "parse_reflector_output",
    "parse_critic_output",
    "validate_formula_operators",
    "extract_operators",
    "ALLOWED_OPERATORS",
]
