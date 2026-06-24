# coding=utf-8
"""
llm - Alpha-GPT M5 子包（仅 JSON 解析，不含 LLM 调度）

LLM 调度复用 nanobot upstream（见 .agent/agents/alpha-gpt-*.md），
本子包仅提供：

- parser.py: 5 阶段 LLM 输出 JSON 三层降级解析器
- utils: 公式算子白名单校验

M5 doc-first 详见：docs/quant_alpha/alpha_gpt_architecture.md
"""

from .parser import (
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
