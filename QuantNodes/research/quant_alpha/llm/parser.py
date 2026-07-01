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
from typing import Any, Callable, Dict, List, Optional, Tuple, Tuple

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
    """4 层降级 JSON 解析

    Layer 1: 直接 json.loads + schema 校验
    Layer 2: 正则提取首个 { ... } 块 + 重新解析
    Layer 3: truncated recovery (扫描内层完整子对象)
    Layer 4: 找最后一个满足 schema 的 JSON 候选 (处理
             "截断 JSON + thinking + 重写 JSON" 模式)

    Args:
        raw: LLM 输出文本
        schema_validator: 可选 schema 校验函数，返回 None 表示通过，
            返回 str 表示失败原因

    Returns:
        ParseResult
    """
    if raw is None or not raw.strip():
        return ParseResult(ok=False, error="empty input", raw=raw)

    _json_ok = False

    def _try(s: str) -> Optional[Dict[str, Any]]:
        nonlocal _json_ok
        try:
            obj = json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(obj, dict):
            return None
        _json_ok = True
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

    if not _json_ok:
        # Layer 4 (优先): 找最后一个满足 schema 的完整 JSON
        # 处理 LLM "截断 JSON + thinking + 重写 JSON" 模式
        last = _find_last_valid_json(raw, schema_validator)
        if last is not None:
            return ParseResult(ok=True, data=last, layer="last_valid", raw=raw)

        # Layer 3 (fallback): 截断恢复
        truncated = _recover_truncated_json(raw, schema_validator)
        if truncated is not None:
            return ParseResult(
                ok=True, data=truncated, layer="truncated", raw=raw
            )

    return ParseResult(
        ok=False,
        error="Cannot parse JSON after 4 layers (full raw in ParseResult.raw)",
        raw=raw,
    )


def _find_last_valid_json(
    raw: str,
    schema_validator: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
) -> Optional[Dict[str, Any]]:
    """扫描所有 JSON 候选对象，返回最后一个满足 schema 的。

    处理 LLM "截断 JSON + thinking + 重写 JSON" 模式：
    ```
    {
      "round": 1, ...   ← 第一次输出，被 max_tokens 截断
    }
    Actually, ...
    ```json
    {"round": 1, "formulas": [...]}   ← 第二次完整输出
    ```
    ```

    Greedy regex `\{[\s\S]*\}` 会匹配从第一个 `{` 到最后一个 `}`，跨过两个
    JSON，导致解析失败。本函数扫描所有可能的 JSON 起始位置，收集所有
    可解的 dict，然后选**最后一个**（最可能是 LLM 重写的完整版本）。

    过滤规则：
    1. 必须是 dict（非 list / 单值）
    2. 至少 2 个 key（排除只含元数据如 `{"round": 1}` 的 dict）
    3. 通过 schema_validator（如果有）

    Args:
        raw: LLM 输出文本
        schema_validator: 可选 schema 校验函数

    Returns:
        满足条件的最后一个 dict，或 None
    """
    if not raw:
        return None
    decoder = json.JSONDecoder()
    candidates: List[Dict[str, Any]] = []
    n = len(raw)

    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        # 快速排除：前一个字符是字母/数字说明这是内嵌的 dict（不是顶层）
        # 顶层 JSON 起始位置前通常是空白/换行/`{`/`[`/`,` 等
        if i > 0:
            prev = raw[i - 1]
            if prev.isalnum() or prev == '"':
                continue
        try:
            obj, end = decoder.raw_decode(raw, i)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if len(obj) < 2:
            # 排除只含元数据的 dict (如 {"round": 1})
            continue
        if schema_validator is not None:
            try:
                err = schema_validator(obj)
            except Exception:
                err = "validator exception"
            if err is not None:
                continue
        candidates.append(obj)

    return candidates[-1] if candidates else None


def _recover_truncated_json(
    raw: str,
    schema_validator: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
) -> Optional[Dict[str, Any]]:
    """截断恢复：扫描 LLM 输出，提取所有已完整闭合的子对象。

    处理 LLM 输出因 max_tokens 截断的场景：
    - LLM 输出 `{"ideas": [完整对象1, 完整对象2, {不完整对象`
    - 简单 json.loads 失败
    - 但内层 [完整对象1, 完整对象2] 仍可解析

    策略：用 json.JSONDecoder().raw_decode() 反复解码，每次从下一个 [ 或 { 开始。
    - 收集可解的 list（直接收）
    - 收集可解的 dict（按"位置相邻"组成 list，因为数组虽然外层 [] 截断，但
      内部 {item1}, {item2} 都可解，它们之间距离通常很近）

    Returns:
        恢复后的 dict（如 {"items": [item1, item2], "_truncated": True}），
        或 None 表示无法恢复
    """
    if not raw or not raw.strip():
        return None

    decoder = json.JSONDecoder()
    text = raw.strip()

    # 收集可解的对象和它们的终止位置
    decoded_objects: List[Dict[str, Any]] = []
    pos = 0
    n = len(text)
    openers_set = set("[{")

    while pos < n:
        next_pos = -1
        for k in range(pos, n):
            if text[k] in openers_set:
                next_pos = k
                break
        if next_pos == -1:
            break
        try:
            obj, end = decoder.raw_decode(text[next_pos:])
        except json.JSONDecodeError:
            pos = next_pos + 1
            continue
        if isinstance(obj, list) and len(obj) > 0:
            return {
                "_truncated": True,
                "_recovered_count": len(obj),
                "items": obj,
            }
        if isinstance(obj, dict) and len(obj) > 0:
            # 跳过只含顶层元数据的 dict (如 {round: 1})
            if not (set(obj.keys()) <= {"round"}):
                decoded_objects.append(obj)
        pos = next_pos + end

    if decoded_objects:
        return {
            "_truncated": True,
            "_recovered_count": len(decoded_objects),
            "items": decoded_objects,
        }

    return None


# ==============================================================================
# 5 阶段 schema 校验
# ==============================================================================


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
    """FormulaTranslator 输出 schema

    P2 升级 (refactor/smart-p2): 智能 explanation 处理 (3 档)
    - 档 1: 含结构化标记 → 拆分为 summary (explanation) + detail (explanation_detail)
    - 档 2: 超长但无结构化 → 截断到 200 chars
    - 档 3: 短小干净 → 保留原样

    其他强化：
    - idea_id 改为 optional（缺失时 fallback 空串）
    - formula 字段缺失直接 fail
    """
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
        if "formula" not in f:
            return f"formulas[{i}] missing formula"
        # 容忍缺失 idea_id（fallback 空串）
        f.setdefault("idea_id", "")
        # P2 升级: 智能 explanation 处理 (3 档)
        if "explanation" in f and isinstance(f["explanation"], str):
            expl = f["explanation"]
            match = _STRUCTURED_MARKERS_RE.search(expl)
            if match:
                # 档 1: 含结构化标记 → 拆分为 summary + detail
                split_pos = match.start()
                f["explanation"] = expl[:split_pos].strip()
                f["explanation_detail"] = expl[split_pos:]
            elif len(expl) > _MAX_EXPLANATION_LEN:
                # 档 2: 普通超长 → 截断
                f["explanation"] = expl[:_MAX_EXPLANATION_LEN - 3] + "..."
            # 档 3: 短小干净 → 保留（无操作）
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
# 截断恢复后的字段映射
# ==============================================================================


# P2 升级 (refactor/smart-p2): 智能 explanation 处理的模块级常量
_STRUCTURED_MARKERS_RE = re.compile(
    r"\b(HYPOTHESIS|MECHANISM|OPERATOR_RATIONALE|PARAMETER_RATIONALE|"
    r"RISK|SUGGESTED_OPS|FORMULA|HYPOTHESES)\s*:",
    re.IGNORECASE,
)
_MAX_EXPLANATION_LEN = 200


# 截断恢复时把 "items" 重命名为对应 stage 的字段
_TRUNCATED_KEY_MAP = {
    "idea_generator": "ideas",
    "formula_translator": "formulas",
    "evaluator": "evaluations",
    "reflector": "formula_feedback",
    "critic": "final_pool",
}


_STAGE_VALIDATORS = {
    "idea_generator": _validate_idea_generator,
    "formula_translator": _validate_formula_translator,
    "evaluator": _validate_evaluator,
    "reflector": _validate_reflector,
    "critic": _validate_critic,
}


def _apply_truncation_mapping(result: ParseResult, stage: str) -> ParseResult:
    """截断恢复后，把 items 字段重命名为对应 stage 的字段

    例如 idea_generator 截断后返回 {items: [idea1, idea2]}，重命名为
    {ideas: [idea1, idea2]}，让下游代码能正常处理。
    然后用对应 stage 的 schema validator 校验 mapped data。
    """
    if result.layer != "truncated" or not result.data:
        return result
    items = result.data.pop("items", None)
    recovered_count = result.data.pop("_recovered_count", 0)
    target_key = _TRUNCATED_KEY_MAP.get(stage)
    if target_key and items is not None:
        mapped = {target_key: items, "round": 1}
        # schema 校验 mapped data
        validator = _STAGE_VALIDATORS.get(stage)
        if validator is not None:
            err = validator(mapped)
            if err is not None:
                # 校验失败：返回失败
                return ParseResult(
                    ok=False,
                    error=f"truncated recovery mapped schema failed: {err}",
                    raw=result.raw,
                )
        result.data[target_key] = items
        result.data["_recovered_count"] = recovered_count
        result.data["round"] = 1
    return result


# ==============================================================================
# 5 阶段 parse 函数
# ==============================================================================


def parse_idea_generator_output(raw: str) -> ParseResult:
    """IdeaGenerator 输出解析（带截断恢复）"""
    result = parse_json_3layer(raw, _validate_idea_generator)
    return _apply_truncation_mapping(result, "idea_generator")


def parse_formula_translator_output(raw: str) -> ParseResult:
    """FormulaTranslator 输出解析（带截断恢复）"""
    result = parse_json_3layer(raw, _validate_formula_translator)
    return _apply_truncation_mapping(result, "formula_translator")


def parse_evaluator_output(raw: str) -> ParseResult:
    """Evaluator 输出解析（带截断恢复）"""
    result = parse_json_3layer(raw, _validate_evaluator)
    return _apply_truncation_mapping(result, "evaluator")


def parse_reflector_output(raw: str) -> ParseResult:
    """Reflector 输出解析（带截断恢复）"""
    result = parse_json_3layer(raw, _validate_reflector)
    return _apply_truncation_mapping(result, "reflector")


def parse_critic_output(raw: str) -> ParseResult:
    """Critic 输出解析（带截断恢复）"""
    result = parse_json_3layer(raw, _validate_critic)
    return _apply_truncation_mapping(result, "critic")


def parse_critic_output(raw: str) -> ParseResult:
    """Critic 输出解析"""
    return parse_json_3layer(raw, _validate_critic)


# ==============================================================================
# 公式白名单校验（FormulaTranslator 专用）
# ==============================================================================

# ALLOWED_OPERATORS 实际定义在 types/constants.py（叶子包），
# 此处 re-export 保持向后兼容。
from QuantNodes.research.quant_alpha.types.constants import ALLOWED_OPERATORS


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
    "ThinkingRecord",
    "parse_thinking_block",
]


# ==============================================================================
# 思维链结构化解析（Tier 1+2：feature/thinking-chain）
# ==============================================================================


@dataclass
class ThinkingRecord:
    """从 LLM <think> 块提取的结构化推理字段。

    Attributes:
        raw: 原始 thinking 文本
        hypothesis: 经济假设（一句话）
        mechanism: 经济学机制（为什么有效）
        operator_rationale: 算子选择理由
        parameter_rationale: 参数选择理由
        risk: 风险因素
        suggested_ops: LLM 显式建议的算子（SUGGESTED_OPS 字段）
        mentioned_ops: thinking 文本中提及的、属于 op_vocab 的算子
        key_insights: 反射器输出的核心洞察列表
        next_round_focus: 反射器建议的下轮焦点
        risk_patterns: 反射器发现的失效模式
        selection_criteria: 评论家选择标准
        diversity: 评论家多样性考虑
        risk_filters: 评论家风险过滤
    """

    raw: str = ""
    hypothesis: str = ""
    mechanism: str = ""
    operator_rationale: str = ""
    parameter_rationale: str = ""
    risk: str = ""
    suggested_ops: List[str] = field(default_factory=list)
    mentioned_ops: List[str] = field(default_factory=list)
    # 反射器（reflector）专用
    key_insights: List[str] = field(default_factory=list)
    next_round_focus: str = ""
    risk_patterns: str = ""
    # 评论家（critic）专用
    selection_criteria: str = ""
    diversity: str = ""
    risk_filters: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw": self.raw,
            "hypothesis": self.hypothesis,
            "mechanism": self.mechanism,
            "operator_rationale": self.operator_rationale,
            "parameter_rationale": self.parameter_rationale,
            "risk": self.risk,
            "suggested_ops": self.suggested_ops,
            "mentioned_ops": self.mentioned_ops,
            "key_insights": self.key_insights,
            "next_round_focus": self.next_round_focus,
            "risk_patterns": self.risk_patterns,
        }


def parse_thinking_block(
    thinking_text: Optional[str],
    op_vocab: Optional[set] = None,
) -> ThinkingRecord:
    """从 LLM <think> 块提取结构化字段。

    Tier 1+2 实现：解析 LLM 在 thinking 块中按结构化指令输出的字段。
    所有字段都是 Optional → 缺失时返回空串/空 list，向后兼容。

    Args:
        thinking_text: <think> 块的文本（不含标签）
        op_vocab: 算子词表（用于 mentioned_ops 过滤）

    Returns:
        ThinkingRecord
    """
    if not thinking_text:
        return ThinkingRecord(raw="")

    result = ThinkingRecord(raw=thinking_text)

    field_pattern = (
        r"(?:^|\n)\s*[-*]?\s*"
        r"(HYPOTHESIS|MECHANISM|OPERATOR_RATIONALE|"
        r"PARAMETER_RATIONALE|RISK|"
        r"KEY_INSIGHTS|NEXT_ROUND_FOCUS|RISK_PATTERNS|"
        r"SELECTION_CRITERIA|DIVERSITY|RISK_FILTERS)\s*:\s*"
        r"(.+?)(?=\n\s*[-*]?[A-Z_]+:|$)"
    )
    matches = re.findall(field_pattern, thinking_text, re.DOTALL)
    for key, value in matches:
        value = value.strip()
        if key == "HYPOTHESIS":
            result.hypothesis = value
        elif key == "MECHANISM":
            result.mechanism = value
        elif key == "OPERATOR_RATIONALE":
            result.operator_rationale = value
        elif key == "PARAMETER_RATIONALE":
            result.parameter_rationale = value
        elif key == "RISK":
            result.risk = value
        elif key == "KEY_INSIGHTS":
            # 反射器专用，存为 key_insights list
            result.key_insights = [
                line.strip().lstrip("-*").strip()
                for line in value.split("\n")
                if line.strip() and not line.strip().startswith("NEXT")
            ]
        elif key == "NEXT_ROUND_FOCUS":
            result.next_round_focus = value
        elif key == "RISK_PATTERNS":
            result.risk_patterns = value
        elif key == "SELECTION_CRITERIA":
            result.selection_criteria = value
        elif key == "DIVERSITY":
            result.diversity = value
        elif key == "RISK_FILTERS":
            result.risk_filters = value

    # SUGGESTED_OPS 是单行字段（值不应跨行）
    ops_match = re.search(r"SUGGESTED_OPS:\s*([^\n]+)", thinking_text)
    if ops_match:
        result.suggested_ops = [
            s.strip() for s in ops_match.group(1).split(",") if s.strip()
        ]

    if op_vocab:
        # 提取算子提及：两种方式
        # 1) 出现在 SUGGESTED_OPS 列表中（已在 suggested_ops 中）
        # 2) 出现在 text 中作为算子调用（带括号）
        # 3) 作为独立单词出现在文本中（用于 narrative mention）
        ops_called = set(re.findall(r"\b([a-zA-Z_]\w*)\s*\(", thinking_text))
        ops_words = set(re.findall(r"\b([a-zA-Z_]\w*)\b", thinking_text))
        all_mentioned = ops_called | ops_words
        result.mentioned_ops = sorted(
            op for op in all_mentioned if op in op_vocab
        )

    return result
