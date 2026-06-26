# coding=utf-8
"""AlphaGptWorkflow — 用 StepAgent 框架重写。

5 个 StepAgentSpec + 注册到 WorkflowRegistry。
复用原有 state.py 的 dataclass 和 parser.py 的验证器。
不修改原 research/quant_alpha/workflow/alpha_gpt.py。
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ..step_agent import StepAgentSpec, _run_async
from ..parsers import (
    parse_json_3layer,
    validate_idea_generator,
    validate_formula_translator,
    validate_reflector,
    validate_critic,
    validate_formula_operators,
    ALLOWED_OPERATORS,
)
from ..registry import WorkflowSpec, REGISTRY

from QuantNodes.research.quant_alpha.workflow.state import (
    AlphaGptState,
    IdeaRecord,
    FormulaRecord,
    EvaluationRecord,
    ReflectionRecord,
    FinalFormulaRecord,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Prompt builders（支持 _prev_error / _prev_raw 重试注入）
# ==============================================================================


def _build_idea_prompt(
    state: Any = None,
    round_idx: int = 1,
    pool_size: int = 10,
    a_share_focus: bool = True,
    objective: str = "",
    _prev_error: Optional[str] = None,
    _prev_raw: Optional[str] = None,
    **kwargs: Any,
) -> str:
    prev_reflection = None
    if state and state.all_reflections:
        prev_reflection = state.all_reflections[-1].to_dict()

    prompt = (
        f"Read .agent/agents/alpha-gpt-idea-generator.md. "
        f"Generate {pool_size} alpha ideas for objective={objective!r}. "
        f"round={round_idx}, a_share_focus={a_share_focus}. "
        f"previous_reflection={prev_reflection}. "
        f"Output STRICT JSON only."
    )
    if _prev_error:
        prompt += (
            f"\n\n[SYSTEM: Your previous response was not valid JSON. "
            f"Error: {_prev_error}\n"
            f"Your full previous response:\n{_prev_raw}\n"
            f"Please output ONLY a JSON object with no additional text.]"
        )
    return prompt


def _build_formula_prompt(
    prev_output: Optional[list] = None,
    round_idx: int = 1,
    a_share_focus: bool = True,
    available_operators: Optional[List[str]] = None,
    data_columns: Optional[List[str]] = None,
    _prev_error: Optional[str] = None,
    _prev_raw: Optional[str] = None,
    **kwargs: Any,
) -> str:
    ideas_payload = [i.to_dict() for i in prev_output] if prev_output else []
    ops = available_operators or sorted(ALLOWED_OPERATORS)
    columns = data_columns or ["close", "open", "high", "low", "vol", "vwap"]

    prompt = (
        f"Read .agent/agents/alpha-gpt-formula-translator.md. "
        f"Translate these ideas to polars formulas. round={round_idx}. "
        f"ideas={ideas_payload}. available_operators={ops}. "
        f"data_columns={columns}. a_share_focus={a_share_focus}. "
        f"Output STRICT JSON only."
    )
    if _prev_error:
        prompt += (
            f"\n\n[SYSTEM: Your previous response was not valid JSON. "
            f"Error: {_prev_error}\n"
            f"Your full previous response:\n{_prev_raw}\n"
            f"Please output ONLY a JSON object with no additional text.]"
        )
    return prompt


def _build_reflector_prompt(
    prev_output: Optional[list] = None,
    round_idx: int = 1,
    _prev_error: Optional[str] = None,
    _prev_raw: Optional[str] = None,
    **kwargs: Any,
) -> str:
    evaluations = [e.to_dict() for e in prev_output] if prev_output else []

    prompt = (
        f"Read .agent/agents/alpha-gpt-reflector.md. "
        f"Reflect on round {round_idx} evaluations. "
        f"evaluations={evaluations}. "
        f"Output STRICT JSON only."
    )
    if _prev_error:
        prompt += (
            f"\n\n[SYSTEM: Your previous response was not valid JSON. "
            f"Error: {_prev_error}\n"
            f"Your full previous response:\n{_prev_raw}\n"
            f"Please output ONLY a JSON object with no additional text.]"
        )
    return prompt


def _build_critic_prompt(
    state: Any = None,
    top_k: int = 10,
    min_ir_threshold: float = 0.5,
    max_mutual_ic_threshold: float = 0.7,
    _prev_error: Optional[str] = None,
    _prev_raw: Optional[str] = None,
    **kwargs: Any,
) -> str:
    all_evaluations = [e.to_dict() for e in (state.all_evaluations if state else [])]
    all_reflections = [r.to_dict() for r in (state.all_reflections if state else [])]

    prompt = (
        f"Read .agent/agents/alpha-gpt-critic.md. "
        f"Select final top-{top_k} from all rounds. "
        f"min_ir_threshold={min_ir_threshold}. "
        f"max_mutual_ic_threshold={max_mutual_ic_threshold}. "
        f"all_evaluations={all_evaluations}. all_reflections={all_reflections}. "
        f"Output STRICT JSON only."
    )
    if _prev_error:
        prompt += (
            f"\n\n[SYSTEM: Your previous response was not valid JSON. "
            f"Error: {_prev_error}\n"
            f"Your full previous response:\n{_prev_raw}\n"
            f"Please output ONLY a JSON object with no additional text.]"
        )
    return prompt


# ==============================================================================
# Record factories
# ==============================================================================


def _idea_factory(d: dict, round_idx: int = 1, **kwargs: Any) -> IdeaRecord:
    return IdeaRecord.from_dict(d, round_idx)


def _formula_factory(
    d: dict,
    round_idx: int = 1,
    formula_counter: Optional[list] = None,
    **kwargs: Any,
) -> Optional[FormulaRecord]:
    formula_str = d.get("formula", "")
    err = validate_formula_operators(formula_str)
    if err:
        logger.debug("formula op-validation failed: %s (%s)", formula_str, err)
        return None

    if formula_counter is not None:
        idx = formula_counter[0]
        formula_counter[0] += 1
    else:
        idx = 1

    return FormulaRecord(
        formula_id=f"FORMULA-{round_idx}-{idx}",
        idea_id=d.get("idea_id", ""),
        formula=formula_str,
        round_discovered=round_idx,
        complexity=d.get("complexity", 0),
        a_share_compatible=d.get("a_share_compatible", True),
    )


def _evaluation_factory(
    d: dict,
    prev_output: Optional[list] = None,
    **kwargs: Any,
) -> EvaluationRecord:
    formula_idx = d.get("formula_id", "")
    formula_str = d.get("formula", "")

    if d.get("status") == "success":
        metrics = d.get("metrics", {})
        return EvaluationRecord(
            formula_id=formula_idx,
            formula=formula_str,
            status="success",
            ic_mean=metrics.get("ic_mean", 0.0),
            ic_std=metrics.get("ic_std", 0.0),
            ir=metrics.get("ir", 0.0),
            ic_decay=metrics.get("ic_decay", {}),
        )
    return EvaluationRecord(
        formula_id=formula_idx,
        formula=formula_str,
        status=d.get("status", "failed"),
        error_msg=d.get("error_msg"),
    )


def _reflection_factory(d: dict, round_idx: int = 1, **kwargs: Any) -> ReflectionRecord:
    return ReflectionRecord(
        round_idx=round_idx,
        verdicts=d.get("formula_feedback", []),
        suggestions=d.get("next_round_suggestions", {}),
    )


def _critic_factory(d: dict, **kwargs: Any) -> dict:
    """Critic 输出是 dict (不是 list)，直接返回。"""
    return d


# ==============================================================================
# Evaluator tool_executor
# ==============================================================================


def _run_evaluator(
    prev_output: Optional[list] = None,
    data: Any = None,
    data_path: Optional[str] = None,
    forward_returns: Optional[list] = None,
    date_column: str = "date",
    code_column: str = "code",
    **kwargs: Any,
) -> List[EvaluationRecord]:
    """直接调用 AlphaEvaluateTool，跳过 LLM。"""
    formulas = prev_output or []
    if not formulas:
        return []

    try:
        from QuantNodes.agent.tools.alpha_evaluate import AlphaEvaluateTool

        tool = AlphaEvaluateTool()
        formulas_str = [f.formula for f in formulas]
        result = _run_async(
            tool.execute(
                formulas=formulas_str,
                data=data,
                data_path=data_path,
                forward_returns=list(forward_returns or (1, 5, 20)),
                date_column=date_column,
                code_column=code_column,
            )
        )
    except Exception as exc:
        logger.exception("alpha_evaluate tool failed: %s", exc)
        return []

    evals_data = result.get("evaluations", [])
    out: List[EvaluationRecord] = []
    for fd, ed in zip(formulas, evals_data):
        if ed.get("status") == "success":
            metrics = ed.get("metrics", {})
            out.append(
                EvaluationRecord(
                    formula_id=fd.formula_id,
                    formula=fd.formula,
                    status="success",
                    ic_mean=metrics.get("ic_mean", 0.0),
                    ic_std=metrics.get("ic_std", 0.0),
                    ir=metrics.get("ir", 0.0),
                    ic_decay=metrics.get("ic_decay", {}),
                )
            )
        else:
            out.append(
                EvaluationRecord(
                    formula_id=fd.formula_id,
                    formula=fd.formula,
                    status=ed.get("status", "failed"),
                    error_msg=ed.get("error_msg"),
                )
            )
    return out


# ==============================================================================
# Result builder
# ==============================================================================


def _build_result(state: AlphaGptState, config: dict) -> dict:
    """构建最终结果 dict。"""
    top_k = config.get("top_k", 10)

    # 从 critic_output 或 fallback 选 top-K
    critic_pool = (state.critic_output or {}).get("final_pool") or []
    if critic_pool:
        pool_data = critic_pool[:top_k]
        final_pool = [FinalFormulaRecord.from_dict(p, i + 1) for i, p in enumerate(pool_data)]
    else:
        # Fallback: 按 IR 排序
        successful = [e for e in state.all_evaluations if e.status == "success"]
        successful.sort(key=lambda e: e.ir, reverse=True)
        top = successful[:top_k]
        final_pool = [
            FinalFormulaRecord(
                rank=i + 1,
                formula_id=e.formula_id,
                formula=e.formula,
                ic_mean=e.ic_mean,
                ir=e.ir,
                round_discovered=int(e.formula_id.split("-")[1]) if "-" in e.formula_id else 0,
                selection_reason=f"IR={e.ir:.3f} (auto-selected by fallback)",
                risk_notes=[],
            )
            for i, e in enumerate(top)
        ]

    # Summary
    successful = [e for e in state.all_evaluations if e.status == "success"]
    irs = [e.ir for e in successful]
    cat_dist: Dict[str, int] = {}
    for f in final_pool:
        cat = f.category or "unknown"
        cat_dist[cat] = cat_dist.get(cat, 0) + 1

    summary = {
        "total_evaluated": len(state.all_evaluations),
        "successful": len(successful),
        "failed": len(state.all_evaluations) - len(successful),
        "selected": len(final_pool),
        "avg_ir": float(np.mean(irs)) if irs else 0.0,
        "best_ir": float(np.max(irs)) if irs else 0.0,
        "category_distribution": cat_dist,
    }

    return {
        "objective": state.objective,
        "iterations_completed": state.iterations_total,
        "total_formulas": len(state.all_formulas),
        "final_pool": [f.to_dict() for f in final_pool],
        "summary": summary,
    }


# ==============================================================================
# Mock LLM
# ==============================================================================


def _mock_response(
    agent_id: str,
    prompt: str,
    state: Any = None,
    config: Any = None,
) -> str:
    """Mock LLM 返回，让 workflow 无 API key 也能端到端跑通。"""
    import json

    pool_size = config.get("pool_size", 10) if isinstance(config, dict) else 10
    round_idx = getattr(state, "round_idx_hint", 1) if state else 1

    if "idea-generator" in agent_id:
        categories = ["reversal", "momentum", "volatility", "value", "quality", "liquidity"]
        ideas = [
            {
                "id": f"IDEA-{round_idx}-{i+1}",
                "name": f"mock-idea-{i+1}",
                "category": categories[i % len(categories)],
                "description": f"Mock idea {i+1}",
                "expected_direction": "long",
                "suggested_lookback": 20,
                "a_share_compatible": True,
                "orthogonal_to": [],
                "complexity_hint": "simple",
            }
            for i in range(pool_size)
        ]
        return json.dumps({"round": round_idx, "ideas": ideas}, ensure_ascii=False)

    if "formula-translator" in agent_id:
        formulas = [
            {
                "id": f"FORMULA-{round_idx}-{i+1}",
                "idea_id": f"IDEA-{round_idx}-{i+1}",
                "formula": "sub(close, ts_mean(close, 10))",
                "complexity": 3,
                "a_share_compatible": True,
                "explanation": "Mock formula",
            }
            for i in range(pool_size)
        ]
        return json.dumps({"round": round_idx, "formulas": formulas}, ensure_ascii=False)

    if "reflector" in agent_id:
        return json.dumps({
            "round": round_idx,
            "formula_feedback": [],
            "next_round_suggestions": {},
        }, ensure_ascii=False)

    if "critic" in agent_id:
        return json.dumps({"final_pool": []}, ensure_ascii=False)

    return json.dumps({})


# ==============================================================================
# StepAgentSpec 定义
# ==============================================================================

IDEA_GEN_SPEC = StepAgentSpec(
    agent_id="alpha-gpt-idea-generator",
    prompt_builder=_build_idea_prompt,
    output_parser=lambda raw: parse_json_3layer(raw, validate_idea_generator),
    output_key="ideas",
    state_output="all_ideas",
    record_factory=_idea_factory,
)

FORMULA_TRANS_SPEC = StepAgentSpec(
    agent_id="alpha-gpt-formula-translator",
    prompt_builder=_build_formula_prompt,
    output_parser=lambda raw: parse_json_3layer(raw, validate_formula_translator),
    output_key="formulas",
    state_output="all_formulas",
    record_factory=_formula_factory,
)

EVALUATOR_SPEC = StepAgentSpec(
    agent_id="alpha-gpt-evaluator",
    prompt_builder=None,
    output_parser=None,
    output_key="evaluations",
    state_output="all_evaluations",
    tool_executor=_run_evaluator,
    record_factory=None,
)

REFLECTOR_SPEC = StepAgentSpec(
    agent_id="alpha-gpt-reflector",
    prompt_builder=_build_reflector_prompt,
    output_parser=lambda raw: parse_json_3layer(raw, validate_reflector),
    output_key="formula_feedback",
    state_output="all_reflections",
    record_factory=_reflection_factory,
    skip_on_last=True,
)

CRITIC_SPEC = StepAgentSpec(
    agent_id="alpha-gpt-critic",
    prompt_builder=_build_critic_prompt,
    output_parser=lambda raw: parse_json_3layer(raw, validate_critic),
    output_key="final_pool",
    state_output="critic_output",
    record_factory=_critic_factory,
)


# ==============================================================================
# WorkflowSpec 注册
# ==============================================================================

ALPHA_GPT_SPEC = WorkflowSpec(
    name="alpha-gpt",
    description=(
        "5-round alpha discovery pipeline: "
        "idea generation → formula translation → IC evaluation → reflection → critic selection. "
        "Config: {objective: str, iterations: int=5, pool_size: int=10, top_k: int=10, "
        "data_path: str, a_share_focus: bool=true, forward_returns: [int]=[1,5,20]}"
    ),
    steps=[IDEA_GEN_SPEC, FORMULA_TRANS_SPEC, EVALUATOR_SPEC, REFLECTOR_SPEC],
    iterations=5,
    final_steps=[CRITIC_SPEC],
    state_factory=lambda: AlphaGptState(objective=""),
    result_builder=_build_result,
)

REGISTRY.register(ALPHA_GPT_SPEC)


__all__ = [
    "ALPHA_GPT_SPEC",
    "IDEA_GEN_SPEC",
    "FORMULA_TRANS_SPEC",
    "EVALUATOR_SPEC",
    "REFLECTOR_SPEC",
    "CRITIC_SPEC",
]
