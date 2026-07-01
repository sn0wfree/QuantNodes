# coding=utf-8
"""MCTS WorkflowSpec — 用 StepAgent 框架封装 MCTS 搜索。

3 个 StepAgentSpec + 注册到 WorkflowRegistry。
复用 research/quant_alpha/mcts/ 的搜索逻辑。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..step_agent import StepAgentSpec
from ..parsers import parse_json_3layer
from ..registry import WorkflowSpec, REGISTRY

from QuantNodes.research.quant_alpha.types import (
    ReflectionRecord,
    FinalFormulaRecord,
    ALLOWED_OPERATORS,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# State
# ==============================================================================


@dataclass
class MCTSState:
    """MCTS 工作流状态。"""
    objective: str = ""
    iterations_total: int = 3
    round_idx_hint: int = 1

    # 种子公式
    seed_formulas: List[str] = field(default_factory=list)

    # 搜索结果（从 tool_executor 返回）
    best_k_nodes: List[Dict[str, Any]] = field(default_factory=list)
    search_stats: Dict[str, Any] = field(default_factory=dict)

    # 反思记录
    all_reflections: List[ReflectionRecord] = field(default_factory=list)

    # 累积结果
    all_best_nodes: List[Dict[str, Any]] = field(default_factory=list)


# ==============================================================================
# Prompt builders
# ==============================================================================


def _build_seed_prompt(
    state: Any = None,
    round_idx: int = 1,
    objective: str = "",
    data_columns: Optional[List[str]] = None,
    _prev_error: Optional[str] = None,
    _prev_raw: Optional[str] = None,
    **kwargs: Any,
) -> str:
    columns = data_columns or ["close", "open", "high", "low", "vol", "vwap"]
    ops = sorted(ALLOWED_OPERATORS)

    prev_reflection = None
    if state and hasattr(state, "all_reflections") and state.all_reflections:
        prev_reflection = state.all_reflections[-1].to_dict()

    prompt = (
        f"Read .agent/agents/mcts-seed-generator.md. "
        f"Generate seed formulas for MCTS search. "
        f"objective={objective!r}. "
        f"data_columns={columns}. "
        f"available_operators={ops}. "
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


def _build_reflect_prompt(
    state: Any = None,
    round_idx: int = 1,
    _prev_error: Optional[str] = None,
    _prev_raw: Optional[str] = None,
    **kwargs: Any,
) -> str:
    tree_stats = getattr(state, "search_stats", {}) if state else {}
    top_k = getattr(state, "best_k_nodes", []) if state else []
    prev_reflection = None
    if state and hasattr(state, "all_reflections") and state.all_reflections:
        prev_reflection = state.all_reflections[-1].to_dict()

    prompt = (
        f"Read .agent/agents/mcts-reflector.md. "
        f"Analyze MCTS search results for round {round_idx}. "
        f"tree_stats={tree_stats}. "
        f"top_k_formulas={top_k[:10]}. "
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


# ==============================================================================
# Validators
# ==============================================================================


def _validate_seeds(data: Dict[str, Any]) -> bool:
    """验证种子公式输出。"""
    if not isinstance(data, dict):
        return False
    seeds = data.get("seed_formulas")
    if not isinstance(seeds, list):
        return False
    for seed in seeds:
        if not isinstance(seed, dict):
            return False
        if "formula" not in seed:
            return False
    return True


def _validate_mcts_reflection(data: Dict[str, Any]) -> bool:
    """验证 MCTS 反思输出。"""
    if not isinstance(data, dict):
        return False
    # 必须有 formula_feedback 或 next_round_suggestions
    return (
        "formula_feedback" in data or
        "next_round_suggestions" in data
    )


# ==============================================================================
# Record factories
# ==============================================================================


def _seed_factory(d: dict, **kwargs: Any) -> str:
    """种子公式：只保留 formula 字符串。"""
    return d.get("formula", "")


def _reflection_factory(d: dict, round_idx: int = 1, **kwargs: Any) -> ReflectionRecord:
    return ReflectionRecord(
        round_idx=round_idx,
        verdicts=d.get("formula_feedback", []),
        suggestions=d.get("next_round_suggestions", {}),
    )


# ==============================================================================
# tool_executor
# ==============================================================================


def _run_mcts_search(
    state: Any = None,
    data: Any = None,
    data_path: Optional[str] = None,
    iterations: int = 50,
    max_depth: int = 5,
    seed: int = 42,
    compute_ic_ir: bool = True,
    forward_returns: Optional[list] = None,
    date_column: str = "date",
    code_column: str = "code",
    **kwargs: Any,
) -> list:
    """MCTS 搜索（tool_executor）。"""
    import polars as pl
    from QuantNodes.research.quant_alpha.mcts.search import MCTSSearch, MCTSSearchConfig
    from QuantNodes.research.quant_alpha.mcts.cache import MCTSCache, MCTSCacheConfig

    # 加载数据
    if data is None and data_path:
        try:
            data = pl.read_parquet(data_path)
        except Exception:
            data = pl.read_csv(data_path)

    if data is None:
        logger.error("MCTS search: no data provided")
        return []

    # 获取种子公式
    seed_formulas = getattr(state, "seed_formulas", []) if state else []
    if not seed_formulas:
        seed_formulas = None  # 使用默认种子

    # 创建缓存
    cache_config = MCTSCacheConfig(enabled=True)
    cache = MCTSCache(cache_config)

    # 创建搜索配置
    config = MCTSSearchConfig(
        iterations=iterations,
        max_depth=max_depth,
        seed=seed,
        compute_ic_ir=compute_ic_ir,
        forward_returns=tuple(forward_returns or (1, 5, 20)),
        date_column=date_column,
        code_column=code_column,
    )

    # 执行搜索
    search = MCTSSearch(config=config, cache=cache)
    result = search.search(
        data=data,
        seed_formulas=seed_formulas,
        date_column=date_column,
        code_column=code_column,
    )

    # 转换为字典列表
    best_nodes = []
    for node in result.best_k_nodes:
        best_nodes.append({
            "entry_id": node.entry_id,
            "formula": node.formula,
            "overall_score": node.overall_score,
            "dimension_scores": node.dimension_scores,
            "depth": node.depth,
            "metadata": node.metadata,
        })

    # 更新状态
    if state:
        state.best_k_nodes = best_nodes
        state.search_stats = {
            "total_nodes": result.formula_count,
            "valid_nodes": result.valid_count,
            "rejected_nodes": result.rejected_count,
            "pruned_nodes": result.pruned_count,
            "elapsed_seconds": result.elapsed_seconds,
        }
        state.all_best_nodes.extend(best_nodes)

    return best_nodes


# ==============================================================================
# Result builder
# ==============================================================================


def _build_mcts_result(state: MCTSState, config: dict) -> dict:
    """构建 MCTS 最终结果。"""
    top_k = config.get("top_k", 10)

    # 收集所有有效节点（去重）
    seen = set()
    unique_nodes = []
    for node in state.all_best_nodes:
        formula = node.get("formula", "")
        if formula not in seen:
            seen.add(formula)
            unique_nodes.append(node)

    # 按 overall_score 排序
    unique_nodes.sort(key=lambda n: n.get("overall_score", 0.0), reverse=True)
    top_nodes = unique_nodes[:top_k]

    # 转换为 FinalFormulaRecord 格式
    final_pool = []
    for i, node in enumerate(top_nodes):
        ic_mean = node.get("metadata", {}).get("ic_mean", 0.0)
        ir = node.get("metadata", {}).get("ir", 0.0)
        final_pool.append(FinalFormulaRecord(
            rank=i + 1,
            formula_id=node.get("entry_id", ""),
            formula=node.get("formula", ""),
            ic_mean=ic_mean,
            ir=ir,
            category=_infer_category(node),
            round_discovered=node.get("depth", 0),
            selection_reason=f"MCTS score={node.get('overall_score', 0.0):.3f}",
            risk_notes=[
                ch for ch, score in node.get("dimension_scores", {}).items()
                if score < 1.0
            ],
        ))

    # 统计
    irs = [f.ir for f in final_pool if f.ir]
    summary = {
        "total_formulas": state.search_stats.get("total_nodes", 0),
        "valid_formulas": state.search_stats.get("valid_nodes", 0),
        "rejected": state.search_stats.get("rejected_nodes", 0),
        "selected": len(final_pool),
        "best_ir": max(irs) if irs else 0.0,
        "avg_ir": sum(irs) / len(irs) if irs else 0.0,
    }

    return {
        "objective": state.objective,
        "iterations_completed": len(state.all_reflections) + 1,
        "total_formulas": summary["total_formulas"],
        "final_pool": [f.to_dict() for f in final_pool],
        "summary": summary,
    }


def _infer_category(node: Dict[str, Any]) -> str:
    """从节点推断类别。"""
    formula = node.get("formula", "")
    # 检查顺序：先检查特定模式，再检查通用模式
    if " - ts_mean(" in formula or " - ts_std(" in formula:
        return "diff"
    if " / ts_lag(" in formula or " / ts_mean(" in formula:
        return "ratio"
    if "rank(" in formula or "zscore(" in formula:
        return "wrap"
    if "ts_mean(" in formula or "ts_std(" in formula or "ts_corr(" in formula:
        return "window"
    if "abs(" in formula or "log(" in formula or "sqrt(" in formula:
        return "unary"
    return "unknown"


# ==============================================================================
# Mock LLM
# ==============================================================================


def _mock_mcts_response(
    agent_id: str,
    prompt: str,
    state: Any = None,
    config: Any = None,
) -> str:
    """Mock LLM 返回。"""
    round_idx = getattr(state, "round_idx_hint", 1) if state else 1

    if "seed-generator" in agent_id:
        seeds = [
            {"formula": "rank(close)", "category": "wrap", "rationale": "截面排名"},
            {"formula": "ts_mean(close, 20)", "category": "window", "rationale": "20 日均线"},
            {"formula": "close - ts_mean(close, 20)", "category": "diff", "rationale": "偏离均线"},
        ]
        return json.dumps({"seed_formulas": seeds}, ensure_ascii=False)

    if "reflector" in agent_id:
        return json.dumps({
            "round": round_idx,
            "formula_feedback": [],
            "next_round_suggestions": {
                "preferred_operators": ["ts_rank", "ts_decay_linear"],
                "preferred_windows": [10, 20],
            },
        }, ensure_ascii=False)

    return json.dumps({})


# ==============================================================================
# StepAgentSpec 定义
# ==============================================================================


MCTS_SEED_GEN_SPEC = StepAgentSpec(
    agent_id="mcts-seed-generator",
    prompt_builder=_build_seed_prompt,
    output_parser=lambda raw: parse_json_3layer(raw, _validate_seeds),
    output_key="seed_formulas",
    state_output="seed_formulas",
    record_factory=_seed_factory,
)

MCTS_SEARCH_SPEC = StepAgentSpec(
    agent_id="mcts-search",
    prompt_builder=None,
    output_parser=None,
    output_key="mcts_result",
    state_output="best_k_nodes",
    tool_executor=_run_mcts_search,
    record_factory=None,
)

MCTS_REFLECT_SPEC = StepAgentSpec(
    agent_id="mcts-reflector",
    prompt_builder=_build_reflect_prompt,
    output_parser=lambda raw: parse_json_3layer(raw, _validate_mcts_reflection),
    output_key="reflection",
    state_output="all_reflections",
    record_factory=_reflection_factory,
    skip_on_last=True,
)


# ==============================================================================
# WorkflowSpec 注册
# ==============================================================================


MCTS_SPEC = WorkflowSpec(
    name="alpha-mcts",
    description=(
        "MCTS factor search with LLM guidance: "
        "seed generation → tree search → reflection. "
        "Config: {objective: str, iterations: int=3, data_path: str, "
        "top_k: int=10, max_depth: int=5, compute_ic_ir: bool=true}"
    ),
    steps=[MCTS_SEED_GEN_SPEC, MCTS_SEARCH_SPEC, MCTS_REFLECT_SPEC],
    iterations=3,
    final_steps=[],
    state_factory=lambda: MCTSState(objective=""),
    result_builder=_build_mcts_result,
)

REGISTRY.register(MCTS_SPEC)


__all__ = [
    "MCTS_SPEC",
    "MCTS_SEED_GEN_SPEC",
    "MCTS_SEARCH_SPEC",
    "MCTS_REFLECT_SPEC",
    "MCTSState",
]
