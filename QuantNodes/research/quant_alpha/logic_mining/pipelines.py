# coding=utf-8
"""
pipelines.py - Logic Mining 三段式 Agent Pipeline

基于 AlphaLogics 论文 (arXiv 2603.20247) §3.1 实现。

三段式流程:
  FormulaStructureAgent → FinancialSemanticsMappingAgent → MarketLogicAbstractionAgent
  → WikiLogicStructured

Usage::

    from QuantNodes.research.quant_alpha.logic_mining.pipelines import (
        LogicMiningPipeline, mine_logic_from_formula, build_initial_logic_library,
    )

    # 单条公式
    logic = mine_logic_from_formula(
        formula="-ts_corr(rank(open), rank(volume), 10)",
        source_lib="alpha101",
    )

    # 批量构建逻辑库
    logics = build_initial_logic_library(
        source_libs=("alpha101", "alpha158"),
        max_per_lib=20,
    )
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from QuantNodes.research.quant_alpha.logic_mining.models import (
    LogicBehavior,
    LogicCondition,
    LogicAbstractionResult,
    WikiLogicStructured,
)
from QuantNodes.research.quant_alpha.logic_mining.parser import (
    parse_financial_semantics,
    parse_formula_structure,
    parse_market_logic,
    _mock_structure_response,
    _mock_semantics_response,
    _mock_abstraction_response,
)
from QuantNodes.research.quant_alpha.logic_mining.metrics import (
    LogicMiningStrictError,
    PipelineMetrics,
    StrictConfig,
)
from QuantNodes.research.quant_alpha.logic_mining.sources import get_formulas_from_source

logger = logging.getLogger(__name__)

__all__ = [
    "LogicMiningPipeline",
    "mine_logic_from_formula",
    "build_initial_logic_library",
]


def _call_llm(
    llm_client: Any,
    agent_id: str,
    prompt: str,
    default_response: str,
    metrics: Optional[PipelineMetrics] = None,
    strict: Optional[StrictConfig] = None,
) -> str:
    """调用 LLM，无客户端或失败时返回 mock

    v3.0.1 (Phase 2): metrics 接入与 strict 模式开关
    - llm_client is None            → 直接返 mock (不算失败)
    - llm_client 抛异常              → metrics.call_failures[agent_id]+1
                                       strict.call 时抛 LogicMiningStrictError
    """
    if llm_client is None:
        return default_response

    try:
        if hasattr(llm_client, "complete"):
            return llm_client.complete(agent_id=agent_id, prompt=prompt)
        return llm_client(prompt)
    except Exception as e:
        logger.warning("LLM call failed for %s: %s, falling back to mock", agent_id, e)
        if metrics is not None:
            metrics.record_call_failure(agent_id)
        if strict is not None and strict.call:
            raise LogicMiningStrictError(
                f"LLM call failed for {agent_id}: {e}",
                kind="call",
                agent_id=agent_id,
                original_error=repr(e),
            ) from e
        return default_response


def _build_structure_prompt(formula: str) -> str:
    """FormulaStructureAgent prompt"""
    return (
        f"Analyze the structure of this alpha formula:\n"
        f"  {formula}\n\n"
        f"Output STRICT JSON with these fields:\n"
        f"- operations: list of function names used\n"
        f"- window_length: largest numeric window parameter\n"
        f"- has_ranking: true if rank() is used\n"
        f"- has_normalization: true if zscore/normalize is used\n\n"
        f"Example output:\n"
        f'{{"operations": ["rank", "ts_corr", "sign"], '
        f'"window_length": 10, "has_ranking": true, "has_normalization": false}}'
    )


def _build_semantics_prompt(formula: str, structure: Dict[str, Any]) -> str:
    """FinancialSemanticsMappingAgent prompt"""
    ops = structure.get("operations", [])
    return (
        f"Given formula: {formula}\n"
        f"Operations: {ops}\n\n"
        f"Map to financial semantics. Output STRICT JSON:\n"
        f"- price_role: role of price (e.g., 'trend', 'reversion', 'momentum')\n"
        f"- volume_role: role of volume ('participation', 'confirmation', 'not used')\n"
        f"- time_pattern: time pattern ('windowed', 'moving average', 'cumulative')\n"
        f"- behavior_interpretation: behavior interpretation (e.g., 'divergence signal')\n\n"
        f'Example: {{"price_role": "trend", "volume_role": "participation", '
        f'"time_pattern": "windowed", "behavior_interpretation": "divergence signal"}}'
    )


def _build_abstraction_prompt(
    formula: str,
    structure: Dict[str, Any],
    semantics: Dict[str, Any],
) -> str:
    """MarketLogicAbstractionAgent prompt"""
    return (
        f"Abstract this alpha into a formal market logic H = ⟨𝒞, ℬ⟩.\n\n"
        f"Formula: {formula}\n"
        f"Structure: {structure}\n"
        f"Semantics: {semantics}\n\n"
        f"Output STRICT JSON:\n"
        f"- predicates: list of {{variable, op, threshold, window, second_variable}}\n"
        f"- behavior: {{target, direction, horizon}}  (target='forward_return_5', direction=+1/-1)\n"
        f"- operator_whitelist: list of allowed operators\n"
        f"- parameter_ranges: {{op: [min, max]}}\n"
        f"- sign_constraint: +1/-1 or null\n\n"
        f'Example: {{"predicates": [{{"variable": "open", "op": "ts_corr", '
        f'"threshold": -0.5, "window": 10, "second_variable": "volume"}}], '
        f'"behavior": {{"target": "forward_return_5", "direction": -1, "horizon": 5}}, '
        f'"operator_whitelist": ["rank", "ts_corr", "sign"], '
        f'"parameter_ranges": {{"ts_corr": [5, 30]}}, '
        f'"sign_constraint": -1}}'
    )


def _structured_from_dict(data: Dict[str, Any]) -> WikiLogicStructured:
    """从 dict 构建 WikiLogicStructured"""
    predicates = []
    for p in data.get("predicates", []):
        predicates.append(LogicCondition(
            variable=p["variable"],
            op=p["op"],
            threshold=p.get("threshold", 0.0),
            window=p.get("window"),
            weight=p.get("weight", 1.0),
            second_variable=p.get("second_variable"),
        ))

    beh_data = data.get("behavior", {})
    behavior = LogicBehavior(
        target=beh_data.get("target", "forward_return_5"),
        direction=beh_data.get("direction", 1),
        horizon=beh_data.get("horizon", 5),
    )

    param_ranges = None
    if data.get("parameter_ranges"):
        param_ranges = {k: tuple(v) for k, v in data["parameter_ranges"].items()}

    return WikiLogicStructured(
        predicates=predicates,
        behavior=behavior,
        operator_whitelist=data.get("operator_whitelist"),
        parameter_ranges=param_ranges,
        sign_constraint=data.get("sign_constraint"),
    )


class LogicMiningPipeline:
    """Logic Mining 三段式 Pipeline

    FormulaStructureAgent → FinancialSemanticsMappingAgent → MarketLogicAbstractionAgent

    v3.0.1 (Phase 2): 新增 metrics/strict 入参; 任何 silent fallback
    都通过 metrics 或 LogicMiningStrictError 对外暴露
    """

    def __init__(
        self,
        llm_client: Any = None,
        metrics: Optional[PipelineMetrics] = None,
        strict: Optional[StrictConfig] = None,
    ):
        self.llm_client = llm_client
        self.metrics = metrics or PipelineMetrics()
        self.strict = strict or StrictConfig()

    def run(self, formula: str, source_lib: str = "alpha101") -> LogicAbstractionResult:
        """运行三段式 Pipeline

        Args:
            formula: 因子公式字符串
            source_lib: 来源库名称

        Returns:
            LogicAbstractionResult — 含 parse_error / parse_layer (失败时)
        """
        result = LogicAbstractionResult(
            source_formula=formula,
            source_lib=source_lib,
        )

        # Step 1: FormulaStructureAgent
        structure_resp = _call_llm(
            self.llm_client,
            "logic-mining-structure",
            _build_structure_prompt(formula),
            _mock_structure_response(formula),
            metrics=self.metrics,
            strict=self.strict,
        )
        struct_result = parse_formula_structure(structure_resp)
        if not struct_result.ok:
            logger.warning("FormulaStructureAgent parse failed: %s", struct_result.error)
            self.metrics.record_parse_failure("logic-mining-structure", struct_result.layer_reached)
            result.parse_error = struct_result.error
            result.parse_layer = struct_result.layer_reached
            if self.strict.parse:
                raise LogicMiningStrictError(
                    f"FormulaStructureAgent parse failed at layer {struct_result.layer_reached}: "
                    f"{struct_result.error}",
                    kind="parse",
                    agent_id="logic-mining-structure",
                    layer=struct_result.layer_reached,
                    last_error=struct_result.last_error,
                )
            result.formula_structure = {"operations": [], "window_length": 0,
                                        "has_ranking": False, "has_normalization": False}
        else:
            result.formula_structure = struct_result.data

        # Step 2: FinancialSemanticsMappingAgent
        semantics_resp = _call_llm(
            self.llm_client,
            "logic-mining-semantics",
            _build_semantics_prompt(formula, result.formula_structure),
            _mock_semantics_response(formula),
            metrics=self.metrics,
            strict=self.strict,
        )
        sem_result = parse_financial_semantics(semantics_resp)
        if not sem_result.ok:
            logger.warning("FinancialSemanticsMappingAgent parse failed: %s", sem_result.error)
            self.metrics.record_parse_failure("logic-mining-semantics", sem_result.layer_reached)
            result.parse_error = sem_result.error
            result.parse_layer = sem_result.layer_reached
            if self.strict.parse:
                raise LogicMiningStrictError(
                    f"FinancialSemanticsMappingAgent parse failed at layer {sem_result.layer_reached}: "
                    f"{sem_result.error}",
                    kind="parse",
                    agent_id="logic-mining-semantics",
                    layer=sem_result.layer_reached,
                    last_error=sem_result.last_error,
                )
            result.financial_semantics = {
                "price_role": "unknown", "volume_role": "unknown",
                "time_pattern": "unknown", "behavior_interpretation": "unknown",
            }
        else:
            result.financial_semantics = sem_result.data

        # Step 3: MarketLogicAbstractionAgent
        abstract_resp = _call_llm(
            self.llm_client,
            "logic-mining-abstraction",
            _build_abstraction_prompt(formula, result.formula_structure, result.financial_semantics),
            _mock_abstraction_response(formula, result.formula_structure, result.financial_semantics),
            metrics=self.metrics,
            strict=self.strict,
        )
        abs_result = parse_market_logic(abstract_resp)
        if not abs_result.ok:
            logger.warning("MarketLogicAbstractionAgent parse failed: %s", abs_result.error)
            self.metrics.record_parse_failure("logic-mining-abstraction", abs_result.layer_reached)
            result.parse_error = abs_result.error
            result.parse_layer = abs_result.layer_reached
            if self.strict.parse:
                raise LogicMiningStrictError(
                    f"MarketLogicAbstractionAgent parse failed at layer {abs_result.layer_reached}: "
                    f"{abs_result.error}",
                    kind="parse",
                    agent_id="logic-mining-abstraction",
                    layer=abs_result.layer_reached,
                    last_error=abs_result.last_error,
                )
            return result

        # Stage 3 JSON 合法,构建 structured_logic (仍可能抛 KeyError/TypeError)
        try:
            result.structured_logic = _structured_from_dict(abs_result.data)
        except (KeyError, TypeError) as e:
            logger.warning("Failed to build WikiLogicStructured: %s", e)
            self.metrics.record_structured_failure("logic-mining-abstraction")
            result.parse_error = f"structured build failed: {e}"
            if self.strict.structured:
                raise LogicMiningStrictError(
                    f"WikiLogicStructured build failed: {e}",
                    kind="structured",
                    agent_id="logic-mining-abstraction",
                    original_error=repr(e),
                ) from e

        return result


def mine_logic_from_formula(
    formula: str,
    source_lib: str = "alpha101",
    llm_client: Any = None,
    metrics: Optional[PipelineMetrics] = None,
    strict: Optional[StrictConfig] = None,
) -> LogicAbstractionResult:
    """从单条公式抽取市场逻辑（三段式）

    Args:
        formula: 因子公式字符串
        source_lib: 来源库
        llm_client: LLM 客户端 (None 时使用 mock)
        metrics: 可观测性指标 (v3.0.1)
        strict:  严格模式开关 (v3.0.1)

    Returns:
        LogicAbstractionResult
    """
    pipeline = LogicMiningPipeline(llm_client=llm_client, metrics=metrics, strict=strict)
    return pipeline.run(formula, source_lib)


def build_initial_logic_library(
    source_libs: Tuple[str, ...] = ("alpha101", "alpha158"),
    llm_client: Any = None,
    max_per_lib: int = 20,
    only_volume_price: bool = True,
    metrics: Optional[PipelineMetrics] = None,
    strict: Optional[StrictConfig] = None,
) -> List[LogicAbstractionResult]:
    """构建初始逻辑库

    Args:
        source_libs: 来源库列表
        llm_client: LLM 客户端
        max_per_lib: 每个库最多提取多少条
        only_volume_price: 仅提取量价类
        metrics: 可观测性指标 (v3.0.1)
        strict:  严格模式开关 (v3.0.1)

    Returns:
        List of LogicAbstractionResult
    """
    metrics = metrics or PipelineMetrics()
    pipeline = LogicMiningPipeline(llm_client=llm_client, metrics=metrics, strict=strict)
    results = []

    for lib in source_libs:
        formulas = get_formulas_from_source(
            lib, max_count=max_per_lib, only_volume_price=only_volume_price
        )
        logger.info("Mining %d formulas from %s", len(formulas), lib)

        for f in formulas:
            try:
                result = pipeline.run(f["formula"], lib)
                if result.structured_logic is not None:
                    results.append(result)
            except LogicMiningStrictError:
                # strict 模式下 swallow exception 反而违背 strict 设计
                # 让上层显式感知单个公式失败
                logger.warning(
                    "Strict-mode failure for %s in %s",
                    f.get("id"), lib,
                )
                raise
            except Exception as e:
                logger.warning("Failed to mine logic for %s: %s", f.get("id"), e)

    logger.info("Built initial logic library: %d logics", len(results))
    return results