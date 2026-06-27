# coding=utf-8
"""
generator.py - 外层循环的两个 Agent

基于 AlphaLogics 论文 (arXiv 2603.20247) §3.3 实现。

两个外层 Agent:
  - MarketLogicGeneratorAgent: 生成新/重构逻辑
  - MarketLogicRefinementDirectionAgent: 逻辑层反馈

Usage::

    from QuantNodes.research.quant_alpha.logic_mining.generator import (
        MarketLogicGenerator, MarketLogicRefinementDirection,
    )

    generator = MarketLogicGenerator(llm_client=llm)
    new_logic = generator.generate(library, history, evidence)

    refiner = MarketLogicRefinementDirection(llm_client=llm)
    feedback = refiner.refine(current_logic, history, evidence)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from QuantNodes.research.quant_alpha.logic_mining.models import (
    LogicBehavior,
    LogicCondition,
    LogicPerformanceEvidence,
    WikiLogicStructured,
)
from QuantNodes.research.quant_alpha.logic_mining.parser import (
    parse_json_response,
)

# WikiLogic 在 research.wiki 中定义，会引用 logic_mining.models，
# 故在此延迟导入以避免循环依赖。
def _get_wiki_logic_class():
    from QuantNodes.research.wiki import WikiLogic
    return WikiLogic

def _get_logic_source_class():
    from QuantNodes.research.wiki import LogicSource
    return LogicSource

logger = logging.getLogger(__name__)

__all__ = [
    "MarketLogicGenerator",
    "MarketLogicRefinementDirection",
    "generate_logic_name",
]


def _call_llm(llm_client: Any, agent_id: str, prompt: str, default_response: str) -> str:
    """调用 LLM，无客户端或失败时返回 mock"""
    if llm_client is None:
        return default_response
    try:
        if hasattr(llm_client, "complete"):
            return llm_client.complete(agent_id=agent_id, prompt=prompt)
        return llm_client(prompt)
    except Exception as e:
        logger.warning("LLM call failed for %s: %s, falling back to mock", agent_id, e)
        return default_response


def _build_generator_prompt(
    library: List[WikiLogic],
    current_logic: Optional[WikiLogic],
    history: List[WikiLogic],
    evidence: List[LogicPerformanceEvidence],
    round_idx: int,
) -> str:
    """MarketLogicGeneratorAgent prompt"""
    lib_names = [l.name for l in library[:10]]
    history_names = [l.name for l in history[:10]]
    ev_summary = [
        {"round": e.refinement_round, "best_ir": e.best_ir, "n": e.n_factors_explored}
        for e in evidence[:10]
    ]

    return (
        f"Generate a NEW market logic H_new for round {round_idx}.\n\n"
        f"Current library (top 10): {lib_names}\n"
        f"History: {history_names}\n"
        f"Evidence history: {ev_summary}\n\n"
        f"Output STRICT JSON:\n"
        f"{{\n"
        f'  "name": "logic_name_v{round_idx}",\n'
        f'  "predicates": [{{"variable": "close", "op": "ts_mean", '
        f'"threshold": 0, "window": 20}}],\n'
        f'  "behavior": {{"target": "forward_return_5", "direction": -1, "horizon": 5}},\n'
        f'  "operator_whitelist": ["ts_mean", "rank", "sub", "div"],\n'
        f'  "parameter_ranges": {{"ts_mean": [5, 60]}},\n'
        f'  "sign_constraint": -1\n'
        f"}}"
    )


def _build_refiner_prompt(
    current_logic: WikiLogic,
    history: List[WikiLogic],
    evidence: List[LogicPerformanceEvidence],
) -> str:
    """MarketLogicRefinementDirectionAgent prompt"""
    ev_current = current_logic.performance_evidence
    current_ir = ev_current.best_ir if ev_current else 0.0

    return (
        f"Provide refinement direction for current logic.\n\n"
        f"Current logic: {current_logic.name}\n"
        f"Current best_ir: {current_ir}\n"
        f"History: {[l.name for l in history[:5]]}\n"
        f"Evidence: {[(e.refinement_round, e.best_ir) for e in evidence[:5]]}\n\n"
        f"Output STRICT JSON:\n"
        f"{{\n"
        f'  "diagnosis": "logic_too_broad or logic_too_narrow or well_calibrated",\n'
        f'  "direction": "tighten_threshold or broaden_operators or refine_window",\n'
        f'  "suggested_changes": {{"parameter_ranges": {{"ts_mean": [10, 30]}}}}\n'
        f"}}"
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


def generate_logic_name(base: str, round_idx: int) -> str:
    """生成逻辑名称"""
    return f"{base}_v{round_idx}"


@dataclass
class MarketLogicGenerator:
    """MarketLogicGeneratorAgent

    在初始轮基于 ℋ_init 发散生成；后续轮基于 ℋ_lib + 历史证据做有方向的生成/重构。
    """
    llm_client: Any = None
    base_name: str = "alpha_logic"

    def generate(
        self,
        library: List[WikiLogic],
        current_logic: Optional[WikiLogic] = None,
        history: Optional[List[WikiLogic]] = None,
        evidence: Optional[List[LogicPerformanceEvidence]] = None,
        round_idx: int = 1,
    ) -> WikiLogic:
        """生成新逻辑

        Args:
            library: 当前全部逻辑库 ℋ_lib
            current_logic: 当前逻辑 H_current
            history: 历史逻辑列表
            evidence: 历史证据列表
            round_idx: 轮次

        Returns:
            新的 WikiLogic
        """
        history = history or []
        evidence = evidence or []

        # Mock 响应：基于 evidence 趋势生成
        mock_response = self._mock_generate_response(
            library, current_logic, history, evidence, round_idx
        )

        # 调用 LLM
        prompt = _build_generator_prompt(library, current_logic, history, evidence, round_idx)
        raw = _call_llm(
            self.llm_client, "market-logic-generator", prompt, mock_response
        )
        result = parse_json_response(raw)
        if not result.ok:
            logger.warning("MarketLogicGenerator parse failed: %s, using mock data", result.error)
            data = json.loads(mock_response)
        else:
            data = result.data

        # 构建 WikiLogic
        name = data.get("name") or generate_logic_name(self.base_name, round_idx)
        try:
            structured = _structured_from_dict(data)
        except Exception as e:
            logger.warning("Failed to build WikiLogicStructured: %s", e)
            structured = None

        parent = current_logic.name if current_logic else None
        WikiLogic = _get_wiki_logic_class()
        LogicSource = _get_logic_source_class()
        return WikiLogic(
            name=name,
            content=f"Auto-generated logic at round {round_idx}",
            source=LogicSource.RESEARCH_REPORT,
            extracted_formula=None,
            validation_status="pending",
            structured=structured,
            parent_logic=parent,
            refinement_round=round_idx,
            created_at=datetime.now().isoformat(),
        )

    def _mock_generate_response(
        self,
        library: List[WikiLogic],
        current_logic: Optional[WikiLogic],
        history: List[WikiLogic],
        evidence: List[LogicPerformanceEvidence],
        round_idx: int,
    ) -> str:
        """Mock 响应：基于历史证据趋势生成"""
        # 基于当前逻辑变体
        if current_logic and current_logic.structured:
            base_predicates = [p.to_dict() for p in current_logic.structured.predicates]
            beh = current_logic.structured.behavior.to_dict()
            whitelist = current_logic.structured.operator_whitelist or ["rank"]
            param_ranges = None
            if current_logic.structured.parameter_ranges:
                param_ranges = {
                    k: list(v)
                    for k, v in current_logic.structured.parameter_ranges.items()
                }
            sign = current_logic.structured.sign_constraint
        else:
            base_predicates = [{"variable": "close", "op": "ts_mean", "threshold": 0, "window": 20}]
            beh = {"target": "forward_return_5", "direction": -1, "horizon": 5}
            whitelist = ["ts_mean", "rank", "sub", "div", "sign"]
            param_ranges = {"ts_mean": [5, 60]}
            sign = -1

        # 根据 evidence 调整
        if evidence and len(evidence) >= 2:
            # IR 提升则保持方向，否则反转
            if evidence[-1].best_ir > evidence[-2].best_ir:
                # 继续优化
                pass
            else:
                # 反转方向
                sign = -sign if sign else 1

        return json.dumps({
            "name": generate_logic_name(self.base_name, round_idx),
            "predicates": base_predicates,
            "behavior": beh,
            "operator_whitelist": whitelist,
            "parameter_ranges": param_ranges or {},
            "sign_constraint": sign,
        })


@dataclass
class MarketLogicRefinementDirection:
    """MarketLogicRefinementDirectionAgent

    综合所有该逻辑名下因子的回测表现，识别"逻辑过宽/过窄/与市场结构错配"的部分。
    """
    llm_client: Any = None

    def refine(
        self,
        current_logic: WikiLogic,
        history: Optional[List[WikiLogic]] = None,
        evidence: Optional[List[LogicPerformanceEvidence]] = None,
    ) -> Dict[str, Any]:
        """生成逻辑层反馈

        Args:
            current_logic: 当前逻辑
            history: 历史逻辑
            evidence: 历史证据

        Returns:
            {
                "diagnosis": str,
                "direction": str,
                "suggested_changes": Dict
            }
        """
        history = history or []
        evidence = evidence or []

        # Mock 响应
        mock_response = self._mock_refine_response(current_logic, evidence)

        # 调用 LLM
        prompt = _build_refiner_prompt(current_logic, history, evidence)
        raw = _call_llm(
            self.llm_client, "market-logic-refinement", prompt, mock_response
        )
        result = parse_json_response(raw)
        if not result.ok:
            logger.warning("Refinement parse failed: %s, using mock", result.error)
            data = json.loads(mock_response)
        else:
            data = result.data

        return {
            "diagnosis": data.get("diagnosis", "well_calibrated"),
            "direction": data.get("direction", "no_change"),
            "suggested_changes": data.get("suggested_changes", {}),
        }

    def _mock_refine_response(
        self,
        current_logic: WikiLogic,
        evidence: List[LogicPerformanceEvidence],
    ) -> str:
        """Mock 响应：基于 IR 趋势给出建议"""
        if not evidence:
            diagnosis = "well_calibrated"
            direction = "no_change"
        elif len(evidence) >= 2 and evidence[-1].best_ir < evidence[-2].best_ir:
            diagnosis = "logic_too_broad"
            direction = "tighten_threshold"
        elif len(evidence) >= 2 and evidence[-1].best_ir == evidence[-2].best_ir:
            diagnosis = "saturated"
            direction = "refine_window"
        else:
            diagnosis = "well_calibrated"
            direction = "no_change"

        return json.dumps({
            "diagnosis": diagnosis,
            "direction": direction,
            "suggested_changes": {
                "parameter_ranges": {"ts_mean": [10, 30]},
            },
        })