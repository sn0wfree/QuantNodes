# coding=utf-8
"""
alpha_gpt.py - AlphaGptWorkflow 协调器（M5 核心）

5 智能体编排 + 5 轮迭代主循环：
1. spawn alpha-gpt-idea-generator       → ideas
2. spawn alpha-gpt-formula-translator   → formulas
3. spawn alpha-gpt-evaluator            → evaluations
4. spawn alpha-gpt-reflector            → verdicts + suggestions
5. spawn alpha-gpt-critic (仅末轮)       → final_pool

LLM 调用复用 nanobot upstream（见 .agent/agents/alpha-gpt-*.md），
不引入新 LLM provider。workflow 只负责状态管理和 spawn 协调。

Usage::

    from QuantNodes.research.quant_alpha.workflow import (
        AlphaGptWorkflow, AlphaGptConfig,
    )

    config = AlphaGptConfig(
        objective="捕捉 A 股反转效应",
        iterations=5,
        pool_size=10,
        llm_provider="deepseek",
    )
    workflow = AlphaGptWorkflow(config=config, data=df)
    result = workflow.run()

    for f in result.final_pool:
        print(f.formula, f.ir)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .state import (
    AlphaGptState,
    IdeaRecord,
    FormulaRecord,
    EvaluationRecord,
    ReflectionRecord,
    FinalFormulaRecord,
)
from ..llm.parser import (
    parse_idea_generator_output,
    parse_formula_translator_output,
    parse_evaluator_output,
    parse_reflector_output,
    parse_critic_output,
    validate_formula_operators,
)

logger = logging.getLogger(__name__)


@dataclass
class AlphaGptConfig:
    """Alpha-GPT 工作流配置"""

    objective: str
    iterations: int = 5
    pool_size: int = 10
    top_k: int = 10
    min_ir_threshold: float = 0.5
    max_mutual_ic_threshold: float = 0.7

    forward_returns: Sequence[int] = (1, 5, 20)
    date_column: str = "date"
    code_column: str = "code"

    llm_provider: str = "deepseek"
    llm_model: Optional[str] = None
    temperature: float = 0.7

    spawn_timeout_seconds: float = 30.0

    a_share_focus: bool = True
    enable_backtest: bool = False
    top_k_backtest: int = 10

    custom_few_shot: Optional[List[Dict[str, Any]]] = None


@dataclass
class AlphaGptResult:
    """Alpha-GPT 工作流最终结果"""

    objective: str
    iterations_completed: int
    total_formulas: int
    final_pool: List[FinalFormulaRecord] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


class AlphaGptWorkflow:
    """Alpha-GPT 工作流协调器

    每轮 5 个 spawn（multi-process via nanobot upstream）。
    支持两种运行模式：

    - run(): 同步执行（mock 友好）
    - run_stream(): 流式输出事件（生产用）
    """

    def __init__(
        self,
        config: AlphaGptConfig,
        data: Any = None,
        data_path: Optional[str] = None,
        llm_client: Optional[Any] = None,
    ) -> None:
        self.config = config
        self.data = data
        self.data_path = data_path
        # llm_client=None → 使用 _mock_llm_response (Stage 1)
        # llm_client=LLMGateway → 使用真实 LLM (Stage 2)
        self.llm_client = llm_client
        self.state = AlphaGptState(
            objective=config.objective,
            iterations_total=config.iterations,
        )
        self._cache_evaluator_results: Dict[str, EvaluationRecord] = {}

    def run(self) -> AlphaGptResult:
        """同步执行工作流"""
        import time

        start = time.time()
        for round_idx in range(1, self.config.iterations + 1):
            logger.info("Round %d/%d starting", round_idx, self.config.iterations)
            self._run_one_round(round_idx)
            if round_idx == self.config.iterations:
                self._run_critic()

        final_pool = self._select_final_pool()
        elapsed = time.time() - start

        summary = self._build_summary(final_pool)
        return AlphaGptResult(
            objective=self.config.objective,
            iterations_completed=self.config.iterations,
            total_formulas=len(self.state.all_formulas),
            final_pool=final_pool,
            summary=summary,
            elapsed_seconds=elapsed,
        )

    def _run_one_round(self, round_idx: int) -> None:
        """一轮完整 5 步骤（除 critic）"""
        self.state.round_idx_hint = round_idx
        ideas = self._step_idea_generator(round_idx)
        self.state.all_ideas.extend(ideas)

        formulas = self._step_formula_translator(round_idx, ideas)
        self.state.all_formulas.extend(formulas)

        evaluations = self._step_evaluator(round_idx, formulas)
        self.state.all_evaluations.extend(evaluations)

        if round_idx < self.config.iterations:
            reflection = self._step_reflector(round_idx, evaluations)
            self.state.all_reflections.append(reflection)

    def _step_idea_generator(self, round_idx: int) -> List[IdeaRecord]:
        """Step 1: spawn idea-generator"""
        prev_reflection = (
            self.state.all_reflections[-1].to_dict()
            if self.state.all_reflections
            else None
        )
        prompt = self._build_idea_prompt(round_idx, prev_reflection)
        raw = self._call_llm("alpha-gpt-idea-generator", prompt)
        parsed = parse_idea_generator_output(raw)
        if not parsed.ok:
            logger.warning("idea-generator parse failed: %s", parsed.error)
            return []
        data = parsed.data or {}
        ideas_data = data.get("ideas", [])[: self.config.pool_size]
        return [IdeaRecord.from_dict(i, round_idx) for i in ideas_data]

    def _step_formula_translator(
        self,
        round_idx: int,
        ideas: List[IdeaRecord],
    ) -> List[FormulaRecord]:
        """Step 2: spawn formula-translator"""
        if not ideas:
            return []
        available_ops = self._get_available_operators()
        data_columns = self._get_data_columns()
        prev_ideas = self._serialize_ideas_for_translator(ideas)
        prompt = self._build_formula_prompt(round_idx, prev_ideas, available_ops, data_columns)
        raw = self._call_llm("alpha-gpt-formula-translator", prompt)
        parsed = parse_formula_translator_output(raw)
        if not parsed.ok:
            logger.warning("formula-translator parse failed: %s", parsed.error)
            return []
        data = parsed.data or {}
        formulas_data = data.get("formulas", [])
        result = []
        for i, fd in enumerate(formulas_data):
            formula_str = fd.get("formula", "")
            err = validate_formula_operators(formula_str)
            if err:
                logger.debug("formula op-validation failed: %s (%s)", formula_str, err)
                continue
            result.append(
                FormulaRecord(
                    formula_id=f"FORMULA-{round_idx}-{i+1}",
                    idea_id=fd.get("idea_id", ""),
                    formula=formula_str,
                    round_discovered=round_idx,
                    complexity=fd.get("complexity", 0),
                    a_share_compatible=fd.get("a_share_compatible", True),
                )
            )
        return result

    def _step_evaluator(
        self,
        round_idx: int,
        formulas: List[FormulaRecord],
    ) -> List[EvaluationRecord]:
        """Step 3: spawn evaluator（同步调用 alpha_evaluate 工具）

        不真走 nanobot spawn（那是框架级）。直接用 alpha_evaluate tool 评估。
        """
        if not formulas:
            return []
        try:
            from QuantNodes.agent.tools.alpha_evaluate import AlphaEvaluateTool
            from QuantNodes.agent.tools.alpha_backtest import AlphaBacktestTool

            tool = AlphaEvaluateTool()
            formulas_str = [f.formula for f in formulas]
            result = _run_async(
                tool.execute(
                    formulas=formulas_str,
                    data=self.data,
                    data_path=self.data_path,
                    forward_returns=list(self.config.forward_returns),
                    date_column=self.config.date_column,
                    code_column=self.config.code_column,
                )
            )
        except Exception as exc:
            logger.exception("alpha_evaluate tool failed")
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
                        status="failed",
                        error_msg=ed.get("error_msg", ""),
                    )
                )
        return out

    def _step_reflector(
        self,
        round_idx: int,
        evaluations: List[EvaluationRecord],
    ) -> ReflectionRecord:
        """Step 4: spawn reflector"""
        evals_dict = [e.to_dict() for e in evaluations]
        prompt = self._build_reflector_prompt(round_idx, evals_dict)
        raw = self._call_llm("alpha-gpt-reflector", prompt)
        parsed = parse_reflector_output(raw)
        if not parsed.ok:
            logger.warning("reflector parse failed: %s", parsed.error)
            return ReflectionRecord(
                round_idx=round_idx,
                verdicts=[],
                suggestions={},
            )
        data = parsed.data or {}
        return ReflectionRecord(
            round_idx=round_idx,
            verdicts=data.get("formula_feedback", []),
            suggestions=data.get("next_round_suggestions", {}),
        )

    def _run_critic(self) -> None:
        """Step 5: spawn critic（仅末轮）"""
        all_evals = [e.to_dict() for e in self.state.all_evaluations]
        all_refl = [r.to_dict() for r in self.state.all_reflections]
        prompt = self._build_critic_prompt(all_evals, all_refl)
        raw = self._call_llm("alpha-gpt-critic", prompt)
        parsed = parse_critic_output(raw)
        if not parsed.ok:
            logger.warning("critic parse failed: %s", parsed.error)
            return
        data = parsed.data or {}
        self.state.critic_output = data

    def _select_final_pool(self) -> List[FinalFormulaRecord]:
        """从 evaluations 中选 top-K（代码方式，不依赖 critic LLM）

        直接从所有成功评估中按 IR 排序选 top-K，然后做互信息去重。
        """
        # 直接从 evaluations 排序（不依赖 critic LLM）
        successful = [e for e in self.state.all_evaluations if e.status == "success"]
        successful.sort(key=lambda e: abs(e.ir), reverse=True)
        top = successful[: self.config.top_k]
        final_pool = [
            FinalFormulaRecord(
                rank=i + 1,
                formula_id=e.formula_id,
                formula=e.formula,
                ic_mean=e.ic_mean,
                ir=e.ir,
                round_discovered=int(e.formula_id.split("-")[1]) if "-" in e.formula_id else 0,
                selection_reason=f"IR={e.ir:.3f} (auto-selected by code)",
                risk_notes=[],
            )
            for i, e in enumerate(top)
        ]

        # 互信息去重
        if self.config.max_mutual_ic_threshold < 1.0 and self.data is not None:
            try:
                from QuantNodes.research.quant_alpha.evaluation.evaluators.polars_evaluator import (
                    deduplicate_mutual_ic,
                )
                from QuantNodes.research.quant_alpha.operator_vocab import OperatorVocab

                vocab = OperatorVocab.default()

                def get_values(record: FinalFormulaRecord) -> Optional[Any]:
                    try:
                        return vocab.evaluate(record.formula, self.data)
                    except Exception:
                        return None

                # 转换为 FactorMetrics 格式用于去重
                from QuantNodes.research.quant_alpha.evaluation.contracts import FactorMetrics
                metrics_list = [
                    FactorMetrics(
                        formula_id=r.formula_id,
                        status="success",
                        ic_mean=r.ic_mean,
                        ir=r.ir,
                        overall_score=r.ir,
                    )
                    for r in final_pool
                ]

                deduped = deduplicate_mutual_ic(
                    metrics_list,
                    get_values,
                    threshold=self.config.max_mutual_ic_threshold,
                )

                # 重建 final_pool
                deduped_ids = {m.formula_id for m in deduped}
                final_pool = [r for r in final_pool if r.formula_id in deduped_ids]

                # 重新编号
                for i, r in enumerate(final_pool):
                    r.rank = i + 1

            except Exception as e:
                logger.warning("互信息去重失败: %s", e)

        return final_pool

    def _build_summary(
        self, final_pool: List[FinalFormulaRecord],
    ) -> Dict[str, Any]:
        successful = [e for e in self.state.all_evaluations if e.status == "success"]
        irs = [e.ir for e in successful]
        cat_dist: Dict[str, int] = {}
        for f in final_pool:
            cat = f.category or "unknown"
            cat_dist[cat] = cat_dist.get(cat, 0) + 1
        return {
            "total_evaluated": len(self.state.all_evaluations),
            "successful": len(successful),
            "failed": len(self.state.all_evaluations) - len(successful),
            "selected": len(final_pool),
            "avg_ir": float(np.mean(irs)) if irs else 0.0,
            "best_ir": float(np.max(irs)) if irs else 0.0,
            "category_distribution": cat_dist,
        }

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------

    def _build_idea_prompt(self, round_idx: int, prev_reflection: Any) -> str:
        return (
            f"Read .agent/agents/alpha-gpt-idea-generator.md. "
            f"Generate {self.config.pool_size} alpha ideas for objective={self.config.objective!r}. "
            f"round={round_idx}, a_share_focus={self.config.a_share_focus}. "
            f"previous_reflection={prev_reflection}. "
            f"Output STRICT JSON only."
        )

    def _build_formula_prompt(
        self,
        round_idx: int,
        ideas_payload: List[Dict[str, Any]],
        available_ops: List[str],
        data_columns: List[str],
    ) -> str:
        return (
            f"Read .agent/agents/alpha-gpt-formula-translator.md. "
            f"Translate these ideas to polars formulas. round={round_idx}. "
            f"ideas={ideas_payload}. available_operators={available_ops}. "
            f"data_columns={data_columns}. a_share_focus={self.config.a_share_focus}. "
            f"CRITICAL: Use ONLY function call format like op(arg1, arg2). "
            f"NO arithmetic operators (+,-,*,/). NO missing parentheses. "
            f"Output STRICT JSON only."
        )

    def _build_reflector_prompt(self, round_idx: int, evaluations: List[Dict[str, Any]]) -> str:
        return (
            f"Read .agent/agents/alpha-gpt-reflector.md. "
            f"Reflect on round {round_idx} evaluations. "
            f"evaluations={evaluations}. "
            f"Output STRICT JSON only."
        )

    def _build_critic_prompt(
        self, all_evaluations: List[Dict[str, Any]], all_reflections: List[Dict[str, Any]]
    ) -> str:
        return (
            f"Read .agent/agents/alpha-gpt-critic.md. "
            f"Select final top-{self.config.top_k} from all rounds. "
            f"min_ir_threshold={self.config.min_ir_threshold}. "
            f"max_mutual_ic_threshold={self.config.max_mutual_ic_threshold}. "
            f"all_evaluations={all_evaluations}. all_reflections={all_reflections}. "
            f"Output STRICT JSON only."
        )

    # ------------------------------------------------------------------
    # LLM 调用（mock 友好）
    # ------------------------------------------------------------------

    def _call_llm(self, agent_id: str, prompt: str) -> str:
        """调用 LLM（mock 时返回预定义 JSON）"""
        if self.llm_client is not None:
            # 兼容旧接口: complete(agent_id, prompt)
            if hasattr(self.llm_client, 'complete'):
                return self.llm_client.complete(agent_id=agent_id, prompt=prompt)
            # 兼容 LLMGateway: __call__(prompt)
            return self.llm_client(prompt)
        # 默认 mock：返回最小 valid JSON（让 workflow 可端到端跑通）
        return _mock_llm_response(agent_id, prompt, self.state, self.config)

    # ------------------------------------------------------------------
    # 数据 metadata
    # ------------------------------------------------------------------

    def _get_available_operators(self) -> List[str]:
        try:
            from QuantNodes.research.quant_alpha.operator_vocab import (
                list_vocab_operators,
            )

            return list(list_vocab_operators())
        except Exception:
            from ..llm.parser import ALLOWED_OPERATORS

            return sorted(ALLOWED_OPERATORS)

    def _get_data_columns(self) -> List[str]:
        if self.data is None:
            return ["close", "open", "high", "low", "vol", "vwap"]
        try:
            return list(self.data.columns)
        except Exception:
            return ["close", "open", "high", "low", "vol"]

    @staticmethod
    def _serialize_ideas_for_translator(ideas: List[IdeaRecord]) -> List[Dict[str, Any]]:
        return [i.to_dict() for i in ideas]


# ==============================================================================
# Mock LLM（默认返回 valid JSON）
# ==============================================================================


def _mock_llm_response(
    agent_id: str,
    prompt: str,
    state: AlphaGptState,
    config: Any = None,
) -> str:
    """Mock LLM 返回（让 workflow 在无 API key 时也能端到端跑通）

    - idea-generator: 返回 pool_size 个简单想法
    - formula-translator: 返回 ideas 数量的简单公式
    - evaluator: mock（evaluator 实际用 tool 计算，跳过）
    - reflector: 返回 keep verdicts
    - critic: 返回空 final_pool（fallback 路径）
    """
    import json

    pool_size = (config.pool_size if config is not None else state.iterations_total)
    round_idx = state.round_idx_hint
    if "idea-generator" in agent_id:
        ideas = []
        categories = ["reversal", "momentum", "volatility", "value", "quality", "liquidity"]
        for i in range(pool_size):
            ideas.append({
                "id": f"IDEA-{round_idx}-{i+1}",
                "name": f"想法-{i+1}",
                "category": categories[i % len(categories)],
                "description": f"Mock idea {i+1}",
                "expected_direction": "long",
                "suggested_lookback": 20,
                "a_share_compatible": True,
                "orthogonal_to": [],
                "complexity_hint": "simple",
            })
        return json.dumps({"round": round_idx, "ideas": ideas}, ensure_ascii=False)

    if "formula-translator" in agent_id:
        ideas_count = pool_size
        formulas = []
        for i in range(ideas_count):
            formulas.append({
                "id": f"FORMULA-{round_idx}-{i+1}",
                "idea_id": f"IDEA-{round_idx}-{i+1}",
                "formula": "sub(close, ts_mean(close, 10))",
                "complexity": 3,
                "a_share_compatible": True,
                "explanation": "Mock formula",
            })
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
# 辅助
# ==============================================================================


def _run_async(coro: Any) -> Any:
    """同步调用 async coroutine"""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(asyncio.run, coro)
                return fut.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


__all__ = [
    "AlphaGptConfig",
    "AlphaGptResult",
    "AlphaGptWorkflow",
]
