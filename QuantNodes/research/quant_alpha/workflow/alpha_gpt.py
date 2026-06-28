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

    # 各阶段温度参数（覆盖 temperature）
    temperature_idea_gen: float = 0.8   # 鼓励创新
    temperature_formula: float = 0.4    # 需要精确
    temperature_reflector: float = 0.6  # 平衡
    temperature_critic: float = 0.3     # 需要稳定

    spawn_timeout_seconds: float = 30.0

    a_share_focus: bool = True
    enable_backtest: bool = False
    top_k_backtest: int = 10

    custom_few_shot: Optional[List[Dict[str, Any]]] = None

    # 自定义反馈（用于多轮迭代，注入到 IdeaGenerator）
    custom_feedback: Optional[str] = None

    # Γ 约束（用于逻辑驱动因子生成）
    gamma: Optional[Any] = None  # CompiledConstraint from logic_mining.compiler


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
        output_dir: Optional[str] = None,
    ) -> None:
        self.config = config
        self.data = data
        self.data_path = data_path
        # llm_client=None → 使用 _mock_llm_response (Stage 1)
        # llm_client=LLMGateway → 使用真实 LLM (Stage 2)
        self.llm_client = llm_client
        # 完整保存 LLM 原始输出（用于调试截断/解析失败）
        self.output_dir = output_dir
        self._llm_raw_dir: Optional[Any] = None
        if output_dir:
            from pathlib import Path
            self._llm_raw_dir = Path(output_dir) / "llm_raw"
            self._llm_raw_dir.mkdir(parents=True, exist_ok=True)
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
        raw, thinking = self._call_llm("alpha-gpt-idea-generator", prompt)
        parsed = parse_idea_generator_output(raw)
        if not parsed.ok:
            logger.warning("idea-generator parse failed: %s", parsed.error)
            return []
        data = parsed.data or {}
        ideas_data = data.get("ideas", [])[: self.config.pool_size]

        # Tier 1+2: 解析 thinking → ThinkingRecord
        thinking_record = None
        if thinking:
            from QuantNodes.research.quant_alpha.llm.parser import parse_thinking_block
            op_vocab = set(self._get_available_operators())
            thinking_record = parse_thinking_block(thinking, op_vocab=op_vocab)

        ideas = []
        for i in ideas_data:
            idea = IdeaRecord.from_dict(i, round_idx)
            if thinking:
                idea.thinking = thinking
                if thinking_record:
                    idea.hypothesis = thinking_record.hypothesis or None
                    idea.mechanism = thinking_record.mechanism or None
                    idea.mentioned_ops = list(thinking_record.mentioned_ops)
            ideas.append(idea)
        return ideas

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
        raw, thinking = self._call_llm("alpha-gpt-formula-translator", prompt)
        parsed = parse_formula_translator_output(raw)
        if not parsed.ok:
            logger.warning("formula-translator parse failed: %s", parsed.error)
            return []
        data = parsed.data or {}
        formulas_data = data.get("formulas", [])

        # P1 (fix/explanation-truncation): post-process explanation 字段
        # 强制截断到 200 chars，防止 LLM 在 explanation 中塞结构化字段导致 JSON 膨胀
        for fd in formulas_data:
            if isinstance(fd, dict) and "explanation" in fd:
                expl = fd["explanation"]
                if isinstance(expl, str) and len(expl) > 200:
                    fd["explanation"] = expl[:197] + "..."

        # Tier 1+2: 解析 thinking
        thinking_record = None
        if thinking:
            from QuantNodes.research.quant_alpha.llm.parser import parse_thinking_block
            thinking_record = parse_thinking_block(thinking, op_vocab=set(available_ops))

        # 共享 thinking 给所有 formulas（同一轮 translator 调用）
        shared_thinking = thinking or None
        shared_hypothesis = (thinking_record.hypothesis if thinking_record else "") or None
        shared_mentioned_ops = (
            list(thinking_record.mentioned_ops) if thinking_record else []
        )

        result = []
        for i, fd in enumerate(formulas_data):
            formula_str = fd.get("formula", "")
            err = validate_formula_operators(formula_str)
            if err:
                logger.info("formula op-validation warning (will try anyway): %s (%s)", formula_str, err)

            # Γ 约束校验
            if self.config.gamma is not None:
                passed, reason = self.config.gamma.validate(formula_str)
                if not passed:
                    logger.info("Γ 校验失败，丢弃公式: %s - %s", formula_str, reason)
                    continue

            formula_record = FormulaRecord(
                formula_id=f"FORMULA-{round_idx}-{i+1}",
                idea_id=fd.get("idea_id", ""),
                formula=formula_str,
                round_discovered=round_idx,
                complexity=fd.get("complexity", 0),
                a_share_compatible=fd.get("a_share_compatible", True),
            )
            if shared_thinking:
                formula_record.thinking = shared_thinking
                formula_record.hypothesis = shared_hypothesis
                formula_record.mentioned_ops = shared_mentioned_ops
            result.append(formula_record)
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
        raw, thinking = self._call_llm("alpha-gpt-reflector", prompt)
        parsed = parse_reflector_output(raw)
        if not parsed.ok:
            logger.warning("reflector parse failed: %s", parsed.error)
            return ReflectionRecord(
                round_idx=round_idx,
                verdicts=[],
                suggestions={},
            )
        data = parsed.data or {}
        record = ReflectionRecord(
            round_idx=round_idx,
            verdicts=data.get("formula_feedback", []),
            suggestions=data.get("next_round_suggestions", {}),
        )
        # Tier 1+2: 提取 insights
        if thinking:
            record.thinking = thinking
            analysis = data.get("analysis", {})
            key_insights = analysis.get("key_insights", [])
            if isinstance(key_insights, list):
                record.key_insights = [str(s) for s in key_insights]
        return record

    def _run_critic(self) -> None:
        """Step 5: spawn critic（仅末轮）"""
        all_evals = [e.to_dict() for e in self.state.all_evaluations]
        all_refl = [r.to_dict() for r in self.state.all_reflections]
        prompt = self._build_critic_prompt(all_evals, all_refl)
        raw, _thinking = self._call_llm("alpha-gpt-critic", prompt)
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
        logger.info("[_select_final_pool] %d successful, %d top selected", len(successful), len(top))
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

        logger.info("[_select_final_pool] before dedup: %d formulas", len(final_pool))

        # 互信息去重
        if self.config.max_mutual_ic_threshold < 1.0 and self.data is not None:
            try:
                from QuantNodes.research.quant_alpha.evaluation.evaluators.polars_evaluator import (
                    deduplicate_mutual_ic,
                )
                from QuantNodes.research.quant_alpha.operator_vocab import OperatorVocab

                vocab = OperatorVocab.default()

                def get_values(record: FactorMetrics) -> Optional[Any]:
                    try:
                        # Find the formula from final_pool
                        for r in final_pool:
                            if r.formula_id == record.formula_id:
                                return vocab.evaluate(r.formula, self.data)
                        return None
                    except Exception as e:
                        logger.debug("[_select_final_pool] eval failed for %s: %s", record.formula_id, e)
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

                logger.info("[_select_final_pool] dedup: %d -> %d (threshold=%.2f)", len(metrics_list), len(deduped), self.config.max_mutual_ic_threshold)

                # 重建 final_pool
                deduped_ids = {m.formula_id for m in deduped}
                final_pool = [r for r in final_pool if r.formula_id in deduped_ids]

                # 重新编号
                for i, r in enumerate(final_pool):
                    r.rank = i + 1

            except Exception as e:
                logger.warning("互信息去重失败: %s", e, exc_info=True)

        logger.info("[_select_final_pool] after dedup: %d formulas", len(final_pool))
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
        # P3 (fix/explanation-truncation): 使用 "rationale" 而非 "description"
        # 避免 LLM 混淆 description 与 formula-translator 的 explanation 字段
        schema = (
            '{"round": 1, "ideas": ['
            '{"id": "IDEA-1-1", "name": "20日反转", "category": "reversal", '
            '"rationale": "经济直觉1-2句（与formula explanation区分）", "expected_direction": "long", '
            '"suggested_lookback": 20, "a_share_compatible": true, '
            '"orthogonal_to": ["IDEA-1-2"], "complexity_hint": "simple"}'
            ']}'
        )
        prompt = (
            f"You are the Alpha-GPT IdeaGenerator. "
            f"Generate {self.config.pool_size} alpha ideas for objective={self.config.objective!r}. "
            f"round={round_idx}, a_share_focus={self.config.a_share_focus}. "
            f"previous_reflection={prev_reflection}. "
            f"6 categories: momentum/reversal/value/quality/volatility/liquidity. "
            f"Each idea must have id (IDEA-{{round}}-{{idx}}), name, category, "
            f"rationale (经济直觉 <150 chars, brief!), expected_direction (long/short/both), "
            f"suggested_lookback, a_share_compatible, orthogonal_to, "
            f"complexity_hint (simple/medium/complex). "
            f"Output STRICT JSON (no markdown, no code blocks) matching this schema: {schema}"
        )

        # Tier 2: 结构化推理指令（在 <think> 块中输出推理）
        available_ops_summary = ", ".join(self._get_available_operators()[:30])
        thinking_template = (
            "\n\n## 推理要求 (Tier 2: Structured Reasoning)\n"
            "在生成 JSON 之前，先在 <think> 块中按以下结构输出你的推理：\n"
            "```\n"
            "<think>\n"
            "HYPOTHESIS: <一句话经济假设，如 'A 股散户过度反应导致 20 日反转'>\n"
            "MECHANISM: <为什么在 A 股有效，提及 T+1/散户主导/涨跌停等特殊约束>\n"
            f"OPERATOR_RATIONALE: <为什么选这些算子，从 {available_ops_summary} 等中选>\n"
            "PARAMETER_RATIONALE: <为什么这个窗口参数，如 20 日对应学术文献经典窗口>\n"
            "RISK: <什么情况下因子会失效，如 流动性危机 / 政策切换 / ST 股扰动>\n"
            "SUGGESTED_OPS: <逗号分隔的算子名，如 rank,ts_mean,div>\n"
            "</think>\n"
            "```\n"
            "然后输出 JSON（不要 markdown 包装）。thinking 内容会被保留用于下游 MCTS 引导。\n"
        )
        prompt += thinking_template

        # 注入自定义反馈（用于多轮迭代）
        if self.config.custom_feedback:
            prompt += f"\n\n## 历史反馈（来自上一轮 MCTS 搜索）\n{self.config.custom_feedback}\n"

        return prompt

    def _build_formula_prompt(
        self,
        round_idx: int,
        ideas_payload: List[Dict[str, Any]],
        available_ops: List[str],
        data_columns: List[str],
    ) -> str:
        schema = (
            '{"round": 1, "formulas": ['
            '{"id": "FORMULA-1-1", "idea_id": "IDEA-1-1", '
            '"formula": "rank(-ts_mean(returns, 20))", '
            '"complexity": 3, "a_share_compatible": true, '
            '"explanation": "20日反转因子"}'
            ']}'
        )
        prompt = (
            f"You are the Alpha-GPT FormulaTranslator. "
            f"Translate these ideas to polars formulas. round={round_idx}. "
            f"ideas={ideas_payload}. "
            f"a_share_focus={self.config.a_share_focus}. "
            f"Each formula must have id (FORMULA-{{round}}-{{idx}}), idea_id, "
            f"formula (function call format like op(arg1, arg2)), complexity, "
            f"a_share_compatible, explanation. "
            f"CRITICAL: Use ONLY function call format. NO arithmetic operators (+,-,*,/). "
            f"NO missing parentheses. Output STRICT JSON (no markdown) matching: {schema}. "
        )

        # Tier 2: 公式翻译也用 thinking 块说明算子选择
        prompt += (
            "\n\n## 推理要求 (Tier 2)\n"
            "在 JSON 前用 <think> 块说明：\n"
            "```\n"
            "<think>\n"
            "HYPOTHESIS: <该 idea 的核心经济假设，引用 idea.description>\n"
            "MECHANISM: <公式如何捕捉该机制>\n"
            "OPERATOR_RATIONALE: <为什么用这些算子>\n"
            "PARAMETER_RATIONALE: <为什么用这些窗口/参数>\n"
            "RISK: <公式的潜在失效模式>\n"
            "SUGGESTED_OPS: <公式中实际使用的算子列表>\n"
            "</think>\n"
            "```\n"
            "CRITICAL: explanation 字段必须 <80 字符！\n"
            "禁止在 explanation 中包含 HYPOTHESIS/MECHANISM 等结构化字段。\n"
            "explanation 只描述公式的'做什么'，不解释'为什么'。\n"
        )

        # 注入 Γ 约束（更清晰的格式）
        if self.config.gamma is not None:
            gamma = self.config.gamma
            
            # 构建约束说明
            constraints = []
            
            # 算子白名单
            if gamma.operator_whitelist:
                ops = sorted(gamma.operator_whitelist)
                constraints.append(f"ALLOWED OPERATORS: {', '.join(ops)}")
                constraints.append(f"You MUST use ONLY these operators. Do NOT use any other operators.")
            
            # 变量白名单
            if gamma.variable_whitelist:
                vars_ = sorted(gamma.variable_whitelist)
                constraints.append(f"ALLOWED VARIABLES: {', '.join(vars_)}")
                constraints.append(f"You MUST use ONLY these variables. Do NOT use any other variables.")
            
            # 参数范围
            if gamma.parameter_ranges:
                constraints.append(f"PARAMETER RANGES:")
                for op, (lo, hi) in sorted(gamma.parameter_ranges.items()):
                    constraints.append(f"  - {op}: window must be between {lo} and {hi}")
            
            # 符号约束
            if gamma.sign_constraint is not None:
                direction = "POSITIVE (+1)" if gamma.sign_constraint > 0 else "NEGATIVE (-1)"
                constraints.append(f"SIGN CONSTRAINT: Overall factor direction must be {direction}")
            
            # 添加到 prompt
            if constraints:
                prompt += "\n\n=== Γ CONSTRAINTS (MUST FOLLOW) ===\n"
                prompt += "\n".join(constraints)
                prompt += "\n===================================\n"
                
                # 添加示例
                if gamma.operator_whitelist and "rank" in gamma.operator_whitelist and "ts_corr" in gamma.operator_whitelist:
                    prompt += "\nEXAMPLE FORMULA (follows constraints):\n"
                    prompt += "  sign(-ts_corr(rank(open), rank(volume), 10))\n"
        
        # 添加可用算子和变量（来自 OperatorVocab）
        prompt += f"\navailable_operators={available_ops}. "
        prompt += f"data_columns={data_columns}. "
        
        prompt += f"Output STRICT JSON only."
        return prompt

    def _build_reflector_prompt(self, round_idx: int, evaluations: List[Dict[str, Any]]) -> str:
        schema = (
            '{"round": 1, "analysis": {"best_categories": ["reversal"], '
            '"worst_categories": ["liquidity"], "key_insights": ["..."]}, '
            '"formula_feedback": [{"formula_id": "FORMULA-1-1", "formula": "...", '
            '"verdict": "keep", "reason": "...", "improvements": ["..."]}]}'
        )
        return (
            f"You are the Alpha-GPT Reflector. "
            f"Reflect on round {round_idx} evaluations. "
            f"evaluations={evaluations}. "
            f"Output STRICT JSON (no markdown) with: round, analysis (best_categories, "
            f"worst_categories, key_insights), formula_feedback (formula_id, formula, "
            f"verdict (keep/mutate/drop), reason, improvements). "
            f"Schema: {schema}\n\n"
            f"## 推理要求 (Tier 2)\n"
            f"在 JSON 前用 <think> 块说明：\n"
            f"<think>\n"
            f"KEY_INSIGHTS: <3-5 条本轮核心洞察，列出成功/失败模式>\n"
            f"NEXT_ROUND_FOCUS: <下一轮应聚焦的方向，如某类算子/参数/逻辑>\n"
            f"RISK_PATTERNS: <本轮发现的失效模式，下轮需规避>\n"
            f"</think>"
        )

    def _build_critic_prompt(
        self, all_evaluations: List[Dict[str, Any]], all_reflections: List[Dict[str, Any]]
    ) -> str:
        schema = (
            '{"final_pool": [{"rank": 1, "formula_id": "FORMULA-1-1", '
            '"formula": "...", "metrics": {"ic_mean": 0.045, "ir": 2.05, '
            '"sharpe": 1.65, "max_drawdown": -0.123}, '
            '"selection_reason": "...", "risk_notes": ["..."], '
            '"category": "reversal", "round_discovered": 1}], '
            '"summary": {"total_evaluated": 50}}'
        )
        return (
            f"You are the Alpha-GPT Critic. "
            f"Select final top-{self.config.top_k} from all rounds. "
            f"min_ir_threshold={self.config.min_ir_threshold}. "
            f"max_mutual_ic_threshold={self.config.max_mutual_ic_threshold}. "
            f"all_evaluations={all_evaluations}. all_reflections={all_reflections}. "
            f"Output STRICT JSON (no markdown) with: final_pool (rank, formula_id, "
            f"formula, metrics (ic_mean, ir, sharpe, max_drawdown), selection_reason, "
            f"risk_notes, category, round_discovered), summary. "
            f"Schema: {schema}\n\n"
            f"## 推理要求 (Tier 2)\n"
            f"在 JSON 前用 <think> 块说明选 top-{self.config.top_k} 的标准。\n"
            f"<think>\n"
            f"SELECTION_CRITERIA: <你如何权衡 IR/IC/sharpe/drawdown>\n"
            f"DIVERSITY: <如何保证 final_pool 的类别/算子多样性>\n"
            f"RISK_FILTERS: <剔除了什么类型的因子（过拟合/高换手/低 IC decay 等）>\n"
            f"</think>"
        )

    # ------------------------------------------------------------------
    # LLM 调用（mock 友好）
    # ------------------------------------------------------------------

    def _call_llm(self, agent_id: str, prompt: str) -> Tuple[str, str]:
        """调用 LLM（mock 时返回预定义 JSON），返回 (content, thinking) tuple。

        若 output_dir 已设置，会把每次 LLM 调用的完整 prompt/response 持久化到
        {output_dir}/llm_raw/{agent_id}_{round_idx}_{ts}.json，方便后续分析
        截断、解析失败等问题。

        如果 LLM client 是 LLMGateway 且支持 ``complete_with_thinking``，
        则同时返回 thinking 块（MiniMax M3 的 <think>...</think>）。
        """
        import inspect
        import json
        import time as _time
        from pathlib import Path
        from QuantNodes.ai.llm.gateway import LLMGateway

        temperature = self._get_temperature_for_agent(agent_id)
        ts = int(_time.time() * 1000)

        thinking = ""
        raw = ""

        # 实际调用 LLM（或 mock）
        if self.llm_client is not None:
            # Tier 1: LLMGateway 路径 → 同时获取 thinking
            if isinstance(self.llm_client, LLMGateway) and hasattr(
                self.llm_client, 'complete_with_thinking'
            ):
                persist_dir = str(self._llm_raw_dir) if self._llm_raw_dir else None
                raw, thinking = self.llm_client.complete_with_thinking(
                    agent_id=agent_id, prompt=prompt,
                    temperature=temperature,
                    persist_thinking_dir=persist_dir,
                )
            elif hasattr(self.llm_client, 'complete'):
                # 兼容不同 mock 接口（部分 mock 不接受 temperature 关键字）
                try:
                    sig = inspect.signature(self.llm_client.complete)
                    if "temperature" in sig.parameters:
                        raw = self.llm_client.complete(
                            agent_id=agent_id, prompt=prompt, temperature=temperature
                        )
                    else:
                        raw = self.llm_client.complete(agent_id=agent_id, prompt=prompt)
                except (TypeError, ValueError):
                    raw = self.llm_client.complete(agent_id=agent_id, prompt=prompt)
            else:
                raw = self.llm_client(prompt)
        else:
            raw = _mock_llm_response(agent_id, prompt, self.state, self.config)

        # 完整持久化（无截断）
        if self._llm_raw_dir is not None:
            try:
                round_idx = self.state.round_idx_hint
                safe_agent = agent_id.replace("/", "_").replace(" ", "_")
                out_file = self._llm_raw_dir / f"r{round_idx}_{safe_agent}_{ts}.json"
                out_file.write_text(
                    json.dumps(
                        {
                            "agent_id": agent_id,
                            "round_idx": round_idx,
                            "temperature": temperature,
                            "prompt": prompt,
                            "response": raw,
                            "response_length": len(raw),
                            "thinking": thinking,
                            "thinking_length": len(thinking),
                            "ts": ts,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.warning("保存 LLM raw 失败: %s", exc)
        return raw, thinking

    def _get_temperature_for_agent(self, agent_id: str) -> float:
        """根据 agent_id 返回对应的温度参数"""
        if "idea-generator" in agent_id:
            return self.config.temperature_idea_gen
        elif "formula-translator" in agent_id:
            return self.config.temperature_formula
        elif "reflector" in agent_id:
            return self.config.temperature_reflector
        elif "critic" in agent_id:
            return self.config.temperature_critic
        else:
            return self.config.temperature

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
