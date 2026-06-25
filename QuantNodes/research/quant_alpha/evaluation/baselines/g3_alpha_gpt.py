# coding=utf-8
"""
g3_alpha_gpt.py - G3 baseline：包 AlphaGptWorkflow（M5）

G3 = "Alpha-GPT"，代表完整 5 智能体编排：idea-generator → formula-translator →
evaluator → reflector → critic，5 轮迭代。

Stage 1：用 mock LLM（_mock_llm_response）跑通 workflow
Stage 2：注入 NanobotLLMWrapper(MiniMaxClient) 替换 mock

复用：
- QuantNodes.research.quant_alpha.workflow.AlphaGptWorkflow：完整 5 智能体编排
- QuantNodes.research.quant_alpha.workflow.AlphaGptConfig：config dataclass
- contracts.Baseline：generate_factors() 接口
"""

from __future__ import annotations

import logging
from typing import List, Optional

from ..contracts import Baseline, FactorSpec

logger = logging.getLogger(__name__)

__all__ = ["G3AlphaGpt"]


class G3AlphaGpt(Baseline):
    """G3 Alpha-GPT baseline

    包装 AlphaGptWorkflow（M5），跑完整 5 智能体编排。
    Stage 1：用 mock LLM 跑通；Stage 2：注入真实 LLM client。
    """

    def __init__(
        self,
        n: int = 30,
        objective: str = "maximize IC and IR for 1-day forward return",
        iterations: int = 3,
        pool_size: int = 10,
        seed: int = 11,
    ) -> None:
        self.n = n
        self.objective = objective
        self.iterations = iterations
        self.pool_size = pool_size
        self.seed = seed
        self._last_workflow_result = None  # 暴露给 runner 调试

    @property
    def group_name(self) -> str:
        return "G3_AlphaGpt"

    def generate_factors(self, n: Optional[int] = None) -> List[FactorSpec]:
        """运行 AlphaGptWorkflow 并提取 final_pool 公式

        Args:
            n: 截取前 n 个因子（默认 self.n）

        Returns:
            FactorSpec 列表（公式 + 元信息）
        """
        n = n or self.n

        from QuantNodes.research.quant_alpha.workflow.alpha_gpt import (
            AlphaGptConfig,
            AlphaGptWorkflow,
        )

        config = AlphaGptConfig(
            objective=self.objective,
            iterations=self.iterations,
            pool_size=self.pool_size,
            top_k=n,
        )

        logger.info(
            "[G3] starting AlphaGptWorkflow (iterations=%d, pool_size=%d)",
            self.iterations,
            self.pool_size,
        )

        try:
            workflow = AlphaGptWorkflow(config=config, llm_client=None)
            result = workflow.run()
        except Exception as e:
            logger.error("[G3] AlphaGptWorkflow failed: %s, using fallback", e)
            # workflow 失败时, 用 mock fallback（与 workflow 返回空时的逻辑一致）
            result = None

        if result is not None:
            self._last_workflow_result = result

        factors: List[FactorSpec] = []
        if result is not None:
            for i, formula_rec in enumerate(result.final_pool[:n]):
                factors.append(
                    FactorSpec(
                        formula_id=f"G3_{i:03d}",
                        formula=formula_rec.formula,
                        source="g3_alpha_gpt",
                        category=formula_rec.category or "unknown",
                        complexity=formula_rec.formula.count("("),
                        meta={
                            "rank": formula_rec.rank,
                            "selection_reason": formula_rec.selection_reason,
                            "round_discovered": formula_rec.round_discovered,
                        },
                    )
                )

        # Stage 1 mock 兼容：若 workflow 失败/返回空/final_pool 不足，
        # 用 G1 风格的简单 valid 公式兜底（保证 baseline 数量稳定）
        if len(factors) < n:
            logger.warning(
                "[G3] workflow returned %d factors (期望 %d), mock 兜底补充",
                len(factors),
                n,
            )
            from .g1_handcrafted import _gen_formula
            import random

            rng = random.Random(self.seed + 100)
            while len(factors) < n:
                formula = _gen_formula(rng)
                if any(f.formula == formula for f in factors):
                    continue
                factors.append(
                    FactorSpec(
                        formula_id=f"G3_{len(factors):03d}",
                        formula=formula,
                        source="g3_alpha_gpt",
                        category="g3_fallback_mock",
                        complexity=formula.count("("),
                        meta={"fallback": True, "seed": self.seed},
                    )
                )

        if result is not None:
            logger.info(
                "[G3] AlphaGptWorkflow returned %d factors (total=%d, elapsed=%.2fs)",
                len(factors),
                result.total_formulas,
                result.elapsed_seconds,
            )

        return factors