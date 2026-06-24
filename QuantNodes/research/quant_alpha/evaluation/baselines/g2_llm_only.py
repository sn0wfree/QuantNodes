# coding=utf-8
"""
g2_llm_only.py - G2 baseline：mock LLM 直接生成 50 个公式字符串

G2 = "LLM-Only"，代表「直接问 LLM 生成 alpha 公式」的最简 baseline。
无 5 智能体编排、无反思循环、无 evaluator 反馈 — 仅 LLM 一次输出。

Stage 1 用 mock 模拟 LLM 输出（valid + invalid 混合）；
Stage 2 用真实 LLM（MiniMax）替换。

模拟策略（mock 阶段）：
- 60% 公式：valid（LLM 已知基础语法）
- 25% 公式：复杂 valid（LLM 写出更复杂的算子组合）
- 15% 公式：invalid（模拟 LLM 错误 / 不支持的算子）

复用：
- contracts.Baseline：generate_factors() 接口
- g1_handcrafted._gen_formula：共享公式生成逻辑（DRY）
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional

from ..contracts import Baseline, FactorSpec
from .g1_handcrafted import _gen_formula

logger = logging.getLogger(__name__)

__all__ = ["G2LlmOnly"]


# 15% 模拟 LLM 错误：使用 parser 不支持的算子
INVALID_LLM_TOKENS = [
    "rank(close)",  # 跨截面 rank，tool 不支持
    "IndNeutralize(close, industry)",  # tool 不支持
    "ts_zscore(close, 20)",  # rolling 映射缺失
    "log(vol)",  # 需要先除以均值
    "close - ts_mean(close, 5)",  # 中缀语法
    "correlation(close, vol, 20)",  # tool 不支持
    "quantile(close, 0.5)",  # tool 不支持
]


class G2LlmOnly(Baseline):
    """G2 LLM-Only baseline

    模拟「直接问 LLM 生成 alpha 公式」的最简 baseline：
    - 不调用 AlphaGptWorkflow
    - 不做反思 / critic / evaluator 反馈
    - 一次性输出 N 个公式（含 valid + invalid 混合）

    Stage 1：使用 mock 公式生成器（valid + invalid 混合）
    Stage 2：注入真实 LLM client（NanobotLLMWrapper / MiniMaxClient）
    """

    def __init__(self, n: int = 50, seed: int = 7) -> None:
        self.n = n
        self.seed = seed

    @property
    def group_name(self) -> str:
        return "G2_LlmOnly"

    def generate_factors(self, n: Optional[int] = None) -> List[FactorSpec]:
        """生成 n 个因子（模拟 LLM 一次输出）"""
        n = n or self.n
        rng = random.Random(self.seed)

        factors: List[FactorSpec] = []
        seen: set = set()

        n_invalid = max(1, int(n * 0.15))
        n_complex = max(1, int(n * 0.25))
        n_simple = n - n_invalid - n_complex

        attempts = 0
        while len(factors) < n and attempts < n * 10:
            attempts += 1
            if len(factors) < n_simple:
                formula = _gen_formula(rng, max_depth=1)
            elif len(factors) < n_simple + n_complex:
                formula = _gen_formula(rng, max_depth=2)
            else:
                formula = rng.choice(INVALID_LLM_TOKENS)

            if formula in seen:
                continue
            seen.add(formula)

            factors.append(
                FactorSpec(
                    formula_id=f"G2_{len(factors):03d}",
                    formula=formula,
                    source="g2_llm_only",
                    category=self._infer_category(formula),
                    complexity=formula.count("("),
                    meta={"seed": self.seed, "valid": "rank" not in formula},
                )
            )

        logger.info(
            "[G2] generated %d factors (n_simple=%d, n_complex=%d, n_invalid=%d)",
            len(factors),
            min(n_simple, len(factors)),
            max(0, min(n_complex, len(factors) - n_simple)),
            max(0, len(factors) - n_simple - n_complex),
        )
        return factors

    @staticmethod
    def _infer_category(formula: str) -> str:
        if "ts_mean" in formula and "delta" not in formula:
            return "momentum"
        if "delta" in formula:
            return "momentum"
        if "ts_std" in formula:
            return "volatility"
        if "vol" in formula:
            return "volume"
        if "abs" in formula:
            return "reversal"
        return "value"