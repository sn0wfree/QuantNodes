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
    Stage 2：注入真实 LLM client（LLMGateway → MiniMax）
    """

    def __init__(self, n: int = 50, seed: int = 7, llm_client=None) -> None:
        self.n = n
        self.seed = seed
        self._llm_client = llm_client

    @property
    def group_name(self) -> str:
        return "G2_LlmOnly"

    def generate_factors(self, n: Optional[int] = None) -> List[FactorSpec]:
        """生成 n 个因子

        Stage 2 (有 llm_client): 用真实 LLM 生成公式
        Stage 1 (无 llm_client): 用 mock 公式生成器
        """
        n = n or self.n

        # Stage 2: 真实 LLM 生成
        if self._llm_client is not None:
            return self._generate_with_llm(n)

        # Stage 1: mock 公式生成
        return self._generate_mock(n)

    def _generate_with_llm(self, n: int) -> List[FactorSpec]:
        """Stage 2: 用真实 LLM + operator_lookup 工具生成公式。"""
        import asyncio

        prompt = (
            f"你是一个量化研究员，负责生成 alpha 因子公式。\n\n"
            f"任务：生成 {n} 个唯一的 alpha 因子公式。\n\n"
            f"步骤：\n"
            f"1. 先调用 operator_lookup(action='list_operators') 获取可用算子列表\n"
            f"2. 调用 operator_lookup(action='get_operator_info', name='xxx') 获取算子详情\n"
            f"3. 生成公式，确保使用可用算子\n"
            f"4. 调用 operator_lookup(action='validate_formula', formula='xxx') 验证每个公式\n\n"
            f"基础特征：open, high, low, close, vol, amount\n"
            f"公式语法：Python 函数调用，如 rank(ts_mean(close, 20))\n\n"
            f"最终输出：仅返回 JSON 数组，如 [\"rank(ts_mean(close, 20))\", \"ts_std(close, 10)\"]\n"
            f"不要解释，不要 markdown，只要 JSON 数组。"
        )

        try:
            # 使用 nanobot agent (有工具访问权限)
            from QuantNodes.agent.nanobot_bridge import Agent
            agent = Agent(workspace=".agent")
            response = asyncio.run(agent.run(prompt, session_id="g2-formula-gen"))
            formulas = self._parse_llm_formulas(response, n)
        except Exception as e:
            logger.warning("[G2] LLM generation failed: %s, falling back to mock", e)
            return self._generate_mock(n)

        # 公式校验 + 重试
        valid_formulas = []
        for formula in formulas:
            if self._validate_formula(formula):
                valid_formulas.append(formula)
            if len(valid_formulas) >= n:
                break

        # 如果 valid 不足，用 mock 补充
        if len(valid_formulas) < n:
            logger.warning(
                "[G2] Only %d/%d valid formulas, supplementing with mock",
                len(valid_formulas), n,
            )
            mock_factors = self._generate_mock(n - len(valid_formulas))
            for f in mock_factors:
                f.formula_id = f"G2_{len(valid_formulas):03d}"
                valid_formulas.append(f.formula)

        factors: List[FactorSpec] = []
        for i, formula in enumerate(valid_formulas[:n]):
            if isinstance(formula, str):
                factors.append(
                    FactorSpec(
                        formula_id=f"G2_{i:03d}",
                        formula=formula,
                        source="g2_llm_only",
                        category=self._infer_category(formula),
                        complexity=formula.count("("),
                        meta={"llm_generated": True},
                    )
                )
            else:
                # FactorSpec from mock supplement
                formula.formula_id = f"G2_{i:03d}"
                factors.append(formula)

        logger.info("[G2] LLM generated %d factors (%d valid)", len(factors), len(valid_formulas))
        return factors

    @staticmethod
    def _validate_formula(formula: str) -> bool:
        """校验公式是否有效。"""
        from QuantNodes.research.quant_alpha.operator_vocab import OperatorVocab
        import polars as pl

        test_data = pl.DataFrame({
            "date": ["2020-01-01", "2020-01-02", "2020-01-03"] * 3,
            "code": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
            "open": [10.0] * 9,
            "high": [10.5] * 9,
            "low": [9.5] * 9,
            "close": [10.0, 10.1, 10.2, 11.0, 11.1, 11.2, 12.0, 12.1, 12.2],
            "vol": [1000.0] * 9,
            "amount": [10000.0] * 9,
        })

        try:
            vocab = OperatorVocab.default()
            result = vocab.evaluate(
                formula=formula,
                data=test_data,
                date_column="date",
                code_column="code",
            )
            return result is not None and len(result) == len(test_data)
        except Exception:
            return False

    @staticmethod
    def _parse_llm_formulas(response: str, expected: int) -> List[str]:
        """从 LLM 响应中解析公式列表。"""
        import json
        import re

        # 尝试直接 JSON 解析
        try:
            result = json.loads(response)
            if isinstance(result, list):
                return [str(f) for f in result]
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown 代码块提取
        match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if isinstance(result, list):
                    return [str(f) for f in result]
            except json.JSONDecodeError:
                pass

        # 尝试提取所有看起来像公式的内容
        formulas = re.findall(r'"([^"]+\([^"]+\)[^"]*)"', response)
        if formulas:
            return formulas

        logger.warning("[G2] Failed to parse LLM response, using fallback")
        return []

    def _generate_mock(self, n: int) -> List[FactorSpec]:
        """Stage 1: mock 公式生成。"""
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