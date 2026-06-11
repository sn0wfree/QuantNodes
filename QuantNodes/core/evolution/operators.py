"""Evolution Operators — LLM-based + mock implementation.

3 operators:
    - Hypothesizer: 从 research direction 生成新因子候选 (round 0)
    - Mutator: 从 parent 因子派生 mutation 子代
    - Crosser: 从两个 parent 因子组合 crossover 子代
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class FactorCandidate:
    """因子候选 — EvolutionLoop 与 PipelineRunner 之间传递的最小单位。"""
    factor_id: str
    name: str
    expression: str
    hypothesis: str = ""
    description: str = ""


_HYPOTHESIZE_PROMPT = """你是一个量化研究员, 负责基于研究假设生成 alpha 因子。
研究假设: {hypothesis}
现有描述: {description}

请生成一个可执行的因子表达式 (Python 语法, 引用基础特征: open/high/low/close/volume/amount/vwap/turnover/mv_float)。
返回 JSON: {{"name": "因子名", "expression": "代码表达式", "description": "因子描述"}}"""


_MUTATE_PROMPT = """你是一个量化研究员, 负责对父因子做变异, 探索新变体。
父因子: {parent_expression}
父假设: {parent_hypothesis}
父描述: {parent_description}

请生成一个变异版本 (在保持核心逻辑前提下, 调整参数 / 调换算子 / 增加过滤)。
返回 JSON: {{"name": "新因子名", "expression": "新表达式", "description": "新描述"}}"""


_CROSSOVER_PROMPT = """你是一个量化研究员, 负责组合两个父因子产生新组合。
父因子 1: {p1_expression} ({p1_description})
父因子 2: {p2_expression} ({p2_description})

请生成一个组合 (可加可减可相乘可平均), 保持经济意义。
返回 JSON: {{"name": "组合因子名", "expression": "组合表达式", "description": "组合描述"}}"""


class BaseOperator:
    """Operator 基类, 统一 LLM 调用协议。"""

    def __init__(
        self,
        model: str = "mock",
        max_correction_attempts: int = 3,
        seed: int = 42,
        llm_callable: Optional[Callable] = None,
    ):
        self.model = model
        self.max_correction_attempts = max_correction_attempts
        self.seed = seed
        self._llm_callable = llm_callable

    def _call(self, prompt: str) -> str:
        if self._llm_callable is not None:
            return self._llm_callable(prompt)
        if self.model == "mock":
            return json.dumps(_mock_variant(prompt))
        raise NotImplementedError(
            f"真实 LLM 未实现, 请提供 llm_callable 或使用 model='mock'"
        )


class Hypothesizer(BaseOperator):
    """从研究假设生成初始因子 (round 0)。"""

    def hypothesize(
        self,
        direction: str,
        description: str = "",
    ) -> FactorCandidate:
        prompt = _HYPOTHESIZE_PROMPT.format(
            hypothesis=direction,
            description=description,
        )
        for attempt in range(self.max_correction_attempts + 1):
            try:
                raw = self._call(prompt)
                data = json.loads(raw)
                return FactorCandidate(
                    factor_id=str(uuid.uuid4()),
                    name=str(data.get("name", f"h_{direction[:8]}")),
                    expression=str(data["expression"]),
                    hypothesis=direction,
                    description=str(data.get("description", description)),
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                if attempt == self.max_correction_attempts:
                    # 兜底: 用 mock 直接生成
                    data = _mock_variant(prompt)
                    return FactorCandidate(
                        factor_id=str(uuid.uuid4()),
                        name=str(data.get("name", f"h_{direction[:8]}")),
                        expression=str(data["expression"]),
                        hypothesis=direction,
                        description=str(data.get("description", description)),
                    )
                continue
        raise RuntimeError("unreachable")  # for type checker


class Mutator(BaseOperator):
    """从单个 parent 生成 mutation 子代。"""

    def mutate(self, parent: FactorCandidate) -> FactorCandidate:
        prompt = _MUTATE_PROMPT.format(
            parent_expression=parent.expression,
            parent_hypothesis=parent.hypothesis,
            parent_description=parent.description,
        )
        for attempt in range(self.max_correction_attempts + 1):
            try:
                raw = self._call(prompt)
                data = json.loads(raw)
                return FactorCandidate(
                    factor_id=str(uuid.uuid4()),
                    name=str(data.get("name", f"m_{parent.name}")),
                    expression=str(data["expression"]),
                    hypothesis=parent.hypothesis,
                    description=str(data.get("description", parent.description)),
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                if attempt == self.max_correction_attempts:
                    data = _mock_variant(prompt)
                    return FactorCandidate(
                        factor_id=str(uuid.uuid4()),
                        name=str(data.get("name", f"m_{parent.name}")),
                        expression=str(data["expression"]),
                        hypothesis=parent.hypothesis,
                        description=str(data.get("description", parent.description)),
                    )
                continue
        raise RuntimeError("unreachable")


class Crosser(BaseOperator):
    """从两个 parent 生成 crossover 子代。"""

    def crossover(
        self,
        parent1: FactorCandidate,
        parent2: FactorCandidate,
    ) -> FactorCandidate:
        prompt = _CROSSOVER_PROMPT.format(
            p1_expression=parent1.expression,
            p1_description=parent1.description,
            p2_expression=parent2.expression,
            p2_description=parent2.description,
        )
        for attempt in range(self.max_correction_attempts + 1):
            try:
                raw = self._call(prompt)
                data = json.loads(raw)
                return FactorCandidate(
                    factor_id=str(uuid.uuid4()),
                    name=str(data.get("name", f"x_{parent1.name}_{parent2.name}")),
                    expression=str(data["expression"]),
                    hypothesis=f"combo({parent1.hypothesis}, {parent2.hypothesis})",
                    description=str(data.get("description", f"combo of {parent1.name} + {parent2.name}")),
                )
            except (json.JSONDecodeError, KeyError, TypeError):
                if attempt == self.max_correction_attempts:
                    data = _mock_variant(prompt)
                    return FactorCandidate(
                        factor_id=str(uuid.uuid4()),
                        name=str(data.get("name", f"x_{parent1.name}_{parent2.name}")),
                        expression=str(data["expression"]),
                        hypothesis=f"combo({parent1.hypothesis}, {parent2.hypothesis})",
                        description=str(data.get("description", f"combo of {parent1.name} + {parent2.name}")),
                    )
                continue
        raise RuntimeError("unreachable")


# ============================================================================
# Mock variant generator (heuristic-based fallback)
# ============================================================================

_MUTATION_TEMPLATES = [
    lambda e: f"({e}).rolling(5).mean()",
    lambda e: f"({e}).diff()",
    lambda e: f"({e}).rank(pct=True)",
    lambda e: f"({e}) - ({e}).shift(5)",
    lambda e: f"({e}) * 2",
    lambda e: f"({e}).abs()",
]

_CROSSOVER_TEMPLATES = [
    lambda a, b: f"({a}) + ({b})",
    lambda a, b: f"({a}) - ({b})",
    lambda a, b: f"({a}) * ({b})",
    lambda a, b: f"(({a}) + ({b})) / 2",
    lambda a, b: f"({a}).rank(pct=True) - ({b}).rank(pct=True)",
]


def _mock_variant(prompt: str) -> dict:
    """基于 prompt 内容启发式生成变体。"""
    # 提取父表达式 (按 "父因子: <expr>" 模式)
    parent_match = re.search(r"父因子[:\s]+([^\n]+)", prompt)
    parents_match = re.findall(r"父因子\s*\d*[:\s]+([^\n(]+)", prompt)
    hyp_match = re.search(r"研究假设[:\s]+([^\n]+)", prompt)

    hyp = hyp_match.group(1).strip() if hyp_match else "alpha"

    if len(parents_match) >= 2:
        # crossover
        idx = sum(ord(c) for c in prompt) % len(_CROSSOVER_TEMPLATES)
        tpl = _CROSSOVER_TEMPLATES[idx]
        expr = tpl(parents_match[0].strip(), parents_match[1].strip())
        return {
            "name": f"x_mock_{idx}",
            "expression": expr,
            "description": f"combo of {parents_match[0][:20]} + {parents_match[1][:20]}",
        }
    if parent_match:
        parent_expr = parent_match.group(1).strip()
        idx = sum(ord(c) for c in prompt) % len(_MUTATION_TEMPLATES)
        tpl = _MUTATION_TEMPLATES[idx]
        return {
            "name": f"m_mock_{idx}",
            "expression": tpl(parent_expr),
            "description": f"mutation of {parent_expr[:30]}",
        }
    # hypothesize fallback
    return {
        "name": f"h_mock_{hyp[:8]}",
        "expression": "(close - close.shift(20)) / close.shift(20)",
        "description": f"default expression for {hyp}",
    }
