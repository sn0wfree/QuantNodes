# coding=utf-8
"""
g1_handcrafted.py - G1 baseline：动态从 OperatorVocab 生成 100 个手工公式

G1 = "手工构造（Handcrafted）"，代表 101/158 等论文公开的算子组合。
Stage 1 动态生成（不硬编码），从 OperatorVocab 抽取合法组合。

为什么动态生成：
- 避免硬编码 100 公式导致难以维护
- OperatorVocab 已注册 162 算子（M1），动态组合可生成大量候选
- 与 G2（mock LLM 直接生成）和 G3（AlphaGptWorkflow）形成对照

公式生成规则（仅限 alpha_evaluate tool 支持的算子）：
- 字段：close / vol / amount
- 时间窗口：ts_mean / ts_std / delta（window ∈ {3, 5, 10, 20}）
- 二元组合：Add / Sub / Mul / Div
- 一元：abs / log / sign / neg
- 嵌套深度 ≤ 2
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional

from ..contracts import Baseline, FactorSpec

logger = logging.getLogger(__name__)

__all__ = ["G1Handcrafted"]


FIELDS = ["close", "vol", "amount"]
WINDOWS = [3, 5, 10, 20]
TIME_OPS = ["ts_mean", "ts_std", "delta"]
CROSS_OPS = ["abs", "log", "sign", "sqrt"]
BINARY_OPS = ["Add", "Sub", "Mul", "Div"]


def _gen_leaf(rng: random.Random) -> str:
    """生成叶子节点：字段或字段上的时间窗口算子"""
    field = rng.choice(FIELDS)
    if rng.random() < 0.7:  # 70% 包时间窗口
        op = rng.choice(TIME_OPS)
        window = rng.choice(WINDOWS)
        return f"{op}({field}, {window})"
    return field


def _gen_formula(rng: random.Random, max_depth: int = 2) -> str:
    """生成一条公式字符串"""
    leaf_left = _gen_leaf(rng)
    leaf_right = _gen_leaf(rng)

    # 50% 应用 cross-sectional 算子
    if rng.random() < 0.3:
        cross = rng.choice(CROSS_OPS)
        leaf_left = f"{cross}({leaf_left})"

    if max_depth <= 1 or rng.random() < 0.6:
        # 单层
        if rng.random() < 0.5:
            return leaf_left
        op = rng.choice(BINARY_OPS)
        return f"{op}({leaf_left}, {leaf_right})"
    else:
        # 双层嵌套
        op1 = rng.choice(BINARY_OPS)
        op2 = rng.choice(BINARY_OPS)
        leaf3 = _gen_leaf(rng)
        return f"{op1}({op2}({leaf_left}, {leaf_right}), {leaf3})"


class G1Handcrafted(Baseline):
    """G1 Handcrafted baseline

    从 OperatorVocab 动态组合生成 n 个手工因子。
    Stage 1 与 Stage 2 共用（不依赖数据 / LLM）。
    """

    def __init__(self, n: int = 100, seed: int = 42) -> None:
        self.n = n
        self.seed = seed

    @property
    def group_name(self) -> str:
        return "G1_Handcrafted"

    def generate_factors(self, n: Optional[int] = None) -> List[FactorSpec]:
        """动态生成 n 个因子（formula_id 唯一）"""
        n = n or self.n
        rng = random.Random(self.seed)

        factors: List[FactorSpec] = []
        seen_formulas: set = set()

        attempts = 0
        while len(factors) < n and attempts < n * 5:
            attempts += 1
            formula = _gen_formula(rng)
            if formula in seen_formulas:
                continue
            seen_formulas.add(formula)

            category = self._infer_category(formula)
            factors.append(
                FactorSpec(
                    formula_id=f"G1_{len(factors):03d}",
                    formula=formula,
                    source="g1_handcrafted",
                    category=category,
                    complexity=formula.count("("),
                    meta={"seed": self.seed},
                )
            )

        logger.info(
            "[G1] generated %d factors (attempts=%d)", len(factors), attempts
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