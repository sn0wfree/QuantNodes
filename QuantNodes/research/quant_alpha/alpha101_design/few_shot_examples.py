# coding=utf-8
"""
few_shot_examples.py - Alpha 101 few-shot 示例（用于 Alpha-GPT 启动 prompt）

提供 5-10 个代表性公式示例，供 M5+ 的 Alpha-GPT 路线使用。
本 M3 PR 仅做示例定义，不做实际评估（评估由 Alpha-GPT 路线完成）。

示例选样标准：
- 覆盖 8 条设计原则
- 覆盖 16 个核心算子
- A 股可移植（避开 Delay-0）
- 复杂度从简单（1-2 算子）到复杂（5+ 算子）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FewShotExample:
    """Alpha 101 few-shot 示例"""
    id: str  # 示例 ID (EX1-EX10)
    name: str  # 简短名称
    formula: str  # 公式
    description: str  # 含义
    category: str  # momentum / reversal / volume_price / volatility / intraday
    design_principles: List[str]  # 引用的设计原则 ID (P1-P8)
    operators_used: List[str]  # 使用的核心算子
    alpha101_ref: str  # Alpha 101 #N 引用
    a_share_compatible: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# 10 个精选 few-shot 示例
# ==============================================================================

ALPHA101_FEW_SHOT_EXAMPLES: List[FewShotExample] = [
    # ============ 简单示例（1-2 算子）============
    FewShotExample(
        id="EX1",
        name="日内动量",
        formula="(close - open) / ((high - low) + 0.001)",
        description=(
            "日内收益率除以日内振幅。"
            "衡量收盘相对开盘的优势，结合日内波动率标准化。"
            "值越大表示日内动量越强。"
        ),
        category="intraday",
        design_principles=["P1", "P2"],
        operators_used=["point:pointwise"],
        alpha101_ref="#101",
        a_share_compatible=True,
    ),
    FewShotExample(
        id="EX2",
        name="截面动量（5 日）",
        formula="rank(close / ts_lag(close, 5) - 1)",
        description=(
            "5 日动量做截面 rank。"
            "捕捉过去 5 天涨幅最大的股票，趋势跟随策略。"
        ),
        category="momentum",
        design_principles=["P2", "P3"],
        operators_used=["ts_lag", "rank"],
        alpha101_ref="#N/A",
        a_share_compatible=True,
    ),
    FewShotExample(
        id="EX3",
        name="量价背离",
        formula="-1 * ts_corr(open, vol, 10)",
        description=(
            "开盘价与成交量的 10 日滚动相关系数取负。"
            "捕捉量价背离现象：股价涨但成交量不增时，预示反转。"
        ),
        category="volume_price",
        design_principles=["P2", "P8"],
        operators_used=["ts_corr"],
        alpha101_ref="#6",
        a_share_compatible=True,
    ),

    # ============ 中等示例（3-4 算子）============
    FewShotExample(
        id="EX4",
        name="Alpha 101 #1 简化版",
        formula="rank(ts_argmax(signedpower(close, 2), 5)) - 0.5",
        description=(
            "5 日内 close² 最大值出现的位置做截面 rank 并居中。"
            "极端值在最近 = 强动量；5 天前 = 弱动量。"
            "体现了 signedpower 保留符号 + argmax 提取位置的核心思想。"
        ),
        category="momentum",
        design_principles=["P3", "P4", "P5"],
        operators_used=["rank", "ts_argmax", "signedpower"],
        alpha101_ref="#1 (简化版)",
        a_share_compatible=True,
    ),
    FewShotExample(
        id="EX5",
        name="量增价跌反转",
        formula="sign(ts_delta(vol, 1)) * (-1 * ts_delta(close, 1))",
        description=(
            "量增 + 价跌 = -1（卖出信号）。"
            "量缩 + 价涨 = +1（买入信号）。"
            "其他情况 = 0（无信号）。"
            "这是 Alpha 101 #12 完整版。"
        ),
        category="volume_price",
        design_principles=["P1", "P2"],
        operators_used=["sign", "ts_delta"],
        alpha101_ref="#12",
        a_share_compatible=True,
    ),
    FewShotExample(
        id="EX6",
        name="波动率加权动量",
        formula="rank(ts_mean(close, 8) - ts_mean(close, 21) + ts_std(close, 8))",
        description=(
            "8 日均线相对 21 日均线的偏离度，加上 8 日波动率。"
            "高偏离 + 高波动 = 强动量或反转信号。"
            "（Alpha 101 #21 简化版）"
        ),
        category="momentum",
        design_principles=["P1", "P3"],
        operators_used=["rank", "ts_mean", "ts_std"],
        alpha101_ref="#21 (简化版)",
        a_share_compatible=True,
    ),

    # ============ 复杂示例（5+ 算子）============
    FewShotExample(
        id="EX7",
        name="波动率反转（ArgMax 提取）",
        formula="rank(ts_argmax(signedpower(close, 2), 5) - ts_argmax(signedpower(close, 2), 20))",
        description=(
            "5 日 vs 20 日的 signedpower 极值位置差。"
            "差值大 = 最近 5 日的极值更新（趋势延续或反转）。"
            "差值小 = 20 日前就有极值（趋势已成熟）。"
        ),
        category="reversal",
        design_principles=["P3", "P4", "P5"],
        operators_used=["rank", "ts_argmax", "signedpower"],
        alpha101_ref="#N/A (派生)",
        a_share_compatible=True,
    ),
    FewShotExample(
        id="EX8",
        name="量价加权反转",
        formula="ts_corr(close, vol, 10) * ts_mean(vol, 20)",
        description=(
            "10 日量价相关 + 20 日均量。"
            "捕捉量价同向 + 高量能时的反转机会。"
        ),
        category="volume_price",
        design_principles=["P1", "P2"],
        operators_used=["ts_corr", "ts_mean"],
        alpha101_ref="#N/A (派生)",
        a_share_compatible=True,
    ),
    FewShotExample(
        id="EX9",
        name="线性加权动量",
        formula="rank(decay_linear(close / ts_lag(close, 5) - 1, 10))",
        description=(
            "5 日收益率做 10 日线性衰减加权（近期权重高）。"
            "捕捉短期动量的持续性。"
        ),
        category="momentum",
        design_principles=["P2", "P3", "P6"],
        operators_used=["rank", "decay_linear", "ts_lag"],
        alpha101_ref="#N/A (派生)",
        a_share_compatible=True,
    ),
    FewShotExample(
        id="EX10",
        name="Alpha 101 #89 完整版",
        formula="rank(decay_linear(ts_sum(close, 10) / 10, 10))",
        description=(
            "10 日 close 求和 / 10 = 10 日均价（与 ts_mean 等价）。"
            "再 10 日线性衰减加权（近期权重高）。"
            "最后截面 rank。"
            "这是 Alpha 101 #89 的 polars 表达。"
        ),
        category="momentum",
        design_principles=["P3", "P6"],
        operators_used=["rank", "decay_linear", "ts_sum"],
        alpha101_ref="#89",
        a_share_compatible=True,
    ),
]


# ==============================================================================
# 辅助函数
# ==============================================================================


def list_examples(category: Optional[str] = None) -> List[FewShotExample]:
    """列出所有示例（可按 category 过滤）"""
    if category is None:
        return list(ALPHA101_FEW_SHOT_EXAMPLES)
    return [e for e in ALPHA101_FEW_SHOT_EXAMPLES if e.category == category]


def get_example(example_id: str) -> Optional[FewShotExample]:
    """按 ID 查示例"""
    for e in ALPHA101_FEW_SHOT_EXAMPLES:
        if e.id == example_id:
            return e
    return None


def get_few_shot_prompt(
    n: int = 5,
    category: Optional[str] = None,
) -> str:
    """构造 few-shot prompt（用于 Alpha-GPT 启动 prompt）

    Args:
        n: 示例数量
        category: 可选 category 过滤

    Returns:
        多行字符串，每行一个示例
    """
    examples = list_examples(category)[:n]
    lines = []
    for e in examples:
        lines.append(
            f"# {e.id} {e.name}\n"
            f"formula: {e.formula}\n"
            f"description: {e.description}\n"
        )
    return "\n".join(lines)


def get_categories() -> List[str]:
    """获取所有 category"""
    return list(set(e.category for e in ALPHA101_FEW_SHOT_EXAMPLES))
