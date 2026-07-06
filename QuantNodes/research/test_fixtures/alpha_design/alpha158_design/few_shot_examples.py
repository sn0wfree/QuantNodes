# coding=utf-8
"""
few_shot_examples.py - Alpha 158/360 few-shot 示例（用于 Alpha-GPT 启动 prompt）

提供 5-10 个代表性特征示例（覆盖 4 类），供 M5+ 的 Alpha-GPT 路线使用。
本 M3 PR 仅做示例定义。

示例选样标准：
- 覆盖 4 类（KBAR / Price / Volume / Rolling）
- 简单 → 复杂
- 经典 + 衍生
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class FewShotExample:
    """Alpha 158/360 few-shot 示例"""
    id: str  # 示例 ID (FX1-FX10)
    name: str  # 简短名称
    formula: str  # 公式
    description: str  # 含义
    category: str  # KBAR / Price / Volume / Rolling
    operators_used: List[str]  # 使用的算子
    qlib_ref: str  # Qlib 引用
    alpha158_or_360: str = "alpha158"  # 属于哪个特征集
    metadata: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# 10 个精选 few-shot 示例（覆盖 4 类）
# ==============================================================================


ALPHA158_FEW_SHOT_EXAMPLES: List[FewShotExample] = [
    # ============ KBAR (2 个) ============
    FewShotExample(
        id="FX1",
        name="日内实体占比 KMID",
        formula="(close - open) / open",
        description=(
            "今日实体（close-open）占开盘价的比例。"
            "正值=阳线，负值=阴线。"
            "Alpha 158 KBAR 特征第 1 个。"
        ),
        category="KBAR",
        operators_used=["point"],
        qlib_ref="$KMID",
    ),
    FewShotExample(
        id="FX2",
        name="日内振幅 KLEN",
        formula="(high - low) / open",
        description=(
            "日内最高减最低的振幅除以开盘价。"
            "捕捉日内波动率。"
        ),
        category="KBAR",
        operators_used=["point"],
        qlib_ref="$KLEN",
    ),

    # ============ Price (2 个) ============
    FewShotExample(
        id="FX3",
        name="昨日开盘相对今日收盘",
        formula="open.shift(1) / close",
        description=(
            "昨日开盘价 / 今日收盘价。"
            "捕捉隔夜 gap 和当日走势。"
        ),
        category="Price",
        operators_used=["ts_lag"],
        qlib_ref="$OPEN1",
    ),
    FewShotExample(
        id="FX4",
        name="3 日前最高相对今日收盘",
        formula="high.shift(3) / close",
        description=(
            "3 日前最高价 / 今日收盘价。"
            "Alpha 158 Price 特征 4 字段 × 5 延迟中的一个。"
        ),
        category="Price",
        operators_used=["ts_lag"],
        qlib_ref="$HIGH3",
    ),

    # ============ Volume (1 个) ============
    FewShotExample(
        id="FX5",
        name="3 日前成交量比",
        formula="volume.shift(3) / (volume + 1e-12)",
        description=(
            "3 日前成交量 / 今日成交量。"
            "Alpha 158 Volume 5 特征之一。"
        ),
        category="Volume",
        operators_used=["ts_lag"],
        qlib_ref="$VOLUME3",
    ),

    # ============ Rolling (5 个) ============
    FewShotExample(
        id="FX6",
        name="20 日均线",
        formula="close.rolling(20).mean()",
        description=(
            "20 日收盘价简单移动平均。"
            "Alpha 158 Rolling MA 之一（25 op × 5 window = 125 MA 类特征）。"
        ),
        category="Rolling",
        operators_used=["ts_mean"],
        qlib_ref="$MA20",
    ),
    FewShotExample(
        id="FX7",
        name="20 日变化率 ROC",
        formula="close / close.shift(20) - 1",
        description=(
            "20 日收益率：今日 close / 20 日前 close - 1。"
            "Alpha 158 Rolling ROC 之一。"
        ),
        category="Rolling",
        operators_used=["ts_lag"],
        qlib_ref="$ROC20",
    ),
    FewShotExample(
        id="FX8",
        name="20 日波动率 STD",
        formula="close.rolling(20).std()",
        description=(
            "20 日收盘价标准差（年化前需 × sqrt(252)）。"
            "Alpha 158 Rolling STD 之一。"
        ),
        category="Rolling",
        operators_used=["ts_std"],
        qlib_ref="$STD20",
    ),
    FewShotExample(
        id="FX9",
        name="量价 20 日相关 CORR",
        formula="close.rolling(20).corr(volume)",
        description=(
            "20 日量价滚动相关系数。"
            "Alpha 158 Rolling CORR 之一。"
            "量价同涨同跌为 +1，反向 -1。"
        ),
        category="Rolling",
        operators_used=["ts_corr"],
        qlib_ref="$CORR20",
    ),
    FewShotExample(
        id="FX10",
        name="20 日极值位置 IMAX",
        formula="ts_argmax(high, 20)",
        description=(
            "20 日内最高价出现的位置（距今天数）。"
            "值小（最近创新高）= 强动量。"
            "Alpha 158 Rolling IMAX 之一。"
        ),
        category="Rolling",
        operators_used=["ts_argmax"],
        qlib_ref="$IMAX20",
    ),
]


# ==============================================================================
# 辅助函数
# ==============================================================================


def list_examples(category: Optional[str] = None) -> List[FewShotExample]:
    """列出所有示例（可按 category 过滤）"""
    if category is None:
        return list(ALPHA158_FEW_SHOT_EXAMPLES)
    return [e for e in ALPHA158_FEW_SHOT_EXAMPLES if e.category == category]


def get_example(example_id: str) -> Optional[FewShotExample]:
    """按 ID 查示例"""
    for e in ALPHA158_FEW_SHOT_EXAMPLES:
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
            f"# {e.id} {e.name} [{e.category}]\n"
            f"formula: {e.formula}\n"
            f"description: {e.description}\n"
        )
    return "\n".join(lines)


def get_categories() -> List[str]:
    """获取所有 category"""
    return list(set(e.category for e in ALPHA158_FEW_SHOT_EXAMPLES))
