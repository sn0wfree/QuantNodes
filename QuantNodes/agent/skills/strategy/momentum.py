# coding=utf-8
"""
Momentum Strategy Skill

Phase 4.3: Strategy Skills
"""

from typing import Any, Dict

from ..base import Skill, SkillCategory, SkillMetadata, SkillResult


class MomentumSkill(Skill):
    """Momentum Strategy"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="momentum_strategy",
            description="动量策略 - 追涨杀跌，过去收益率为正则持有，为负则卖出",
            category=SkillCategory.STRATEGY,
            tags=["趋势跟踪", "动量", "追涨杀跌"],
            examples=[
                "生成动量策略代码",
                "动量因子参数设置",
            ],
        )

    async def execute(self, context: Dict[str, Any]) -> SkillResult:
        """Execute momentum strategy generation"""
        period = context.get("period", 20)

        code = f'''
def momentum_strategy(data, period={period}):
    """
    动量策略
    - Period: {period}日回顾期
    Signals:
      - 动量为正持有(1)
      - 动量为负卖出(-1)
    """
    import pandas as pd

    data["momentum"] = data["close"] / data["close"].shift({period}) - 1

    data["signal"] = 0
    data.loc[data["momentum"] > 0, "signal"] = 1
    data.loc[data["momentum"] <= 0, "signal"] = -1

    data["position"] = data["signal"].shift(1)
    return data
'''
        return SkillResult(
            success=True,
            data={
                "strategy": "momentum",
                "period": period,
                "code": code.strip(),
                "description": f"动量策略: {period}日回顾期",
            },
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return parameter schema"""
        return {
            "type": "object",
            "properties": {
                "period": {
                    "type": "integer",
                    "description": "动量回顾周期",
                    "default": 20,
                },
            },
        }
