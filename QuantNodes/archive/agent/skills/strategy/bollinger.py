# coding=utf-8
"""
Bollinger Bands Strategy Skill

Phase 4.3: Strategy Skills
"""

from typing import Any, Dict

from ..base import Skill, SkillCategory, SkillMetadata, SkillResult


class BollingerSkill(Skill):
    """Bollinger Bands Strategy"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="bollinger_strategy",
            description="布林带策略 - 价格触及下轨买入，触及上轨卖出",
            category=SkillCategory.STRATEGY,
            tags=["趋势跟踪", "布林带", "均值回归"],
            examples=[
                "生成布林带策略代码",
                "布林带策略参数优化",
            ],
        )

    async def execute(self, context: Dict[str, Any]) -> SkillResult:
        """Execute Bollinger bands strategy generation"""
        period = context.get("period", 20)
        std_dev = context.get("std_dev", 2)

        code = f'''
def bollinger_strategy(data, period={period}, std_dev={std_dev}):
    """
    布林带策略
    - Period: {period}日移动平均
    - Std Dev: {std_dev}倍标准差
    Signals:
      - 价格触及下轨买入(1)
      - 价格触及上轨卖出(-1)
      - 持有(0)
    """
    import pandas as pd
    
    data["mid"] = data["close"].rolling({period}).mean()
    data["std"] = data["close"].rolling({period}).std()
    data["upper"] = data["mid"] + {std_dev} * data["std"]
    data["lower"] = data["mid"] - {std_dev} * data["std"]
    
    data["signal"] = 0
    data.loc[data["close"] <= data["lower"], "signal"] = 1
    data.loc[data["close"] >= data["upper"], "signal"] = -1
    
    data["position"] = data["signal"].shift(1)
    return data
'''
        return SkillResult(
            success=True,
            data={
                "strategy": "bollinger",
                "period": period,
                "std_dev": std_dev,
                "code": code.strip(),
                "description": f"布林带策略: {period}日周期，{std_dev}倍标准差",
            },
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return parameter schema"""
        return {
            "type": "object",
            "properties": {
                "period": {
                    "type": "integer",
                    "description": "移动平均周期",
                    "default": 20,
                },
                "std_dev": {
                    "type": "number",
                    "description": "标准差倍数",
                    "default": 2,
                },
            },
        }