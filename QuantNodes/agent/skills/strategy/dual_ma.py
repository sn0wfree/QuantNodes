# coding=utf-8
"""
Dual Moving Average Strategy Skill

Phase 4.3: Strategy Skills
"""

from typing import Any, Dict

from ..base import Skill, SkillCategory, SkillMetadata, SkillResult


class DualMaSkill(Skill):
    """Dual Moving Average Crossover Strategy"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="dual_ma_strategy",
            description="双均线交叉策略 - 短周期均线上穿长周期均线买入，下穿卖出",
            category=SkillCategory.STRATEGY,
            tags=["趋势跟踪", "均线", "双均线"],
            examples=[
                "生成双均线策略代码",
                "如何使用双均线指标",
            ],
        )

    async def execute(self, context: Dict[str, Any]) -> SkillResult:
        """Execute dual MA strategy generation"""
        fast_period = context.get("fast_period", 5)
        slow_period = context.get("slow_period", 20)

        code = f'''
def dual_ma_strategy(data, fast_period={fast_period}, slow_period={slow_period}):
    """
    双均线策略
    - Fast MA: {fast_period}日均线
    - Slow MA: {slow_period}日均线
    Signals:
      - 金叉买入(1), 死叉卖出(-1), 持有(0)
    """
    import pandas as pd

    data["fast_ma"] = data["close"].rolling({fast_period}).mean()
    data["slow_ma"] = data["close"].rolling({slow_period}).mean()

    data["signal"] = 0
    data.loc[data["fast_ma"] > data["slow_ma"], "signal"] = 1
    data.loc[data["fast_ma"] <= data["slow_ma"], "signal"] = -1

    data["position"] = data["signal"].shift(1)
    return data
'''
        return SkillResult(
            success=True,
            data={
                "strategy": "dual_ma",
                "fast_period": fast_period,
                "slow_period": slow_period,
                "code": code.strip(),
                "description": f"双均线策略: {fast_period}日均线与{slow_period}日均线交叉",
            },
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return parameter schema"""
        return {
            "type": "object",
            "properties": {
                "fast_period": {
                    "type": "integer",
                    "description": "快速均线周期",
                    "default": 5,
                },
                "slow_period": {
                    "type": "integer",
                    "description": "慢速均线周期",
                    "default": 20,
                },
            },
        }
