# coding=utf-8
"""
RSI Mean Reversion Strategy Skill

Phase 4.3: Strategy Skills
"""

from typing import Any, Dict

from ..base import Skill, SkillCategory, SkillMetadata, SkillResult


class RSIReversalSkill(Skill):
    """RSI Mean Reversion Strategy"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="rsi_reversal_strategy",
            description="RSI均值回归策略 - RSI超卖买入，RSI超买卖出",
            category=SkillCategory.STRATEGY,
            tags=["均值回归", "RSI", "超买超卖"],
            examples=[
                "生成RSI均值回归策略代码",
                "RSI参数优化",
            ],
        )

    async def execute(self, context: Dict[str, Any]) -> SkillResult:
        """Execute RSI mean reversion strategy generation"""
        period = context.get("period", 14)
        oversold = context.get("oversold", 30)
        overbought = context.get("overbought", 70)

        code = f'''
def rsi_reversal_strategy(data, period={period}, oversold={oversold}, overbought={overbought}):
    """
    RSI均值回归策略
    - Period: {period}日RSI计算周期
    - Oversold: {oversold} (超卖阈值)
    - Overbought: {overbought} (超买阈值)
    Signals:
      - RSI超卖时买入(1)
      - RSI超买时卖出(-1)
      - 持有(0)
    """
    import pandas as pd
    
    delta = data["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling({period}).mean()
    avg_loss = loss.rolling({period}).mean()
    
    rs = avg_gain / avg_loss
    data["rsi"] = 100 - (100 / (1 + rs))
    
    data["signal"] = 0
    data.loc[data["rsi"] < {oversold}, "signal"] = 1
    data.loc[data["rsi"] > {overbought}, "signal"] = -1
    
    data["position"] = data["signal"].shift(1)
    return data
'''
        return SkillResult(
            success=True,
            data={
                "strategy": "rsi_reversal",
                "period": period,
                "oversold": oversold,
                "overbought": overbought,
                "code": code.strip(),
                "description": f"RSI均值回归: 周期{period}, 超卖{oversold}, 超买{overbought}",
            },
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return parameter schema"""
        return {
            "type": "object",
            "properties": {
                "period": {
                    "type": "integer",
                    "description": "RSI计算周期",
                    "default": 14,
                },
                "oversold": {
                    "type": "integer",
                    "description": "超卖阈值",
                    "default": 30,
                },
                "overbought": {
                    "type": "integer",
                    "description": "超买阈值",
                    "default": 70,
                },
            },
        }