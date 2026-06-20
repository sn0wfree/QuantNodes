# coding=utf-8
"""
Group Backtest Skill

Phase 4.4: Factor Skills
"""

from typing import Any, Dict

from ..base import Skill, SkillCategory, SkillMetadata, SkillResult


class GroupBacktestSkill(Skill):
    """Group Backtest by Factor Quantiles"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="group_backtest",
            description="按因子分位数分组回测，计算各组收益差异",
            category=SkillCategory.FACTOR,
            tags=["因子分析", "分组回测", "分位数", "多空组合"],
            examples=[
                "按因子分5组回测",
                "计算多空组合收益",
            ],
        )

    async def execute(self, context: Dict[str, Any]) -> SkillResult:
        """Execute group backtest"""
        n_groups = context.get("n_groups", 5)
        period = context.get("period", 20)

        code = f'''
def group_backtest(
    data, factor_col="factor", return_col="return",
    n_groups={n_groups}, period={period},
):
    """
    分组回测
    - 按因子值分{n_groups}组
    - 计算每组未来{period}天收益
    - 计算多空组合收益 (Group{n_groups} - Group1)

    Parameters:
    - data: DataFrame with factor and return columns
    - factor_col: 因子列名
    - return_col: 收益列名
    - n_groups: 分组数量
    - period: 持有周期

    Returns:
    - group_returns: 各组平均收益
    - long_short_return: 多空组合收益
    - turnover: 换手率
    """
    import pandas as pd
    import numpy as np

    data = data.copy()
    data["group"] = pd.qcut(data[factor_col], q={n_groups}, labels=False, duplicates="drop")

    future_returns = data[return_col].shift(-{period})
    data["future_return"] = future_returns

    group_returns = data.groupby("group")["future_return"].mean()

    if len(group_returns) >= {n_groups}:
        long_short_return = group_returns.iloc[-1] - group_returns.iloc[0]
    else:
        long_short_return = np.nan

    position = data.groupby("date")["group"].shift(1)
    turnover = (position != position.shift(1)).sum() / len(data)

    result = {{
        "n_groups": {n_groups},
        "period": {period},
        "group_returns": group_returns.to_dict(),
        "long_short_return": long_short_return,
        "turnover": turnover,
        "top_group_return": group_returns.iloc[-1] if len(group_returns) > 0 else np.nan,
        "bottom_group_return": group_returns.iloc[0] if len(group_returns) > 0 else np.nan,
    }}
    return result
'''
        return SkillResult(
            success=True,
            data={
                "skill": "group_backtest",
                "n_groups": n_groups,
                "period": period,
                "code": code.strip(),
                "description": f"分组回测: {n_groups}组, {period}天持有期",
            },
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return parameter schema"""
        return {
            "type": "object",
            "properties": {
                "n_groups": {
                    "type": "integer",
                    "description": "分组数量",
                    "default": 5,
                },
                "period": {
                    "type": "integer",
                    "description": "持有周期(天)",
                    "default": 20,
                },
            },
        }
