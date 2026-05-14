# coding=utf-8
"""
IC Analysis Skill

Phase 4.4: Factor Skills
"""

from typing import Any, Dict

from ..base import Skill, SkillCategory, SkillMetadata, SkillResult


class ICAnalysisSkill(Skill):
    """IC (Information Coefficient) Analysis Skill"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="ic_analysis",
            description="计算并分析因子IC值，包括IC均值、ICIR、Rank IC等指标",
            category=SkillCategory.FACTOR,
            tags=["因子分析", "IC", "ICIR", "Rank IC"],
            examples=[
                "分析因子的IC表现",
                "计算ICIR指标",
            ],
        )

    async def execute(self, context: Dict[str, Any]) -> SkillResult:
        """Execute IC analysis"""
        factor_name = context.get("factor_name", "factor")
        n_days = context.get("n_days", 252)

        code = f'''
def calculate_ic_analysis(data, factor_col="factor", return_col="return", n_days={n_days}):
    """
    IC分析
    - 计算因子IC序列
    - 计算IC均值、IC标准差、ICIR
    - 计算Rank IC
    
    Parameters:
    - data: DataFrame with factor and return columns
    - factor_col: 因子列名
    - return_col: 收益列名
    - n_days: 计算周期
    
    Returns:
    - ic_series: IC时间序列
    - ic_mean: IC均值
    - ic_std: IC标准差
    - icir: ICIR (IC均值/IC标准差)
    - rank_ic_mean: Rank IC均值
    """
    import pandas as pd
    import numpy as np
    
    ic_series = data[factor_col].corr(data[return_col])
    rank_ic_series = data[factor_col].rank().corr(data[return_col].rank())
    
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    icir = ic_mean / ic_std if ic_std != 0 else 0
    
    rank_ic_mean = rank_ic_series.mean()
    
    result = {{
        "factor_name": "{factor_name}",
        "n_dates": n_days,
        "ic_series": ic_series,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": icir,
        "rank_ic_mean": rank_ic_mean,
        "ic_t_stat": ic_mean / (ic_std / np.sqrt(n_days)) if ic_std != 0 else 0,
    }}
    return result
'''
        return SkillResult(
            success=True,
            data={
                "skill": "ic_analysis",
                "factor_name": factor_name,
                "n_days": n_days,
                "code": code.strip(),
                "description": f"IC分析: {factor_name}, {n_days}天",
            },
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return parameter schema"""
        return {
            "type": "object",
            "properties": {
                "factor_name": {
                    "type": "string",
                    "description": "因子名称",
                    "default": "factor",
                },
                "n_days": {
                    "type": "integer",
                    "description": "分析天数",
                    "default": 252,
                },
            },
        }