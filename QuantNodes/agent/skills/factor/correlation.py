# coding=utf-8
"""
Correlation Analysis Skill

Phase 4.4: Factor Skills
"""

from typing import Any, Dict

from ..base import Skill, SkillCategory, SkillMetadata, SkillResult


class CorrelationSkill(Skill):
    """Factor Correlation Analysis Skill"""

    @property
    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="correlation",
            description="计算因子之间的相关性矩阵，分析因子独立性",
            category=SkillCategory.FACTOR,
            tags=["因子分析", "相关性", "多重共线性"],
            examples=[
                "计算多个因子的相关性矩阵",
                "检测高相关性因子",
            ],
        )

    async def execute(self, context: Dict[str, Any]) -> SkillResult:
        """Execute correlation analysis"""
        factors = context.get("factors", ["factor1", "factor2"])

        code = f'''
def correlation_analysis(data, factor_cols={factors}):
    """
    相关性分析
    - 计算因子之间的Pearson相关系数矩阵
    - 识别高相关性因子对 (|corr| > 0.8)

    Parameters:
    - data: DataFrame with factor columns
    - factor_cols: 因子列名列表

    Returns:
    - corr_matrix: 相关性矩阵
    - high_corr_pairs: 高相关性因子对
    """
    import pandas as pd
    import numpy as np

    factor_data = data[factor_cols]

    corr_matrix = factor_data.corr()

    high_corr_threshold = 0.8
    high_corr_pairs = []
    for i in range(len(factor_cols)):
        for j in range(i + 1, len(factor_cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > high_corr_threshold:
                high_corr_pairs.append({{
                    "factor1": factor_cols[i],
                    "factor2": factor_cols[j],
                    "correlation": corr_val,
                }})

    result = {{
        "factors": {factors},
        "corr_matrix": corr_matrix.to_dict(),
        "high_corr_pairs": high_corr_pairs,
        "mean_abs_corr": corr_matrix.abs().values[
            np.triu_indices_from(corr_matrix.values, k=1)
        ].mean(),
    }}
    return result
'''
        return SkillResult(
            success=True,
            data={
                "skill": "correlation",
                "factors": factors,
                "code": code.strip(),
                "description": f"相关性分析: {', '.join(factors)}",
            },
        )

    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return parameter schema"""
        return {
            "type": "object",
            "properties": {
                "factors": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "因子列表",
                    "default": ["factor1", "factor2"],
                },
            },
        }
