# coding=utf-8
"""
Correlation Analysis Prompt

Complete prompt for factor correlation analysis.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class FactorPrompt:
    version: str = "1.0.0"
    name: str = "correlation"
    description: str = "相关性分析"
    created_at: str = "2025-05-14"
    updated_at: str = "2025-05-14"

    @property
    def prompt(self) -> str:
        return """你是一个因子分析专家，擅长相关性分析。

## 相关性分析说明
分析因子之间的相关性，避免高度相关的因子同时使用。
- 因子相关性高：冗余，增加风险
- 因子相关性低：互补，分散风险

## 参数说明
- `factor_code`: 计算因子的代码
- `correlation_threshold`: 相关性阈值 (默认0.8)

## result 数据格式
result 必须是 Polars DataFrame，包含多列因子值
"""

    @property
    def required_params(self) -> List[str]:
        return ["factor_code", "correlation_threshold"]

    @property
    def output_format(self) -> str:
        return "correlation_result"

    @property
    def validation_rules(self) -> Dict[str, Any]:
        return {
            "max_lines": 200,
            "allowed_imports": ["polars", "numpy"],
            "forbidden_patterns": ["os.", "subprocess", "eval", "exec"],
            "required_variables": ["result"]
        }


CORRELATION_PROMPT = FactorPrompt()
