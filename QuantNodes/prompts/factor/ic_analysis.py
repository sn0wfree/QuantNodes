# coding=utf-8
"""
IC Analysis Prompt

Complete prompt for IC (Information Coefficient) analysis.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class FactorPrompt:
    version: str = "1.0.0"
    name: str = "ic_analysis"
    description: str = "IC分析"
    created_at: str = "2025-05-14"
    updated_at: str = "2025-05-14"

    @property
    def prompt(self) -> str:
        return """你是一个因子分析专家，擅长IC分析。

## IC分析说明
IC (Information Coefficient) 衡量因子预测能力。
- IC > 0 表示因子与收益正相关
- IC < 0 表示因子与收益负相关
- |IC| 越大表示预测能力越强

## IC指标
- IC Mean: IC时间序列均值
- IC Std: IC时间序列标准差
- ICIR: IC Mean / IC Std (信息比)
- Rank IC Mean: 秩相关系数均值

## 参数说明
- `factor_code`: 计算因子的代码，必须将结果赋给 result 变量
- `start_date`: 分析开始日期
- `end_date`: 分析结束日期

## result 数据格式
result 必须是 Polars DataFrame，包含：
- date: 日期
- code: 标的代码
- factor_value: 因子值
- forward_return: 未来收益

## 参考代码
```python
import polars as pl

result = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    "code": ["A", "B", "A", "B"],
    "factor_value": [0.1, 0.2, 0.3, 0.4],
    "forward_return": [0.05, 0.03, 0.02, 0.01],
})
```
"""

    @property
    def required_params(self) -> List[str]:
        return ["factor_code", "start_date", "end_date"]

    @property
    def output_format(self) -> str:
        return "ic_analysis_result"

    @property
    def validation_rules(self) -> Dict[str, Any]:
        return {
            "max_lines": 200,
            "allowed_imports": ["polars", "numpy"],
            "forbidden_patterns": ["os.", "subprocess", "eval", "exec"],
            "required_variables": ["result"]
        }


IC_ANALYSIS_PROMPT = FactorPrompt()
