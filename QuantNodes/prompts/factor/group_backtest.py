# coding=utf-8
"""
Group Backtest Prompt

Complete prompt for group-based backtest analysis.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class FactorPrompt:
    version: str = "1.0.0"
    name: str = "group_backtest"
    description: str = "分组回测"
    created_at: str = "2025-05-14"
    updated_at: str = "2025-05-14"

    @property
    def prompt(self) -> str:
        return """你是一个因子分析专家，擅长分组回测。

## 分组回测说明
分组回测是检验因子有效性的经典方法：
1. 按因子值将股票分成N组
2. 持有各组一定时间
3. 计算各组收益率差异

## 参数说明
- `factor_code`: 计算因子的代码
- `num_groups`: 分组数量 (通常5组)
- `start_date`: 回测开始日期
- `end_date`: 回测结束日期

## result 数据格式
result 必须是 DataFrame (Polars 或 Pandas 均可)，包含：
- date: 日期
- code: 标的代码
- factor_value: 因子值
- forward_return: 未来收益

## 参考代码 (Polars)
```python
import polars as pl

result = pl.DataFrame({
    "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    "code": ["A", "B", "A", "B"],
    "factor_value": [0.1, 0.2, 0.3, 0.4],
    "forward_return": [0.05, 0.03, 0.02, 0.01],
})
```

## 参考代码 (Pandas)
```python
import pandas as pd

result = pd.DataFrame({
    "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
    "code": ["A", "B", "A", "B"],
    "factor_value": [0.1, 0.2, 0.3, 0.4],
    "forward_return": [0.05, 0.03, 0.02, 0.01],
})
```
"""

    @property
    def required_params(self) -> List[str]:
        return ["factor_code", "num_groups", "start_date", "end_date"]

    @property
    def output_format(self) -> str:
        return "group_backtest_result"

    @property
    def validation_rules(self) -> Dict[str, Any]:
        return {
            "max_lines": 200,
            "allowed_imports": ["polars", "numpy", "pandas"],
            "forbidden_patterns": ["os.", "subprocess", "eval", "exec"],
            "required_variables": ["result"]
        }


GROUP_BACKTEST_PROMPT = FactorPrompt()
