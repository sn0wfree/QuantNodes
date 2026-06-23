# coding=utf-8
"""
Factor-Based Backtest Prompt

Complete prompt for running factor-based backtests.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class BacktestPrompt:
    version: str = "1.0.0"
    name: str = "factor_based"
    description: str = "因子回测"
    created_at: str = "2025-05-14"
    updated_at: str = "2025-05-14"

    @property
    def prompt(self) -> str:
        return """你是一个量化回测专家，擅长运行因子回测。

## 因子回测流程
1. 计算因子值
2. 根据因子值排序分组
3. 计算各组收益率
4. 分析因子有效性

## 参数说明
- `factor_code`: 计算因子的代码，必须将结果赋给 result 变量
- `start_date`: 回测开始日期
- `end_date`: 回测结束日期
- `num_groups`: 分组数量 (默认5组)

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
        return ["factor_code", "start_date", "end_date", "num_groups"]

    @property
    def output_format(self) -> str:
        return "factor_backtest_result"

    @property
    def validation_rules(self) -> Dict[str, Any]:
        return {
            "max_lines": 300,
            "allowed_imports": ["polars", "numpy", "pandas"],
            "forbidden_patterns": ["os.", "subprocess", "eval", "exec"],
            "required_variables": ["result"]
        }


FACTOR_BACKTEST_PROMPT = BacktestPrompt()
