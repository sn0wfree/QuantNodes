# coding=utf-8
"""
Standard Backtest Prompt

Complete prompt for running standard backtests.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class BacktestPrompt:
    version: str = "1.0.0"
    name: str = "standard"
    description: str = "标准回测"
    created_at: str = "2025-05-14"
    updated_at: str = "2025-05-14"

    @property
    def prompt(self) -> str:
        return """你是一个量化回测专家，擅长运行标准回测。

## 回测流程
1. 准备历史价格数据
2. 创建策略实例
3. 创建经纪商实例
4. 运行回测获取结果

## 参数说明
- `start_date`: 回测开始日期 (YYYY-MM-DD)
- `end_date`: 回测结束日期 (YYYY-MM-DD)
- `initial_cash`: 初始资金 (默认 100000)
- `commission`: 手续费率 (默认 0.001)

## 约束
- 策略代码必须安全，使用 CodeSandbox 验证
- 必须创建 strategy, broker, quote_data 变量
- quote_data 必须包含 date, close 列

## 参考代码框架
```python
import pandas as pd
import numpy as np
from QuantNodes.backtest.strategy_node import MAStrategyNode
from QuantNodes.backtest.broker_node import SimulatedBrokerNode

strategy = MAStrategyNode(config={'short_window': 5, 'long_window': 20})
broker = SimulatedBrokerNode(config={'cash': 1000000, 'commission': 0.001})

quote_data = pd.read_csv('data.csv')
```
"""

    @property
    def required_params(self) -> List[str]:
        return ["pipeline_code", "start_date", "end_date"]

    @property
    def output_format(self) -> str:
        return "backtest_result"

    @property
    def validation_rules(self) -> Dict[str, Any]:
        return {
            "max_lines": 300,
            "allowed_imports": ["numpy", "pandas"],
            "forbidden_patterns": ["os.", "subprocess", "eval", "exec"],
            "required_variables": ["strategy", "broker", "quote_data"]
        }


STANDARD_BACKTEST_PROMPT = BacktestPrompt()
