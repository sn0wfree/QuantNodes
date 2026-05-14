# coding=utf-8
"""
Market Neutral Strategy Prompt

Complete prompt for generating market neutral strategies with reference code.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class StrategyPrompt:
    version: str = "1.0.0"
    name: str = "market_neutral"
    description: str = "市场中性策略生成"
    created_at: str = "2025-05-14"
    updated_at: str = "2025-05-14"

    @property
    def prompt(self) -> str:
        return """你是一个量化策略专家，专注于市场中性策略。

## 策略逻辑
市场中性策略旨在消除市场系统性风险，获取alpha收益。
核心思想：
1. 同时持有多头和空头仓位
2. 确保投资组合对市场涨跌的暴露为0
3. 通过选股能力获取超额收益

## 参数说明
- `symbols`: 标的列表
- `lookback`: 计算因子和风险的回看窗口
- `long_short_ratio`: 多空仓比例 (如 0.5 表示一半多头一半空头)
- `risk_factor`: 对冲用的风险因子 (如 "market")

## 输出要求
生成完整的 Python 代码"""

    @property
    def required_params(self) -> List[str]:
        return ["symbols", "lookback", "long_short_ratio"]

    @property
    def output_format(self) -> str:
        return "python_code"

    @property
    def validation_rules(self) -> Dict[str, Any]:
        return {
            "max_lines": 700,
            "allowed_imports": ["numpy", "pandas", "talib"],
            "forbidden_patterns": ["os.", "subprocess", "eval", "exec"],
            "required_variables": ["strategy", "quote_data"],
            "required_classes": ["StrategyNode"]
        }

    @property
    def example_code(self) -> str:
        return '''import pandas as pd
import numpy as np
from QuantNodes.backtest.strategy_node import StrategyNode, OrdersResult

class MarketNeutralStrategyNode(StrategyNode):
    """市场中性策略节点

    通过多空对冲消除市场风险
    """
    def __init__(self, config: dict = None):
        self.lookback = config.get('lookback', 20) if config else 20
        self.long_short_ratio = config.get('long_short_ratio', 0.5) if config else 0.5

    def execute(self, data: pd.DataFrame) -> OrdersResult:
        returns = data.pct_change()

        signal = pd.Series(0, index=data.index)

        result = OrdersResult()
        result.signals = signal
        result.orders = []
        return result


strategy = MarketNeutralStrategyNode(config={
    'lookback': 20,
    'long_short_ratio': 0.5
})

quote_data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=100),
    'close': np.random.randn(100).cumsum() + 100
})
'''


MARKET_NEUTRAL_PROMPT = StrategyPrompt()