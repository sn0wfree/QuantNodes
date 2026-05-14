# coding=utf-8
"""
Trend Following Strategy Prompt

Complete prompt for generating trend following strategies with reference code.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class StrategyPrompt:
    version: str = "1.0.0"
    name: str = "trend_following"
    description: str = "趋势跟踪策略生成"
    created_at: str = "2025-05-14"
    updated_at: str = "2025-05-14"

    @property
    def prompt(self) -> str:
        return """你是一个量化策略专家，专注于趋势跟踪策略。

## 策略逻辑
趋势跟踪策略旨在捕捉大趋势，一旦趋势形成就持仓直到趋势反转。
核心思想：
1. 使用移动平均线判断趋势方向
2. 当短期均线上穿长期均线时买入 (金叉)
3. 当短期均线下穿长期均线时卖出 (死叉)

## 参数说明
- `symbol`: 交易标的代码
- `short_window`: 短期均线窗口
- `long_window`: 长期均线窗口

## 输出要求
生成完整的 Python 代码"""

    @property
    def required_params(self) -> List[str]:
        return ["symbol", "short_window", "long_window"]

    @property
    def output_format(self) -> str:
        return "python_code"

    @property
    def validation_rules(self) -> Dict[str, Any]:
        return {
            "max_lines": 500,
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

class TrendFollowingStrategyNode(StrategyNode):
    """趋势跟踪策略节点

    使用双均线交叉判断趋势
    """
    def __init__(self, config: dict = None):
        self.short_window = config.get('short_window', 10) if config else 10
        self.long_window = config.get('long_window', 30) if config else 30

    def execute(self, data: pd.DataFrame) -> OrdersResult:
        close = data['close']
        short_ma = close.rolling(window=self.short_window).mean()
        long_ma = close.rolling(window=self.long_window).mean()

        signal = pd.Series(0, index=data.index)
        signal[(short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))] = 1
        signal[(short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))] = -1

        result = OrdersResult()
        result.signals = signal
        result.orders = []
        return result


strategy = TrendFollowingStrategyNode(config={'short_window': 10, 'long_window': 30})

quote_data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=100),
    'close': np.random.randn(100).cumsum() + 100
})
'''


TREND_FOLLOWING_PROMPT = StrategyPrompt()