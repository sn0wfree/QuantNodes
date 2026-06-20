# coding=utf-8
"""
Pairs Trading Strategy Prompt

Complete prompt for generating pairs trading strategies with reference code.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class StrategyPrompt:
    version: str = "1.0.0"
    name: str = "pairs_trading"
    description: str = "配对交易策略生成"
    created_at: str = "2025-05-14"
    updated_at: str = "2025-05-14"

    @property
    def prompt(self) -> str:
        return """你是一个量化策略专家，专注于配对交易策略。

## 策略逻辑
配对交易基于两只相关证券之间的价差会回归均值的假设。
核心思想：
1. 选择一对高度相关的证券
2. 当价差偏离均值时，做空高估的、做多低估的
3. 价差回归时平仓获利

## 参数说明
- `symbol1`: 第一个标的代码
- `symbol2`: 第二个标的代码
- `lookback`: 计算价差均值和标准差的回看窗口
- `entry_threshold`: 入场阈值 (标准差倍数)
- `exit_threshold`: 出场阈值

## 输出要求
生成完整的 Python 代码"""

    @property
    def required_params(self) -> List[str]:
        return ["symbol1", "symbol2", "lookback", "entry_threshold"]

    @property
    def output_format(self) -> str:
        return "python_code"

    @property
    def validation_rules(self) -> Dict[str, Any]:
        return {
            "max_lines": 600,
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

class PairsTradingStrategyNode(StrategyNode):
    """配对交易策略节点

    当两只股票的价差偏离均值时进行配对交易
    """
    def __init__(self, config: dict = None):
        self.lookback = config.get('lookback', 20) if config else 20
        self.entry_threshold = config.get('entry_threshold', 2.0) if config else 2.0
        self.exit_threshold = config.get('exit_threshold', 0.5) if config else 0.5

    def execute(self, data: pd.DataFrame) -> OrdersResult:
        spread = data['close1'] - data['close2']
        ma = spread.rolling(window=self.lookback).mean()
        std = spread.rolling(window=self.lookback).std()

        z_score = (spread - ma) / (std + 1e-8)

        signal = pd.Series(0, index=data.index)
        signal[z_score > self.entry_threshold] = -1
        signal[z_score < -self.entry_threshold] = 1
        signal[abs(z_score) < self.exit_threshold] = 0

        result = OrdersResult()
        result.signals = signal
        result.orders = []
        return result


strategy = PairsTradingStrategyNode(config={
    'lookback': 20,
    'entry_threshold': 2.0,
    'exit_threshold': 0.5
})

quote_data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=100),
    'close1': np.random.randn(100).cumsum() + 100,
    'close2': np.random.randn(100).cumsum() + 100
})
'''


PAIRS_TRADING_PROMPT = StrategyPrompt()
