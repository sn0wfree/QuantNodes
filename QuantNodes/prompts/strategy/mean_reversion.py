# coding=utf-8
"""
Mean Reversion Strategy Prompt

Complete prompt for generating mean reversion strategies with reference code.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class StrategyPrompt:
    version: str = "1.0.0"
    name: str = "mean_reversion"
    description: str = "均值回归策略生成"
    created_at: str = "2025-05-14"
    updated_at: str = "2025-05-14"

    @property
    def prompt(self) -> str:
        return """你是一个量化策略专家，专注于均值回归策略。

## 策略逻辑
均值回归策略基于"价格围绕价值波动"的假设，当价格偏离均值时迟早会回归。
核心思想：
1. 计算移动平均线作为均值
2. 当价格偏离均值超过阈值时做相反方向的交易
3. 预期价格会回归到均值

## 参数说明
- `symbol`: 交易标的代码
- `window`: 计算均值的窗口大小
- `std_threshold`: 标准差阈值 (如 2.0 表示2个标准差)

## 输出要求
生成完整的 Python 代码，包含：
1. 导入必要的库
2. 定义均值回归策略类继承 StrategyNode
3. 返回交易信号

## 约束
- 代码必须安全，不能包含危险操作
- 必须创建名为 `strategy` 的 StrategyNode 变量
- 必须创建名为 `quote_data` 的 DataFrame 变量
"""

    @property
    def required_params(self) -> List[str]:
        return ["symbol", "window", "std_threshold"]

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

class MeanReversionStrategyNode(StrategyNode):
    """均值回归策略节点

    当价格偏离均值超过阈值时反向交易
    """
    def __init__(self, config: dict = None):
        self.window = config.get('window', 20) if config else 20
        self.std_threshold = config.get('std_threshold', 2.0) if config else 2.0

    def execute(self, data: pd.DataFrame) -> OrdersResult:
        close = data['close']
        ma = close.rolling(window=self.window).mean()
        std = close.rolling(window=self.window).std()

        z_score = (close - ma) / (std + 1e-8)

        signal = pd.Series(0, index=data.index)
        signal[z_score < -self.std_threshold] = 1
        signal[z_score > self.std_threshold] = -1

        result = OrdersResult()
        result.signals = signal
        result.orders = []
        return result


strategy = MeanReversionStrategyNode(config={'window': 20, 'std_threshold': 2.0})

quote_data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=100),
    'close': np.random.randn(100).cumsum() + 100
})
'''


MEAN_REVERSION_PROMPT = StrategyPrompt()