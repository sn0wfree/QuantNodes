# coding=utf-8
"""
Momentum Strategy Prompt

Complete prompt for generating momentum strategies with reference code.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class StrategyPrompt:
    version: str = "1.0.0"
    name: str = "momentum"
    description: str = "动量策略生成"
    created_at: str = "2025-05-14"
    updated_at: str = "2025-05-14"

    @property
    def prompt(self) -> str:
        return """你是一个量化策略专家，专注于动量策略。

## 策略逻辑
动量策略基于"强者恒强"的假设，过去表现良好的资产未来继续表现良好。
核心思想：
1. 计算一定时间窗口的收益率
2. 买入收益率最高的资产
3. 持有一定时间后重新平衡

## 参数说明
- `symbol`: 交易标的代码 (如 "BTC", "ETH", "AAPL")
- `window`: 计算动量的回看窗口 (如 20 表示过去20天)
- `threshold`: 触发交易的阈值 (如 0.05 表示5%)

## 输出要求
生成完整的 Python 代码，包含：
1. 导入必要的库 (pandas, numpy)
2. 定义策略函数 `momentum_strategy(data, window, threshold)`
3. 返回交易信号 (1=买入, -1=卖出, 0=持有)
4. 包含完整的 docstring

## 约束
- 代码必须安全，不能包含 os, subprocess, eval 等危险操作
- 必须创建名为 `strategy` 的 StrategyNode 变量
- 必须创建名为 `quote_data` 的 DataFrame 变量
- 使用 QuantNodes 回测框架

## 参考代码框架
```python
import pandas as pd
import numpy as np
from QuantNodes.backtest.strategy_node import StrategyNode, OrdersResult

class MomentumStrategyNode(StrategyNode):
    def __init__(self, config: dict = None):
        self.window = config.get('window', 20)
        self.threshold = config.get('threshold', 0.05)

    def execute(self, data: pd.DataFrame) -> OrdersResult:
        close = data['close']
        returns = close.pct_change(self.window)
        signal = (returns > self.threshold).astype(int)
        signal[returns < -self.threshold] = -1

        result = OrdersResult()
        result.signals = signal
        result.orders = []
        return result

# 创建策略实例
strategy = MomentumStrategyNode(config={
    'window': 20,
    'threshold': 0.05
})

# 示例数据
quote_data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=100),
    'close': np.random.randn(100).cumsum() + 100
})
```
"""

    @property
    def required_params(self) -> List[str]:
        return ["symbol", "window", "threshold"]

    @property
    def output_format(self) -> str:
        return "python_code"

    @property
    def validation_rules(self) -> Dict[str, Any]:
        return {
            "max_lines": 500,
            "allowed_imports": ["numpy", "pandas", "talib"],
            "forbidden_patterns": ["os.", "subprocess", "eval", "exec", "open("],
            "required_variables": ["strategy", "quote_data"],
            "required_classes": ["StrategyNode"]
        }

    @property
    def example_code(self) -> str:
        return '''import pandas as pd
import numpy as np
from QuantNodes.backtest.strategy_node import StrategyNode, OrdersResult

class MomentumStrategyNode(StrategyNode):
    """动量策略节点

    基于过去N天收益率计算动量信号
    """
    def __init__(self, config: dict = None):
        self.window = config.get('window', 20) if config else 20
        self.threshold = config.get('threshold', 0.05) if config else 0.05

    def execute(self, data: pd.DataFrame) -> OrdersResult:
        close = data['close']
        returns = close.pct_change(self.window)

        signal = pd.Series(0, index=data.index)
        signal[returns > self.threshold] = 1
        signal[returns < -self.threshold] = -1

        result = OrdersResult()
        result.signals = signal
        result.orders = self._signals_to_orders(signal, data)
        return result

    def _signals_to_orders(self, signals, data):
        orders = []
        position = 0
        for i, (idx, row) in enumerate(data.iterrows()):
            sig = signals.iloc[i]
            if sig != position:
                if sig == 1:
                    orders.append({
                        'date': idx, 'symbol': 'BTC',
                        'action': 'buy', 'price': row['close'],
                    })
                elif sig == -1:
                    orders.append({
                        'date': idx, 'symbol': 'BTC',
                        'action': 'sell', 'price': row['close'],
                    })
                position = sig
        return orders


strategy = MomentumStrategyNode(config={'window': 20, 'threshold': 0.05})

quote_data = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=100),
    'close': np.random.randn(100).cumsum() + 100
})
'''


MOMENTUM_PROMPT = StrategyPrompt()
